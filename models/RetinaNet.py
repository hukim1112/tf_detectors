import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, UpSampling2D
from tensorflow.keras import Sequential, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Input
from .Decoder import RetinaPredictions
from data.retina import resize_and_pad_image

class RetinaNet(Model):
    """A subclassed Keras model implementing the RetinaNet architecture.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, num_classes, **kwargs):
        super(RetinaNet, self).__init__(name="RetinaNet", **kwargs)
        self.InputLayer = tf.keras.layers.Lambda(tf.keras.applications.resnet.preprocess_input,
                                     name='preprocessing')
        backbone = get_backbone()
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")
        self.loss_fn = RetinaNetLoss(num_classes)
        self.l2_reg = 0.00005
        self.metric_objects = {"regression_loss" : tf.keras.metrics.Mean(name='regression_loss'),
                        "classification_loss" : tf.keras.metrics.Mean(name='classification_loss'),
                        "objective_loss" : tf.keras.metrics.Mean(name='objective_loss'),
                        "l2_loss" : tf.keras.metrics.Mean(name='l2_loss'),
                        "loss" : tf.keras.metrics.Mean(name='loss')}
    @property
    def metrics(self):
        #metric objects will be reset after each epoch.
        return list(self.metric_objects.values())

    def inference(self, confidence_threshold):
        image = tf.keras.Input(shape=[None, None, 3], name="image")
        image = resize_and_pad_image(image)
        predictions = self(image, training=False)
        detections = RetinaPredictions(confidence_threshold=confidence_threshold)(image, predictions)
        inference_model = tf.keras.Model(inputs=image, outputs=detections)
        return inference_model

    def train_step(self, data):
        x, y_true= data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            objective_losses = self.loss_fn(y_true, y_pred)
            l2_loss = [tf.nn.l2_loss(t) for t in self.trainable_variables]
            l2_loss = self.l2_reg * tf.math.reduce_sum(l2_loss)
            loss = tf.reduce_sum(objective_losses) + l2_loss
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.metric_objects["regression_loss"](objective_losses[0])
        self.metric_objects["classification_loss"](objective_losses[1])
        self.metric_objects["objective_loss"](tf.reduce_sum(objective_losses))
        self.metric_objects["l2_loss"](l2_loss)
        self.metric_objects["loss"](loss)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y_true= data
        y_pred = self(x, training=True)
        objective_losses = self.loss_fn(y_true, y_pred)
        loss = tf.reduce_sum(objective_losses)
        # Update metrics (includes the metric that tracks the loss)
        self.metric_objects["regression_loss"](objective_losses[0])
        self.metric_objects["classification_loss"](objective_losses[1])
        self.metric_objects["objective_loss"](tf.reduce_sum(objective_losses))
        self.metric_objects["loss"](loss)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def call(self, image, training=False):
        preprecessed_image = image
        preprecessed_image = self.InputLayer(image)
        features = self.fpn(preprecessed_image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)


def build_head(output_filters, bias_init):
    """Builds the class/box predictions head.

    Arguments:
      output_filters: Number of convolution filters in the final layer.
      bias_init: Bias Initializer for the final convolution layer.

    Returns:
      A keras sequential model representing either the classification
        or the box regression head depending on `output_filters`.
    """
    head = Sequential([Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(Conv2D(256, 3, padding="same", kernel_initializer=kernel_init))
        head.add(ReLU())
    head.add(Conv2D(output_filters, 3, 1, padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init))
    return head

class FeaturePyramid(tf.keras.layers.Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, backbone=None, **kwargs):
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_c3_1x1 = Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = Conv2D(256, 3, 2, "same")
        self.upsample_2x = UpSampling2D(2)

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        return p3_output, p4_output, p5_output, p6_output, p7_output

def get_backbone():
    """Builds ResNet50 with pre-trained imagenet weights"""
    backbone = ResNet50(
        include_top=False, input_shape=[None, None, 3]
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return Model(inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )

class RetinaNetBoxLoss(tf.losses.Loss):
    """Implements Smooth L1 loss"""

    def __init__(self, delta):
        super(RetinaNetBoxLoss, self).__init__(
            reduction="none", name="RetinaNetBoxLoss"
        )
        self._delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetClassificationLoss(tf.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha, gamma):
        super(RetinaNetClassificationLoss, self).__init__(
            reduction="none", name="RetinaNetClassificationLoss"
        )
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), 1-probs, probs)
        tf.clip_by_value(pt, tf.keras.backend.epsilon(), 1.0)
        loss = alpha * tf.pow(pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetLoss(tf.losses.Loss):
    """Wrapper to combine both the losses"""

    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0, delta=1.0):
        super(RetinaNetLoss, self).__init__(reduction="none", name="RetinaNetLoss")
        self._clf_loss = RetinaNetClassificationLoss(alpha, gamma)
        self._box_loss = RetinaNetBoxLoss(delta)
        self._num_classes = num_classes

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]
        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=tf.float32,
        )
        cls_predictions = y_pred[:, :, 4:]
        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)
        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions)
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        objective_losses = tf.stack([tf.reduce_mean(clf_loss), tf.reduce_mean(box_loss)])
        return objective_losses
