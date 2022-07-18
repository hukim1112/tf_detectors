import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, MaxPooling2D, BatchNormalization, ReLU, Add
from tensorflow.keras import Sequential, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Input
from .Decoder import SSDPredictions
from data.anchor import generate_retina_boxes_v2, generate_default_boxes

class SSD300(Model):
    """ Class for SSD model
    Attributes:
        num_classes: number of classes
    """
    def __init__(self, num_classes, **kwargs):
        super(SSD300, self).__init__()

        self.num_classes = num_classes

        conf_head_layers = [
            Conv2D(4 * (num_classes+1), kernel_size=3, padding='same'),  # for 4th block
            Conv2D(6 * (num_classes+1), kernel_size=3, padding='same'),  # for 7th block
            Conv2D(6 * (num_classes+1), kernel_size=3, padding='same'),  # for 8th block
            Conv2D(6 * (num_classes+1), kernel_size=3, padding='same'),  # for 9th block
            Conv2D(4 * (num_classes+1), kernel_size=3, padding='same'),  # for 10th block
            Conv2D(4 * (num_classes+1), kernel_size=1)  # for 11th block
        ]
        loc_head_layers = [
            Conv2D(4 * 4, kernel_size=3, padding='same'), # for 4th block
            Conv2D(6 * 4, kernel_size=3, padding='same'), # for 7th block
            Conv2D(6 * 4, kernel_size=3, padding='same'), # for 8th block
            Conv2D(6 * 4, kernel_size=3, padding='same'), # for 9th block
            Conv2D(4 * 4, kernel_size=3, padding='same'), # for 10th block
            Conv2D(4 * 4, kernel_size=1) # for 11th block

        ]
        input = Input(shape=(300,300,3))
        x = tf.keras.layers.Lambda(tf.keras.applications.vgg16.preprocess_input,
                                     name='preprocessing')(input)
        features = self.multiscale_feature_extractor()(x)
        confs = []; locs = [];
        for feature, conf_head, loc_head in zip(features, conf_head_layers, loc_head_layers):
            conf = conf_head(feature)
            B,H,W,C = conf.shape
            confs.append(tf.reshape(conf, [-1, H*W*(C//(num_classes+1)), (num_classes+1)]))
            loc = loc_head(feature)
            B,H,W,C = loc.shape
            locs.append(tf.reshape(loc, [-1, H*W*(C//4), 4]))
        self.net = Model(input, [tf.concat(confs, axis=-2), tf.concat(locs, axis=-2)])
        self.loss_fn = SSDLoss(len(self.anchors()))
        self.l2_reg = 0.0005
        self.metric_objects = {"regression_loss" : tf.keras.metrics.Mean(name='regression_loss'),
                        "classification_loss" : tf.keras.metrics.Mean(name='classification_loss'),
                        "objective_loss" : tf.keras.metrics.Mean(name='objective_loss'),
                        "l2_loss" : tf.keras.metrics.Mean(name='l2_loss'),
                        "loss" : tf.keras.metrics.Mean(name='loss')}

    def anchors(self):
        anchor_param = {"ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                                   "scales": [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075],
                                   "fm_sizes": [38, 19, 10, 5, 3, 1],
                                   "image_size": 300} #anchor parameters
        return generate_default_boxes(anchor_param)

    def inference(self, confidence_threshold):
        image = tf.keras.Input(shape=[None, None, 3], name="image")
        predictions = self(image, training=False)
        detections = SSDPredictions(anchors=self.anchors(),
                              num_classes=self.num_classes, confidence_threshold=confidence_threshold)(image, predictions)
        inference_model = tf.keras.Model(inputs=image, outputs=detections)
        return inference_model


    @property
    def metrics(self):
        #metric objects will be reset after each epoch.
        return list(self.metric_objects.values())

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

    def call(self, x):
        """ The forward pass
        Args:
            x: the input image
        Returns: concatenatation of confs and locs
            confs: list of outputs of all classification heads
            locs: list of outputs of all regression heads
        """
        confs, locs = self.net(x)
        return tf.concat([locs,confs], axis=-1) # [None, 8732, 4+(1+num_classses)]

    def multiscale_feature_extractor(self):
        vgg = VGG16(include_top=False, weights='imagenet', input_shape=(300,300,3))

        features = []
        input = tf.keras.Input(shape=(300,300,3))

        x = input
        for layer in vgg.layers[:-1]:
            if "pool" in layer.name:
                layer = MaxPooling2D(padding="same")
            x = layer(x)
            if layer.name == "block4_conv3":
                features.append(x) #block4 (B,38,38,512)

        x = Sequential([# Difference from original VGG16:
                    # 5th maxpool layer has kernel size = 3 and stride = 1
                    MaxPooling2D(3, 1, padding='same'),
                    # atrous conv2d for 6th block
                    Conv2D(1024, 3, padding='same',
                                  dilation_rate=6, activation='relu'),
                    Conv2D(1024, 1, padding='same', activation='relu'),
                ])(x)
        features.append(x) #block7 (B,19,19,1024)

        x = Sequential([
            Conv2D(256, 1, activation='relu'),
            Conv2D(512, 3, strides=2, padding='same',
                          activation='relu'),
        ])(x)
        features.append(x) #block8 (B,10,10,512)

        x = Sequential([
            Conv2D(128, 1, activation='relu'),
            Conv2D(256, 3, strides=2, padding='same',
                          activation='relu'),
        ])(x)
        features.append(x) #block9 (B,5,5,256)

        x = Sequential([
            Conv2D(128, 1, activation='relu'),
            Conv2D(256, 3, strides=2, padding='same',
                          activation='relu'),
        ])(x)
        features.append(x) #block10 (B,3,3,256)

        x = Sequential([
            Conv2D(128, 1, activation='relu'),
            Conv2D(256, 3, strides=2,
                          activation='relu'),
        ])(x)
        features.append(x) #block11 (B,1,1,256)
        return Model(input, features)



class EfficientSSD300(Model):
    """ Class for SSD model
    Attributes:
        num_classes: number of classes
    """
    def __init__(self, num_classes, **kwargs):
        super(EfficientSSD300, self).__init__()
        self.num_classes = num_classes
        self.net = self.multiscale_feature_extractor()
        self.loss_fn = SSDLoss(len(self.anchors()))
        self.l2_reg = 0.0005
        self.metric_objects = {"regression_loss" : tf.keras.metrics.Mean(name='regression_loss'),
                        "classification_loss" : tf.keras.metrics.Mean(name='classification_loss'),
                        "objective_loss" : tf.keras.metrics.Mean(name='objective_loss'),
                        "l2_loss" : tf.keras.metrics.Mean(name='l2_loss'),
                        "loss" : tf.keras.metrics.Mean(name='loss')}

    def anchors(self):
        anchor_param = {"ratios": [0.5, 1, 2],
                        "scales": [1.0, 1.25,1.58],
                        "dimensions" : [16, 32, 64, 128, 256, 512], #anchor base dimensions
                        "fm_sizes": [38, 19, 10, 5, 3, 1],
                        "image_size": 300} #anchor parameters
        return generate_retina_boxes_v2(anchor_param)

    def inference(self, confidence_threshold):
        image = tf.keras.Input(shape=[None, None, 3], name="image")
        predictions = self(image, training=False)
        detections = SSDPredictions(anchors=self.anchors(),
                              num_classes=self.num_classes, confidence_threshold=confidence_threshold)(image, predictions)
        inference_model = tf.keras.Model(inputs=image, outputs=detections)
        return inference_model

    @property
    def metrics(self):
        #metric objects will be reset after each epoch.
        return list(self.metric_objects.values())

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

    def call(self, x):
        """ The forward pass
        Args:
            x: the input image
        Returns: concatenatation of confs and locs
            confs: list of outputs of all classification heads
            locs: list of outputs of all regression heads
        """
        confs, locs = self.net(x)
        return tf.concat([locs,confs], axis=-1) # [None, 8732, 4+(1+num_classses)]

    def multiscale_feature_extractor(self):
        from tensorflow.keras.applications.efficientnet import preprocess_input
        from tensorflow.keras.applications import EfficientNetB0 # CNN architecture
        from .layers import dconv, dconv_downsampled, classification_head, bbox_regression_head

        # feature extractor
        backbone = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(300,300,3))
        fmap38x38 = backbone.get_layer("block4a_expand_activation") # 38x38x240
        fmap19x19 = backbone.get_layer("block5c_add") # 19x19x112
        efficient_backbone = tf.keras.Model(inputs=[backbone.input], outputs=[fmap38x38.output, fmap19x19.output], name="efficient_backbone")

        input_layer = tf.keras.Input(shape=(300,300,3))
        x = tf.keras.layers.Lambda(preprocess_input,
                                   name='preprocessing',
                                   input_shape=(300,300,3))(input_layer)

        P1, P2 = efficient_backbone(x)

        x = P2
        y = dconv(112, 112)(x)
        out = Add()([x,y]) #residual connection
        x = out
        y1 = dconv_downsampled(112,224)(x)
        y2 = Conv2D(224, 1, strides=2)(x) #downsampled
        y2 = BatchNormalization()(y2)
        y = Add()([y1,y2]) #residual connection
        P3 = ReLU()(y)
        # 19x19x112 => 10x10x224

        x = P3
        y = dconv(224, 224)(x)
        out = Add()([x,y]) #residual connection
        x = out
        y1 = dconv_downsampled(224,448)(x)
        y2 = Conv2D(448, 1, strides=2)(x) #downsampled
        y2 = BatchNormalization()(y2)
        y = Add()([y1,y2]) #residual connection
        P4 = ReLU()(y)
        # 10x10x224 => 5x5x448

        x = P4
        y = dconv(448, 448)(x)
        out = Add()([x,y]) #residual connection
        x = out
        y1 = dconv_downsampled(448,896)(x)
        y2 = Conv2D(896, 1, strides=2)(x) #downsampled
        y2 = BatchNormalization()(y2)
        y = Add()([y1,y2]) #residual connection
        P5 = ReLU()(y)
        # 5x5x448 => 3x3x896

        x = P5
        y = dconv(896, 896)(x)
        out = Add()([x,y]) #residual connection
        x = out

        in_depth=896
        out_depth=896
        y1 = Conv2D(in_depth, 1, padding='same')(x)
        y1 = BatchNormalization()(y1)
        y1 = ReLU()(y1)
        y1 = DepthwiseConv2D(3)(y1)
        y1 = BatchNormalization()(y1)
        y1 = ReLU()(y1)
        y1 = Conv2D(out_depth, 1, padding='same')(y1)
        y1 = BatchNormalization()(y1)


        y2 = Conv2D(896, 3)(x) #downsampled
        y2 = BatchNormalization()(y2)
        y = Add()([y1,y2]) #residual connection

        P6 = ReLU()(y)
        # 3x3x896 => 1x1x896

        channel_depth = 256
        num_classes = self.num_classes+1
        coords = 4
        k_anchors = 9

        #conf header
        confs = []
        fmap_size = 38
        conf = classification_head(P1, channel_depth, fmap_size, num_classes, k_anchors)
        confs.append(conf)

        fmap_size = 19
        conf = classification_head(P2, channel_depth, fmap_size, num_classes, k_anchors)
        confs.append(conf)

        fmap_size = 10
        conf = classification_head(P3, channel_depth, fmap_size, num_classes, k_anchors)
        confs.append(conf)

        fmap_size = 5
        conf = classification_head(P4, channel_depth, fmap_size, num_classes, k_anchors)
        confs.append(conf)

        fmap_size = 3
        conf = classification_head(P5, channel_depth, fmap_size, num_classes, k_anchors)
        confs.append(conf)

        fmap_size = 1
        conf = classification_head(P6, channel_depth, fmap_size, num_classes, k_anchors)
        confs.append(conf)

        confs = tf.concat(confs, axis=-2)

        #conf header
        locs = []
        fmap_size = 38
        loc = bbox_regression_head(P1, channel_depth, fmap_size, coords, k_anchors)
        locs.append(loc)

        fmap_size = 19
        loc = bbox_regression_head(P2, channel_depth, fmap_size, coords, k_anchors)
        locs.append(loc)

        fmap_size = 10
        loc = bbox_regression_head(P3, channel_depth, fmap_size, coords, k_anchors)
        locs.append(loc)

        fmap_size = 5
        loc = bbox_regression_head(P4, channel_depth, fmap_size, coords, k_anchors)
        locs.append(loc)

        fmap_size = 3
        loc = bbox_regression_head(P5, channel_depth, fmap_size, coords, k_anchors)
        locs.append(loc)

        fmap_size = 1
        loc = bbox_regression_head(P6, channel_depth, fmap_size, coords, k_anchors)
        locs.append(loc)

        locs = tf.concat(locs, axis=-2)

        return tf.keras.Model(inputs=input_layer, outputs=[confs, locs])


def hard_negative_mining(loss, gt_confs, neg_ratio):
    """ Hard negative mining algorithm
        to pick up negative examples for back-propagation
        base on classification loss values
    Args:
        loss: list of classification losses of all default boxes (B, num_default)
        gt_confs: classification targets (B, num_default)
        neg_ratio: negative / positive ratio
    Returns:
        conf_loss: classification loss
        loc_loss: regression loss
    """
    # loss: B x N
    # gt_confs: B x N
    pos_idx = gt_confs > 0
    num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.int32), axis=1)
    num_neg = num_pos * neg_ratio

    rank = tf.argsort(loss, axis=1, direction='DESCENDING')
    rank = tf.argsort(rank, axis=1)
    neg_idx = rank < tf.expand_dims(num_neg, 1)

    return pos_idx, neg_idx


class SSDLoss(tf.losses.Loss):
    """Wrapper to combine both the losses"""

    def __init__(self, num_anchors=8732, neg_ratio=9):
        super(SSDLoss, self).__init__(reduction="none", name="SSDLoss")
        self.num_anchors = num_anchors
        self.neg_ratio = neg_ratio

    def call(self, y_true, y_pred):
        """ Compute losses for SSD
            regression loss: smooth L1
            classification loss: cross entropy
        Args:
            y_true: classification+regression targets (B, num_default, 5)
            y_pred: outputs of classification+regression heads (B, num_default, 4+num_classes+1)
        Returns:
            loss: classification loss + regression loss
        """
        gt_locs = y_true[:, :, :4]
        gt_confs = y_true[:, :, 4:]
        gt_confs = tf.squeeze(gt_confs, axis=-1)
        locs = y_pred[:, :, :4]
        confs = y_pred[:, :, 4:]

        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        # compute classification losses
        # without reduction
        temp_loss = cross_entropy(
            gt_confs, confs)
        pos_idx, neg_idx = hard_negative_mining(
            temp_loss, gt_confs, self.neg_ratio)

        # classification loss will consist of positive and negative examples

        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='sum')
        smooth_l1_loss = tf.keras.losses.Huber(reduction='sum')
        pos_idx = tf.reshape(pos_idx, [-1, self.num_anchors])
        neg_idx = tf.reshape(neg_idx, [-1, self.num_anchors])
        conf_loss = cross_entropy(
            gt_confs[tf.math.logical_or(pos_idx, neg_idx)],
            confs[tf.math.logical_or(pos_idx, neg_idx)])
        # regression loss only consist of positive examples

        loc_loss = smooth_l1_loss(
            # tf.boolean_mask(gt_locs, pos_idx),
            # tf.boolean_mask(locs, pos_idx))
            gt_locs[pos_idx],
            locs[pos_idx])
        num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.float32))
        conf_loss = tf.math.truediv(conf_loss, num_pos)
        loc_loss = tf.math.truediv(loc_loss, num_pos)
        objective_losses = tf.stack([tf.reduce_mean(conf_loss), tf.reduce_mean(loc_loss)])
        return objective_losses
