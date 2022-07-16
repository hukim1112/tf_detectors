import random, os
from pycocotools.coco import COCO
from .box_utils import compute_target, transform_corner_to_center, transform_center_to_corner, compute_iou
from .anchor import RetinaNetAnchorBox
import tensorflow as tf
import cv2
import numpy as np

class RetinaDataset():
    def __init__(self, image_path, annotation_path, transform=None, target_transform=None, num_examples=None, shuffle=True):
        self.image_path = image_path # path to images
        self.coco = COCO(annotation_path) #path to .json file
        self.data_shuffle = shuffle # whether shuffle your data randomly or not
        self.transform = transform #image transform function.
        self.target_transform = target_transform #bbox transform function.

        image_ids = self.coco.getImgIds()
        self.image_ids = self.filter_image_id(image_ids)
        self.cat_ids = self.coco.getCatIds()

        classes, labels, coco_labels, coco_labels_inverse = self.coco_category_to_class_id()
        self.classes = classes                          # "name" to "class id"
        self.labels = labels                            # "class id" to "name"
        self.coco_labels = coco_labels                  # "class_id" to "coco category id"
        self.coco_labels_inverse = coco_labels_inverse  # "coco category id" to "class id"
        if num_examples is not None:
            self.image_ids = self.image_ids[:num_examples]
        if shuffle:
            random.shuffle(self.image_ids)
        self.label_encoder = LabelEncoder()

    def __len__(self):
        return len(self.ids)

    def load_tfds(self, batch_size=None, trainable_form=True):
        autotune = tf.data.AUTOTUNE
        tfds = tf.data.Dataset.from_tensor_slices(self.image_ids)
        if self.data_shuffle:
            tfds = tfds.shuffle(128)
        tfds = tfds.map(lambda image_id: tf.py_function(func=self.get_item, inp=[image_id],
                        Tout=[tf.float32, tf.float32, tf.int32]),
                        num_parallel_calls=autotune)
        if trainable_form:
            tfds = tfds.padded_batch(
                batch_size=batch_size, padded_shapes=([None,None,None], [None, None], [None]),
                padding_values=(0.0, 1e-8, -1), drop_remainder=True)
            tfds = tfds.map(self.label_encoder.encode_batch, num_parallel_calls=autotune)
            tfds = tfds.apply(tf.data.experimental.ignore_errors())
            tfds = tfds.prefetch(autotune)
        return tfds

    def get_item(self, image_id):
        image_id = int(image_id)
        image, (height, width) = self.get_image(image_id)
        gt_labels, gt_boxes  = self.get_labels(image_id)
        gt_boxes = list(map(lambda box : (box[0]/width, box[1]/height, box[2]/width, box[3]/height), gt_boxes))
        gt_boxes = np.array(gt_boxes, np.float32); gt_labels = np.array(gt_labels, np.float32);
        image, gt_boxes, gt_labels = self.preprocess_data(image, gt_boxes, gt_labels)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            gt_boxes = self.target_transform(gt_boxes)
        return image, gt_boxes, gt_labels

    def get_image(self, image_id):
        image_info = self.coco.loadImgs(image_id)[0]
        filename = image_info['file_name']
        original_size = (int(image_info['height']), int(image_info['width']))
        path = os.path.join(self.image_path, filename)
        image = cv2.imread(path)[:,:,::-1]
        return image, original_size

    def get_labels(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids, iscrowd=None)
        annotions = self.coco.loadAnns(ids=ann_ids)
        boxes = []
        labels = []
        for ann in annotions:
            x,y,w,h = ann["bbox"]
            xmin = x
            ymin = y
            xmax = (x+w)
            ymax = (y+h)
            box = [xmin, ymin, xmax, ymax]
            category = ann["category_id"]
            boxes.append(box)
            labels.append(self.coco_labels_inverse[category])
        return labels, boxes

    def filter_image_id(self, image_ids):
        filtered = []
        for image_id in image_ids:
            available = True
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            #각 이미지는 foreground box가 하나 이상있어야 한다.
            if len(ann_ids)==0:
                available = False
                continue
            #모든 바운딩 박스의 크기는 0보다 커야 한다.
            annotations = self.coco.loadAnns(ann_ids)
            for ann in annotations:
                x,y,w,h = ann["bbox"]
                if w*h <=0:
                    available = False
            if available:
                filtered.append(image_id)
        return filtered

    def coco_category_to_class_id(self):
        categories = self.coco.loadCats(ids=self.cat_ids)
        categories.sort(key=lambda x: x['id'])
        classes             = {} # "name" to "class id"
        coco_labels         = {} # "class_id" to "coco category id"
        coco_labels_inverse = {} # "coco category id" to "class id"
        for c in categories:
            coco_labels[len(classes)] = c['id']
            coco_labels_inverse[c['id']] = len(classes)
            classes[c['name']] = len(classes)
        # also load the reverse (label -> name)
        labels = {}
        for key, value in classes.items():
            labels[value] = key
        return classes, labels, coco_labels, coco_labels_inverse

    def preprocess_data(self, image, bbox, label):
        """Applies preprocessing step to a single sample

        Arguments:
          sample: A dict representing a single training sample.

        Returns:
          image: Resized and padded image with random horizontal flipping applied.
          bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
            of the format `[x, y, width, height]`.
          class_id: An tensor representing the class id of the objects, having
            shape `(num_objects,)`.
        """

        image, bbox = random_flip_horizontal(image, tf.constant(bbox, tf.float32))
        image, image_shape, _ = resize_and_pad_image(tf.cast(image, tf.float32))
        bbox = tf.stack(
            [
                bbox[:, 0] * image_shape[1],
                bbox[:, 1] * image_shape[0],
                bbox[:, 2] * image_shape[1],
                bbox[:, 3] * image_shape[0],
            ],
            axis=-1,
        )
        return image, bbox, label


def random_flip_horizontal(image, boxes):
    """Flips image and boxes horizontally with 50% chance

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.

    Returns:
      Randomly flipped image and boxes
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes


def resize_and_pad_image(
    image, min_side=800.0, max_side=1333.0, jitter=[640, 800], stride=128.0
):
    """Resizes and pads image while preserving aspect ratio.

    1. Resizes images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image
      with longer side equal to `max_side`
    3. Pad with zeros on right and bottom to make the image shape divisible by
    `stride`

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      min_side: The shorter side of the image is resized to this value, if
        `jitter` is set to None.
      max_side: If the longer side of the image exceeds this value after
        resizing, the image is resized such that the longer side now equals to
        this value.
      jitter: A list of floats containing minimum and maximum size for scale
        jittering. If available, the shorter side of the image will be
        resized to a random value in this range.
      stride: The stride of the smallest feature map in the feature pyramid.
        Can be calculated using `image_size / feature_map_size`.

    Returns:
      image: Resized and padded image.
      image_shape: Shape of the image before padding.
      ratio: The scaling factor used to resize the image
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio

class LabelEncoder:
    """Transforms the raw labels into targets for training.

    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.

    Attributes:
      anchor_box: Anchor box generator to encode the bounding boxes.
      box_variance: The scaling factors used to scale the bounding box targets.
    """

    def __init__(self):
        self._anchor_box = RetinaNetAnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _match_anchor_boxes(
        self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4
    ):
        """Matches ground truth boxes to anchor boxes based on IOU.

        1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
          to get a `(M, N)` shaped matrix.
        2. The ground truth box with the maximum IOU in each row is assigned to
          the anchor box provided the IOU is greater than `match_iou`.
        3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
          box is assigned with the background class.
        4. The remaining anchor boxes that do not have any class assigned are
          ignored during training.

        Arguments:
          anchor_boxes: A float tensor with the shape `(total_anchors, 4)`
            representing all the anchor boxes for a given input image shape,
            where each anchor box is of the format `[x, y, width, height]`.
          gt_boxes: A float tensor with shape `(num_objects, 4)` representing
            the ground truth boxes, where each box is of the format
            `[x, y, width, height]`.
          match_iou: A float value representing the minimum IOU threshold for
            determining if a ground truth box can be assigned to an anchor box.
          ignore_iou: A float value representing the IOU threshold under which
            an anchor box is assigned to the background class.

        Returns:
          matched_gt_idx: Index of the matched object
          positive_mask: A mask for anchor boxes that have been assigned ground
            truth boxes.
          ignore_mask: A mask for anchor boxes that need to by ignored during
            training
        """
        iou_matrix = self.compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def compute_iou(self, boxes1, boxes2):
        """Computes pairwise IOU matrix for given two sets of boxes

        Arguments:
          boxes1: A tensor with shape `(N, 4)` representing bounding boxes
            where each box is of the format `[x, y, width, height]`.
            boxes2: A tensor with shape `(M, 4)` representing bounding boxes
            where each box is of the format `[x, y, width, height]`.

        Returns:
          pairwise IOU matrix with shape `(N, M)`, where the value at ith row
            jth column holds the IOU between ith box and jth box from
            boxes1 and boxes2 respectively.
        """
        boxes1_corners = transform_center_to_corner(boxes1)
        boxes2_corners = transform_center_to_corner(boxes2)
        lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
        rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
        intersection = tf.maximum(0.0, rd - lu)
        intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
        boxes1_area = boxes1[:, 2] * boxes1[:, 3]
        boxes2_area = boxes2[:, 2] * boxes2[:, 3]
        union_area = tf.maximum(
            boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
        )
        return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Transforms the ground truth boxes into targets for training"""
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        """Creates box and classification targets for a single sample"""
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, transform_corner_to_center(gt_boxes))
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids
        )
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)
        return label

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        """Creates box and classification targets for a batch"""
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        return batch_images, labels.stack()
