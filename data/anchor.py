import itertools
import math
import tensorflow as tf

def generate_retina_boxes_v2(param):
    """ Generate default boxes for all feature maps

    Args:
        config: information of feature maps
            scales: boxes' size relative to image's size
            fm_sizes: sizes of feature maps
            ratios: box ratios used in each feature maps

    Returns:
        default_boxes: tensor of shape (num_default, 4)
                       with format (cx, cy, w, h)
    """
    default_boxes = []
    image_size = param["image_size"]
    fm_sizes = param['fm_sizes']
    dimensions = param['dimensions']
    scales = param['scales']
    ratios = param['ratios']

    for m, (fm_size, dimension) in enumerate(zip(fm_sizes, dimensions)):
        for i, j in itertools.product(range(fm_size), repeat=2):
            cx = (j + 0.5) / fm_size
            cy = (i + 0.5) / fm_size
            size = dimension/image_size

            for scale in scales:
                for ratio in ratios:
                    default_boxes.append([
                                    cx,
                                    cy,
                                    size*scale*(ratio**0.5),
                                    size*scale/(ratio**0.5)
                                ])
        print(len(default_boxes))
    default_boxes = tf.constant(default_boxes)
    #default_boxes = tf.clip_by_value(default_boxes, 0.0, 1.0)

    return default_boxes

def generate_retina_boxes(param):
    """ Generate default boxes for all feature maps

    Args:
        config: information of feature maps
            scales: boxes' size relative to image's size
            fm_sizes: sizes of feature maps
            ratios: box ratios used in each feature maps

    Returns:
        default_boxes: tensor of shape (num_default, 4)
                       with format (cx, cy, w, h)
    """
    default_boxes = []
    scales = param['scales']
    fm_sizes = param['fm_sizes']
    ratios = param['ratios']

    for m, fm_size in enumerate(fm_sizes):
        for i, j in itertools.product(range(fm_size), repeat=2):
            cx = (j + 0.5) / fm_size
            cy = (i + 0.5) / fm_size
            size = (1/fm_size)

            for scale in scales:
                for ratio in ratios:
                    default_boxes.append([
                                    cx,
                                    cy,
                                    size*scale*(ratio**0.5),
                                    size*scale/(ratio**0.5)
                                ])
        print(len(default_boxes))
    default_boxes = tf.constant(default_boxes)
    default_boxes = tf.clip_by_value(default_boxes, 0.0, 1.0)

    return default_boxes

def generate_default_boxes(param):
    """ Generate default boxes for all feature maps

    Args:
        config: information of feature maps
            scales: boxes' size relative to image's size
            fm_sizes: sizes of feature maps
            ratios: box ratios used in each feature maps

    Returns:
        default_boxes: tensor of shape (num_default, 4)
                       with format (cx, cy, w, h)
    """
    '''
    param examples
    "SSD300" : {"ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                           "scales": [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075],
                           "fm_sizes": [38, 19, 10, 5, 3, 1],
                           "image_size": 300},
    "SSD512" : {"ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
                           "scales": [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
                           "fm_sizes": [64, 32, 16, 8, 6, 4, 1],
                           "image_size": 512},



    '''
    default_boxes = []
    scales = param['scales']
    fm_sizes = param['fm_sizes']
    ratios = param['ratios']

    for m, fm_size in enumerate(fm_sizes):
        for i, j in itertools.product(range(fm_size), repeat=2):
            cx = (j + 0.5) / fm_size
            cy = (i + 0.5) / fm_size
            default_boxes.append([
                cx,
                cy,
                scales[m],
                scales[m]
            ])

            default_boxes.append([
                cx,
                cy,
                math.sqrt(scales[m] * scales[m + 1]),
                math.sqrt(scales[m] * scales[m + 1])
            ])

            for ratio in ratios[m]:
                r = math.sqrt(ratio)
                default_boxes.append([
                    cx,
                    cy,
                    scales[m] * r,
                    scales[m] / r
                ])

                default_boxes.append([
                    cx,
                    cy,
                    scales[m] / r,
                    scales[m] * r
                ])

    default_boxes = tf.constant(default_boxes)
    default_boxes = tf.clip_by_value(default_boxes, 0.0, 1.0)

    return default_boxes



class RetinaNetAnchorBox:
    """Generates RetinaNet's anchor boxes.

    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.

    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    """

    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """Generates anchor boxes for a given feature map size and level

        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.

        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        """Generates anchor boxes for all the feature maps of the feature pyramid.

        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.

        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)
