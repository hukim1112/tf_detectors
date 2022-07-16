import random, os
from pycocotools.coco import COCO
from .box_utils import compute_target
from .anchor import generate_default_boxes
import tensorflow as tf
import cv2
import numpy as np

anchor_param = {"ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                           "scales": [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075],
                           "fm_sizes": [38, 19, 10, 5, 3, 1],
                           "image_size": 300} #anchor parameters
anchors = generate_default_boxes(anchor_param)

class SSDDataset():
    def __init__(self, image_path, annotation_path, anchors=None, image_size = 300, transform=None, target_transform=None, num_examples=None, shuffle=True):
        self.image_path = image_path # path to images
        self.coco = COCO(annotation_path) #path to .json file
        self.data_shuffle = shuffle # whether shuffle your data randomly or not
        self.anchors = anchors # anchor boxes
        self.size = image_size #size of resized image (integer)
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

    def num_classes(self):
        return len(self.coco_labels_inverse)

    def __len__(self):
        return len(self.ids)

    def load_tfds(self, batch_size, model):
        self.anchors = model.anchors()
        autotune = tf.data.AUTOTUNE
        trainable_form = True
        tfds = tf.data.Dataset.from_tensor_slices(self.image_ids)
        if self.data_shuffle:
            tfds = tfds.shuffle(128)
        tfds = tfds.map(lambda image_id: tf.py_function(func=self.get_item, inp=[image_id, trainable_form],
                        Tout=[tf.float32, tf.float32, tf.float32]),
                        num_parallel_calls=autotune)
        tfds = tfds.batch(batch_size=batch_size)
        tfds = tfds.map(self.label_encoder.encode_batch, num_parallel_calls=autotune)
        tfds = tfds.apply(tf.data.experimental.ignore_errors())
        tfds = tfds.prefetch(autotune)
        return tfds

    def get_item(self, image_id, trainable_form=False):
        image_id = int(image_id)
        trainable_form = bool(trainable_form)
        image, (height, width) = self.get_image(image_id)
        gt_labels, gt_boxes  = self.get_labels(image_id)
        gt_boxes = list(map(lambda box : (box[0]/width, box[1]/height, box[2]/width, box[3]/height), gt_boxes))
        gt_boxes = np.array(gt_boxes, np.float32); gt_labels = np.array(gt_labels, np.float32);

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            gt_boxes = self.target_transform(gt_boxes)
        if trainable_form:
            gt_labels, gt_boxes = compute_target(self.anchors, gt_boxes, gt_labels)
        else:
            gt_boxes = list(map(lambda box : (box[0]*image.shape[1], box[1]*image.shape[0],
                        box[2]*image.shape[1], box[3]*image.shape[0]), gt_boxes))
            gt_boxes = np.array(gt_boxes, np.float32)
        return image, gt_boxes, gt_labels

    def get_image(self, image_id):
        image_info = self.coco.loadImgs(image_id)[0]
        filename = image_info['file_name']
        original_size = (int(image_info['height']), int(image_info['width']))
        path = os.path.join(self.image_path, filename)
        image = cv2.imread(path)[:,:,::-1]
        image = cv2.resize(image, (self.size, self.size))
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
        classes["background"] = 0
        for c in categories:
            coco_labels[len(classes)] = c['id']
            coco_labels_inverse[c['id']] = len(classes)
            classes[c['name']] = len(classes)
        # also load the reverse (label -> name)
        labels = {}
        for key, value in classes.items():
            labels[value] = key
        return classes, labels, coco_labels, coco_labels_inverse

class LabelEncoder:
    def __init__(self):
        pass
    def encode_batch(self, batch_images, gt_locs, gt_confs):
        gt_confs = tf.expand_dims(gt_confs, axis=-1)
        return batch_images, tf.concat([gt_locs, gt_confs], axis=-1)
