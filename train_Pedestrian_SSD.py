import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import datetime
import tensorflow as tf
from data.ssd_coco import COCO_Dataset
from models.SSD import SSD300

epochs = 100
batch_size = 32
autotune = tf.data.AUTOTUNE
model_dir = os.path.join("experiments/large_Pedestrian_SSD", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(model_dir, exist_ok=True)

#train dataset
image_path = "/home/files/datasets/COCO/images/train2017"
annotation_path = "/home/files/datasets/COCO/annotations/large_pedestrian_train.json"
train_dataset = COCO_Dataset(image_path, annotation_path ,shuffle=True)
#validation dataset
image_path = "/home/files/datasets/COCO/images/val2017"
annotation_path = "/home/files/datasets/COCO/annotations/large_pedestrian_val.json"
val_dataset = COCO_Dataset(image_path, annotation_path, shuffle=False)

model = SSD300(len(val_dataset.coco_labels_inverse))
train_dataset.anchors = model.anchors()
val_dataset.anchors = model.anchors()

#train dataloader
train_tfds = train_dataset.load_tfds(batch_size=batch_size, trainable_form=True)
#validation dataloader
val_tfds = val_dataset.load_tfds(batch_size=batch_size, trainable_form=True)

optimizer = tf.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer)

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.TensorBoard(log_dir=model_dir)
]

model.fit(
    train_tfds,
    validation_data=val_tfds,
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)
