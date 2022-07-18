import os
import tensorflow as tf

#environment setting
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)

import datetime
from data.retina import RetinaDataset
from models.RetinaNet import RetinaNet
from config.path import PATH # config/path.py to manage your dataset paths
coco_path = PATH["COCO"]


#-----------paramter setting------------#
os.environ["CUDA_VISIBLE_DEVICES"]="1"
epochs = 100
batch_size = 8
optimizer_clipnorm = 0.001
autotune = tf.data.AUTOTUNE
model_dir = os.path.join("experiments/RetinaNet", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(model_dir, exist_ok=True)


#load datasets
image_path = f"{coco_path}/images/train2017"
annotation_path = f"{coco_path}/annotations/instances_train2017.json"
train_dataset = RetinaDataset(image_path, annotation_path ,shuffle=True)
image_path = f"{coco_path}/images/val2017"
annotation_path = f"{coco_path}/annotations/instances_val2017.json"
val_dataset = RetinaDataset(image_path, annotation_path, shuffle=False)

# load model
model = RetinaNet(num_classes = len(val_dataset.coco_labels_inverse))

#load data pipeline
train_tfds = train_dataset.load_tfds(batch_size=batch_size, model=model)
val_tfds = val_dataset.load_tfds(batch_size=batch_size, model=model)

# compile model.
optimizer = tf.optimizers.Adam(learning_rate=0.001, clipnorm=optimizer_clipnorm)
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
# fit model.
model.fit(
    train_tfds,
    validation_data=val_tfds,
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)
