import numpy as np
from matplotlib import pyplot as plt

def visualize_detections(
    image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    return ax

def visualize_inference(image, id_to_name, detections):
    num_detections = detections.valid_detections[0]
    class_names = [id_to_name[int(x)] for x in detections.nmsed_classes[0][:num_detections]]
    visualize_detections(image,
    detections.nmsed_boxes[0][:num_detections],
    class_names,
    detections.nmsed_scores[0][:num_detections])
