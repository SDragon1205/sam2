import numpy as np

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor, YOLOEVPDetectPredictor
from ultralytics.utils.plotting import plot_images
import cv2
import supervision as sv

# Initialize a YOLOE model
model = YOLOE("./ckpt/yoloe-11m-seg.pt")
model = model.to("cuda")
model.info()

# # Define visual prompts using bounding boxes and their corresponding class IDs.
# # Each box highlights an example of the object you want the model to detect.
# visual_prompts = dict(
#     bboxes=np.array(
#         [
#             [221.52, 405.8, 344.98, 857.54],  # Box enclosing person
#             [120, 425, 160, 445],  # Box enclosing glasses
#         ],
#     ),
#     cls=np.array(
#         [
#             0,  # ID to be assigned for person
#             1,  # ID to be assigned for glassses
#         ]
#     ),
# )

# # Run inference on an image, using the provided visual prompts as guidance
# results = model.predict(
#     "assets/bus.jpg",
#     visual_prompts=visual_prompts,
#     predictor=YOLOEVPSegPredictor,
# )

visual_prompts = dict(
    bboxes=[np.array([221.52, 405.8, 344.98, 857.54])],  # Box enclosing person
    cls=[0],  # ID to be assigned for person
)

# Run prediction on a different image, using reference image to guide what to look for
for i in range(2):
    results = model.predict(
        "assets/zidane.jpg",  # Target image for detection
        refer_image=["assets/bus.jpg"],  # Reference image used to get visual prompts
        visual_prompts=visual_prompts,
        predictor=YOLOEVPDetectPredictor,
    )

results = model.predict(
    "image/cylee.jpg",  # Target image for detection
    # refer_image="assets/bus.jpg",  # Reference image used to get visual prompts
    # visual_prompts=visual_prompts,
    # predictor=YOLOEVPDetectPredictor,
)

bus_image=cv2.imread("assets/bus.jpg")

detections = sv.Detections(
  xyxy=np.array([[221.52, 405.8, 344.98, 857.54]]),
  class_id=np.array([0]),
  confidence=np.array([1])
)

bounding_box_annotator = sv.BoxAnnotator()
annotated_frame = bounding_box_annotator.annotate(
    scene=bus_image.copy(),
    detections=detections
)

cv2.imwrite("assets/bus_ann.jpg", annotated_frame)
# sv.utils.image.save_image(annotated_frame, "assets/bus_ann.jpg")

# plot_images(images=np.expand_dims(bus_image, axis=0),
#             batch_idx=np.array(1),
#             cls=np.array([0]),
#             bboxes=np.array([[221.52, 405.8, 344.98, 857.54]]),
#             fname="assets/bus_ann.jpg",)

# Show results
results[0].save("assets/zidane_predict.jpg")