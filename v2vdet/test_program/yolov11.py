from ultralytics import YOLO

if __name__ == "__main__":
  model = YOLO("yolo11s.yaml").load("./ckpt/yolo11s.pt")
  model.info()

  # metrics = model.predict(data="v2vdet_ultralytics/cfg/datasets/coco.yaml")
  for i in range(10):
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
  
  # results = model("image/cat_scenario.jpg")  # predict on an image

  # Access the results
  for result in results:
      xywh = result.boxes.xywh  # center-x, center-y, width, height
      xywhn = result.boxes.xywhn  # normalized
      xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
      xyxyn = result.boxes.xyxyn  # normalized
      names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
      confs = result.boxes.conf  # confidence score of each box