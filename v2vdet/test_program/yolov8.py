from ultralytics import YOLO

def count_trainable_parameters(model):
    # 計算所有需要梯度的參數總量
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
  
def count_parameters(model):
    # 計算所有需要梯度的參數總量
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":

    # Load a COCO-pretrained YOLOv8n model
    model = YOLO("yolov8s.yaml")
    model = model.load("yolov8s.pt")

    # Display model information (optional)
    model.info()

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Access the results
for result in results:
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box