from ultralytics.models.yolo.model import YOLOE
from ultralytics.nn.tasks import YOLOEModel, YOLOESegModel
from ultralytics.models import yolo
from v2vdet.v2vdet_ultralytics.models.yolo.yoloe.val import YOLOEDetectValidator, YOLOESegValidator

class YOLOE_v2v(YOLOE):
    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOEModel,
                "validator": YOLOEDetectValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.yoloe.YOLOETrainer,
            },
            "segment": {
                "model": YOLOESegModel,
                "validator": YOLOESegValidator,
                "predictor": yolo.segment.SegmentationPredictor,
                "trainer": yolo.yoloe.YOLOESegTrainer,
            },
        }