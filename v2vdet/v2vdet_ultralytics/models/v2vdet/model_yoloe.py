from ultralytics.engine.model import Model
from ultralytics.models.yolo import YOLOE

class multi_visual_prompt_YOLOE(YOLOE):
    """Multi visual prompt YOLOE object detection and segmentation model."""
    
    def __init__(self, model="yoloe-v8s-seg.pt", task=None, verbose=False) -> None:
        """
        Initialize YOLOE model with a pre-trained model file.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            task (str, optional): Task type for the model. Auto-detected if None.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task=task, verbose=verbose)
    
    