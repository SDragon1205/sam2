from pathlib import Path
import torch

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel
from ultralytics.utils import ROOT, YAML
from ultralytics.utils import LOGGER

from v2vdet.v2vdet_ultralytics.nn import (WorldModel,
                                   v2vdetModel,
                                   v2vWorldModel, V2V_with_Patch_Attn_Pooling_Model, V2V_with_2_Patch_Attn_Pooling_Model,
                                   V2V_multi_scale_clip_Model,
                                   V2V_Template_YOLO_Backbone_Model,
                                   V2V_Template_YOLO_Backbone_Share_Param_Model,
                                   V2V_Template_YOLO_Backbone_Share_Param_For_Only_Train_Linear_Layer,
                                   V2V_Template_YOLO_Backbone_Model_Contrastive_Loss_Model,
                                   V2V_DINO_Model,
                                   V2V_template_DINO_multi_scale_Model,
                                   V2V_DINO_with_registers_Model,
                                   V2V_template_DINO_with_registers_multi_scale_Model,
                                   V2V_template_SigLIP_Model,
                                   V2V_template_SigLIP_multi_scale_Model,
                                   V2V_template_SigLIP_multi_scale_multi_head_Model,
                                   V2V_template_SigLIPv2_Model,
                                   V2V_template_SigLIPv2_multi_scale_Model,
                                   V2V_template_SigLIP_with_new_dataset_Model,
                                   V2V_With_MultiScale_SAVPE_Model,
                                   V2V_With_MultiScale_SAVPE_SigLIP2_B_Model,
                                   V2V_With_MultiScale_SAVPE_SigLIP2_L_Model,
                                   V2V_With_MultiScale_SAVPE_PE_B16_Model,
                                   V2V_With_MultiScale_SAVPE_PE_L14_Model
                                   )
from v2vdet.v2vdet_ultralytics.nn.tasks import load_state_dict_layer_by_layer
from ultralytics.models.yolo.world.train_world import (WorldTrainerFromScratch)
from v2vdet.v2vdet_ultralytics.models.v2vdet.detect.train import (
                                                    v2vTrainerFromScratch,
                                                    v2vWorldTrainer,
                                                    V2V_with_2_Patch_Attn_Pooling_Trainer,
                                                    V2V_multi_scale_clip_Trainer, 
                                                    V2V_Template_YOLO_Backbone_Trainer,
                                                    SA_V_V2V_Template_YOLO_Backbone_Share_Param_Trainer,
                                                    V2V_Template_YOLO_Backbone_Share_Param_Trainer,
                                                    V2V_Template_YOLO_Backbone_Share_Param_Only_Train_Linear_Layer_Trainer,
                                                    V2V_Template_YOLO_Backbone_Model_Contrastive_Loss_Trainer,
                                                    v2v_DINO_Trainer,
                                                    v2v_DINO_multi_scale_Trainer,
                                                    v2v_DINO_with_registers_Trainer,
                                                    v2v_DINO_with_registers_multi_scale_Trainer,
                                                    V2V_template_SigLIP_Trainer,
                                                    V2V_template_SigLIP_multi_scale_Trainer,
                                                    V2V_template_SigLIP_new_dataset_Trainer,
                                                    V2V_Template_YOLO_Backbone_Share_Param_new_Segmentation_Dataset_Trainer,
                                                    V2V_template_SigLIPv2_Trainer,
                                                    V2V_With_MultiScale_SAVPE_Trainer,
                                                    V2V_With_MultiScale_SAVPE_SigLIP2_B_Trainer,
                                                    V2V_With_MultiScale_SAVPE_SigLIP2_L_Trainer,
                                                    V2V_template_SigLIPv2_multi_scale_Trainer,
                                                    V2V_With_MultiScale_SAVPE_PE_B16_Trainer,
                                                    V2V_With_MultiScale_SAVPE_PE_L14_Trainer)
from v2vdet.v2vdet_ultralytics.models.v2vdet.predict import (v2v_DetectionPredictor, V2V_Template_YOLO_Backbone_Share_Param_DetectionPredictor, v2v_WITH_SAVPE_DetectionPredictor)
from v2vdet.v2vdet_ultralytics.models.v2vdet.val import (v2v_DetectionValidator, 
                                                  v2v_with_SAVPE_DetectionValidator,
                                                  v2v_new_DetectionValidator,
                                                  v2v_with_attn_pooling_DetectionValidator, v2v_template_feats_DetectionValidator)
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, yaml_model_load
from PIL import Image

from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    SETTINGS,
    callbacks,
    checks,
    emojis,
)

class v2vdet_model(Model):
    """vision2vision any class object detection model."""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
      """
      Initialize YOLOv8-World model with a pre-trained model file.

      Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
      COCO class names.

      Args:
          model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
          verbose (bool): If True, prints additional information during initialization.
      """
      LOGGER.info(f"🤡🤡 You are using v2vdet_model! 🏚️🏚️")
      super().__init__(model=model, task="detect", verbose=verbose)

      # Assign default COCO class names when there are no custom names
      if not hasattr(self.model, "names"):
          self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": v2vdetModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": v2vTrainerFromScratch,
            }
        }

    def set_classes(self, classes):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e. ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        # self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = classes

    def load_state_dict(self, checkpoint, strict=True):
      """
      Load state dict layer by layer with shape mismatch handling.

      Args:
          checkpoint:
          strict:
      """
      current_state = self.model.state_dict()
      checkpoint_state = checkpoint.state_dict()
      processed_layers = {}
      skipped_layers = []

      for key in checkpoint_state.keys():
        try:
          checkpoint_param = checkpoint_state[key]
          current_param = current_state[key]

          if checkpoint_param.shape != current_param.shape:
              if checkpoint_param.numel() == 1 and current_param.numel() == 1:
                  checkpoint_param = checkpoint_param.reshape(current_param.shape)
                  print(f"Reshaped single element tensor for layer: {key}")
              else:
                  skipped_layers.append(f"{key} (model: {current_param.shape}, checkpoint: {checkpoint_param.shape})")
                  continue

          if checkpoint_param.dtype != current_param.dtype:
              checkpoint_param = checkpoint_param.to(dtype=current_param.dtype)

          processed_layers[key] = checkpoint_param

        except Exception as e:
            print(f"\nError processing layer {key}: {str(e)}")
            continue

      if skipped_layers:
          print("\nSkipped layers due to shape mismatch:")
          for layer in skipped_layers:
              print(f"- {layer}")

      self.model.load_state_dict(processed_layers, strict=False)
      self.ckpt = self.model.state_dict()
      return processed_layers

class v2vYOLOWorld(Model):
    """v2v with CLIP object detection model."""

    def __init__(self, model="yolov8s-world.pt", task="detect", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        # LOGGER.info(f"🥺 "*20)
        if self.__class__ == v2vYOLOWorld:
            LOGGER.info(f"You are using v2vYOLOWorld (Use CLIP to Process Image)!")
            LOGGER.info(f"You are fired 🤡")
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

        # self.model

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": v2vWorldModel,
                "validator": v2v_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": v2vWorldTrainer,
            }
        }

    def set_classes(self, classes, crop_img=None):
        """
        Set classes.

        Args:
            classes: classes name
            crop_img (List(PIL.Image)): A list of categories image embedding i.e. .
        """
        if (crop_img is None):
            blank_pil_list = [Image.new('RGB', (128, 128), color='white') for _ in range(len(classes))]
            self.model.set_classes(blank_pil_list)
        else:
            self.model.set_classes(crop_img)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = classes

    def _load(self, weights: str, task=None) -> None:
        """
        Loads a model from a checkpoint file or initializes it from a weights file.

        This method handles loading models from either .pt checkpoint files or other weight file formats. It sets
        up the model, task, and related attributes based on the loaded weights.

        Args:
            weights (str): Path to the model weights file to be loaded.
            task (str | None): The task associated with the model. If None, it will be inferred from the model.

        Raises:
            FileNotFoundError: If the specified weights file does not exist or is inaccessible.
            ValueError: If the weights file format is unsupported or invalid.

        Examples:
            >>> model = Model()
            >>> model._load("yolo11n.pt")
            >>> model._load("path/to/weights.pth", task="detect")
        """
        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            weights = checks.check_file(weights, download_dir=SETTINGS["weights_dir"])  # download and return local file
        weights = checks.check_model_file_from_stem(weights)  # add suffix, i.e. yolov8n -> yolov8n.pt

        if Path(weights).suffix == ".pt":
            temp_model, self.ckpt = attempt_load_one_weight(weights)
            if isinstance(self.ckpt, dict) and self.ckpt['model'] is None:
                self.model.load(weights=self.ckpt['ema'])
            else:
                self.model.load(weights=self.ckpt)
            self.model.to("cuda" if torch.cuda.is_available() else 'cpu').float()
            self.task = self.model.args["task"]
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = temp_model.pt_path
        else:
            weights = checks.check_file(weights)  # runs in all cases, not redundant with above call
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

    def load_state_dict(self, checkpoint):
        """
        Load state dict layer by layer.

        Args:
            checkpoint: Checkpoint. If you load ultralytics checkpoint, you can access like this: model_ckpt['model']
        """
        self.ckpt = checkpoint
        return load_state_dict_layer_by_layer(current_model=self.model, checkpoint=checkpoint)

class V2V_with_Patch_Attn_Pooling(v2vYOLOWorld):
    """V2V with CLIP Patch Attention Pooling's object detection model."""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        # LOGGER.info(f"🥺 "*20)
        if self.__class__ == V2V_with_Patch_Attn_Pooling:
            LOGGER.info(f"You are using v2v_with_Patch_Attn_Pooling (Get Template With Attn Pooling)!")
            LOGGER.info(f"🥺 Why are you fired 🤡")

        super().__init__(model=model, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": V2V_with_Patch_Attn_Pooling_Model,
                "validator": v2v_with_attn_pooling_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": v2vWorldTrainer,
            }
        }

class V2V_with_2_Patch_Attn_Pooling(v2vYOLOWorld):
    """V2V with CLIP Patch Attention Pooling's object detection model."""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        # LOGGER.info(f"🥺 "*20)
        if self.__class__ == V2V_with_2_Patch_Attn_Pooling:
            LOGGER.info(f"🥺You are using V2V_with_2_Patch_Attn_Pooling (Put different clip patch into c2fattn layer)🥺")

        super().__init__(model=model, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": V2V_with_2_Patch_Attn_Pooling_Model,
                "validator": v2v_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": V2V_with_2_Patch_Attn_Pooling_Trainer,
            }
        }

class V2V_multi_scale_clip(v2vYOLOWorld):
    """V2V with multi scaling clip patch. (Take multiple CLIP patch's to do attention pooling)"""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """

        if self.__class__ == V2V_multi_scale_clip:
            LOGGER.info(f"🎂 You are using V2V_multi_scale_clip (Take multiple CLIP patch's to do attention pooling, then send into model) 🎂")

        super().__init__(model=model, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": V2V_multi_scale_clip_Model,
                "validator": v2v_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": V2V_multi_scale_clip_Trainer,
            }
        }

class V2V_DINO(v2vYOLOWorld):
    """V2V with DINO. (Take DINO into model)"""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """

        if self.__class__ == V2V_DINO:
            LOGGER.info(f"🧋🧋 You are using V2V_DINO. (Take multiple Take DINO into model) 🧋🧋")

        super().__init__(model=model, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": V2V_DINO_Model,
                "validator": v2v_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": v2v_DINO_Trainer,
            }
        }
        
class V2V_DINO_multi_scale(v2vYOLOWorld):
    """V2V with DINO with multi-scale. (Take several layer's patch embedding of DINO)"""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """

        if self.__class__ == V2V_DINO_multi_scale:
            LOGGER.info(f"🦖🦖🦖 You are using V2V_DINO. (Take several layer's patch embedding of DINO) 🦖🦖🦖")

        super().__init__(model=model, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": V2V_template_DINO_multi_scale_Model,
                "validator": v2v_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": v2v_DINO_multi_scale_Trainer,
            }
        }
        
class V2V_DINO_with_registers(V2V_DINO):
    """V2V with DINO with register.
     VISION TRANSFORMERS NEED REGISTERS: https://arxiv.org/pdf/2309.16588
    """

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """

        if self.__class__ == V2V_DINO_with_registers:
            LOGGER.info(f"🛺🛺 You are using V2V DINO with register. 🛺🛺")

        super().__init__(model=model, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": V2V_DINO_with_registers_Model,
                "validator": v2v_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": v2v_DINO_with_registers_Trainer,
            }
        }
        
class V2V_DINO_with_registers_multi_scale(V2V_DINO_with_registers):
    """V2V with DINO with multi-scale. (Take several layer's patch embedding of DINO)"""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """

        if self.__class__ == V2V_DINO_with_registers_multi_scale:
            LOGGER.info(f"📆📆 You are using V2V DINO with register with multi-scale . (Take several layer's patch embedding of DINO) 📆📆")

        super().__init__(model=model, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": V2V_template_DINO_with_registers_multi_scale_Model,
                "validator": v2v_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": v2v_DINO_with_registers_multi_scale_Trainer,
            }
        }


class V2V_Template_YOLO_Backbone(v2vYOLOWorld):
    """V2V with Template_YOLO_Backbone. (Take YOLOv8's backbone as template)"""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """

        if self.__class__ == V2V_Template_YOLO_Backbone:
            LOGGER.info(f"🎃 You are using V2V with Template_YOLO_Backbone. (Take YOLOv8's backbone as template's encoder) 🎃")

        super().__init__(model=model, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": V2V_Template_YOLO_Backbone_Model,
                "validator": v2v_template_feats_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": V2V_Template_YOLO_Backbone_Trainer,
            }
        }


class V2V_Template_YOLO_Backbone_Share_Param(v2vYOLOWorld):
    """V2V_Template_YOLO_Backbone_Share_Param. (Take YOLOv8's backbone as template, and using same parameter with input image's backbone)"""

    def __init__(self, model="yolov8s-world.pt", task='detect', verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """

        if self.__class__ == V2V_Template_YOLO_Backbone_Share_Param:
            LOGGER.info(f"🥸 V2V_Template_YOLO_Backbone_Share_Param. (Take YOLOv8's backbone as template, and using same parameter with input image's backbone) 🥸")

        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return{
            "detect": {
                "model": V2V_Template_YOLO_Backbone_Share_Param_Model,
                "validator": v2v_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": V2V_Template_YOLO_Backbone_Share_Param_Trainer,
            }
            # "detect_m": {
            #     "model": V2V_Template_YOLO_Backbone_Share_Param_Model,
            #     "validator": v2v_DetectionValidator,
            #     "predictor": v2v_DetectionPredictor,
            #     "trainer": V2V_Template_YOLO_Backbone_Share_Param_Trainer,
            # },
            # "stage1": {
            #     "model": V2V_Template_YOLO_Backbone_Share_Param_Model,
            #     "validator": v2v_DetectionValidator,
            #     "predictor": v2v_DetectionPredictor,
            #     "trainer": First_Stage_V2V_Template_YOLO_Backbone_Share_Param_Trainer,
            # },
            # "stage2": {
            #     "model": V2V_Template_YOLO_Backbone_Share_Param_Model,
            #     "validator": v2v_template_feats_DetectionValidator,
            #     "predictor": V2V_Template_YOLO_Backbone_Share_Param_DetectionPredictor,
            #     "trainer": V2V_Template_YOLO_Backbone_Share_Param_Trainer,
            # },
        }

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"], template_feats=batch["template_img"]) if preds is None else preds
        return self.criterion(preds, batch)
    
class V2V_Template_YOLO_Backbone_Share_Param_Segm_Dataset(v2vYOLOWorld):
    """V2V_Template_YOLO_Backbone_Share_Param. (Take YOLOv8's backbone as template, and using same parameter with input image's backbone)"""

    def __init__(self, model="yolov8s-world.pt", task='detect', verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """

        if self.__class__ == V2V_Template_YOLO_Backbone_Share_Param_Segm_Dataset:
            LOGGER.info(f"🥸 V2V_Template_YOLO_Backbone_Share_Param with Segmentation Dataset. (Take YOLOv8's backbone as template, and using same parameter with input image's backbone) 🥸")

        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return{
            "detect": {
                "model": V2V_Template_YOLO_Backbone_Share_Param_Model,
                "validator": v2v_new_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": V2V_Template_YOLO_Backbone_Share_Param_new_Segmentation_Dataset_Trainer,
            }
        }

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"], template_feats=batch["template_img"]) if preds is None else preds
        return self.criterion(preds, batch)

class V2V_Template_YOLO_Backbone_Model_Contrastive_Loss(v2vYOLOWorld):
    """V2V_Template_YOLO_Backbone_Share_Param. (Take YOLOv8's backbone as template, and using same parameter with input image's backbone)"""

    def __init__(self, model="yolov8s-world.pt", task='detect', verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """

        if self.__class__ == V2V_Template_YOLO_Backbone_Model_Contrastive_Loss_Model:
            LOGGER.info(f"😎 V2V_Template_YOLO_Backbone_Model_Contrastive_Loss_Model. 😎")

        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return{
            "detect": {
                "model": V2V_Template_YOLO_Backbone_Model_Contrastive_Loss_Model,
                "validator": v2v_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": V2V_Template_YOLO_Backbone_Model_Contrastive_Loss_Trainer,
            }
        }

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"], template_feats=batch["template_img"]) if preds is None else preds
        return self.criterion(preds, batch)

class V2V_Template_YOLO_Backbone_Share_Param_Train_Linear_Layer(V2V_Template_YOLO_Backbone_Share_Param):
    """V2V_Template_YOLO_Backbone_Share_Param. (Take YOLOv8's backbone as template, and using same parameter with input image's backbone)"""

    def __init__(self, model="yolov8s-world.pt", task='detect', verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """

        if self.__class__ == V2V_Template_YOLO_Backbone_Share_Param_Train_Linear_Layer:
            LOGGER.info(f"🙈 V2V_Template_YOLO_Backbone_Share_Param_Train_Linear_Layer. (Train Linear layer for big model only. Take YOLOv8's backbone as template, and using same parameter with input image's backbone) 🙈")

        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return{
            "detect": {
                "model": V2V_Template_YOLO_Backbone_Share_Param_Model,
                "validator": v2v_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": V2V_Template_YOLO_Backbone_Share_Param_Only_Train_Linear_Layer_Trainer,
            },
        }
        
class V2V_template_SigLIP(V2V_Template_YOLO_Backbone_Share_Param):
    """V2V_Template_SigLIP. (Take SigLIP's output as template)"""

    def __init__(self, model="yolov8s-world.pt", task='detect', verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """

        if self.__class__ == V2V_template_SigLIP:
            LOGGER.info(f"🏚️ 🎱 V2V_template_SigLIP. 🎱🏚️")

        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return{
            "detect": {
                "model": V2V_template_SigLIP_with_new_dataset_Model,
                "validator": v2v_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": V2V_template_SigLIP_Trainer,
            },
        }

class V2V_template_SigLIP_with_new_dataset(V2V_template_SigLIP):
    """V2V_Template_SigLIP. (Take SigLIP's output as template)"""

    def __init__(self, model="yolov8s-world.pt", task='detect', verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """

        if self.__class__ == V2V_template_SigLIP_with_new_dataset:
            LOGGER.info(f"👨🏿‍💻👨🏿‍💻 V2V_template_SigLIP. WITH NEW DATASET! 👨🏿‍💻👨🏿‍💻")

        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return{
            "detect": {
                "model": V2V_template_SigLIP_Model,
                "validator": v2v_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": V2V_template_SigLIP_new_dataset_Trainer,
            },
        }
        
class V2V_template_SigLIP_multi_scale(V2V_template_SigLIP):
    """V2V_Template_SigLIP. (Take SigLIP's output as template)"""

    def __init__(self, model="yolov8s-world.pt", task='detect', verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """

        if self.__class__ == V2V_template_SigLIP_multi_scale:
            LOGGER.info(f"🍊 V2V_template_SigLIP with multi scale. 🍊")

        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return{
            "detect": {
                "model": V2V_template_SigLIP_multi_scale_Model,
                "validator": v2v_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": V2V_template_SigLIP_multi_scale_Trainer,
            },
        }

class V2V_template_SigLIP_multi_scale_multi_head(V2V_template_SigLIP_multi_scale):
    """V2V with multi scaling SigLIP patch, with multi-head template attention pooling. (Take multiple SigLIP patch's to do attention pooling)
    """

    def __init__(self, model="yolov8s-world.pt", task='detect', verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """

        if self.__class__ == V2V_template_SigLIP_multi_scale:
            LOGGER.info(f"🍊🐱 V2V with multi scaling SigLIP patch, with multi-head template attention pooling. (Take multiple SigLIP patch's to do attention pooling). 🐱🍊")

        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return{
            "detect": {
                "model": V2V_template_SigLIP_multi_scale_multi_head_Model,
                "validator": v2v_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": V2V_template_SigLIP_multi_scale_Trainer,
            },
        }
    
class V2V_template_SigLIPv2(V2V_template_SigLIP):
    """V2V_Template_SigLIPv2. (Take SigLIP's output as template)"""

    def __init__(self, model="yolov8s-world.pt", task='detect', verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """

        if self.__class__ == V2V_template_SigLIPv2:
            LOGGER.info(f"🥳🥳 V2V template SigLIPv2. 🥳🥳")

        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return{
            "detect": {
                "model": V2V_template_SigLIPv2_Model,
                "validator": v2v_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": V2V_template_SigLIPv2_Trainer,
            },
        }
    
class V2V_template_SigLIPv2_multi_scale(V2V_template_SigLIPv2):
    """V2V_Template_SigLIPv2 with multi scale. (Take SigLIP's output as template)"""

    def __init__(self, model="yolov8s-world.pt", task='detect', verbose=False) -> None:
        """


        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """

        if self.__class__ == V2V_template_SigLIPv2_multi_scale:
            LOGGER.info(f"🧌🧌 V2V template SigLIPv2 with multi scale features. 🧌🧌")

        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return{
            "detect": {
                "model": V2V_template_SigLIPv2_multi_scale_Model,
                "validator": v2v_DetectionValidator,
                "predictor": v2v_DetectionPredictor,
                "trainer": V2V_template_SigLIPv2_multi_scale_Trainer,
            },
        }

class YOLOWorld(Model):
    """YOLO-World object detection model."""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        LOGGER.info(f"You are using YOLOWorld!")
        super().__init__(model=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

        # self.model

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": WorldTrainerFromScratch,
            }
        }

    def set_classes(self, classes):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e. ["person"].
        """

        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        # self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = classes

    def load_state_dict(self, checkpoint):
        """
        Load state dict layer by layer.

        Args:
            checkpoint: Checkpoint. If you load ultralytics checkpoint, you can access like this: model_ckpt['model']
        """
        self.ckpt = checkpoint
        return load_state_dict_layer_by_layer(self.model, checkpoint)

class V2V_With_MultiScale_SAVPE(v2vYOLOWorld):

    def __init__(self, model="yolo11s.pt", task="detect", verbose=False) -> None:
        """
        Initialize V2V with SAVPE model with a pre-trained model file.

        Loads a yolo11 model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        if self.__class__ == V2V_With_MultiScale_SAVPE:
            LOGGER.info(f"🐱🐱 You are using V2V_With_MultiScale_SAVPE 🐱🐱")
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

        # self.model

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": V2V_With_MultiScale_SAVPE_Model,
                "validator": v2v_with_SAVPE_DetectionValidator,
                "predictor": v2v_WITH_SAVPE_DetectionPredictor,
                "trainer": V2V_With_MultiScale_SAVPE_Trainer,
            }
        }
        
class V2V_With_MultiScale_SAVPE_SigLIP2_B(V2V_With_MultiScale_SAVPE):

    def __init__(self, model="yolo11s.pt", task="detect", verbose=False) -> None:
        """
        Initialize V2V with SAVPE model with a pre-trained model file.

        Loads a yolo11 model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        if self.__class__ == V2V_With_MultiScale_SAVPE_SigLIP2_B:
            LOGGER.info(f"🥶🥶 You are using V2V_With_MultiScale_SAVPE_SigLIP2_B 🥶🥶")
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

        # self.model

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": V2V_With_MultiScale_SAVPE_SigLIP2_B_Model,
                "validator": v2v_with_SAVPE_DetectionValidator,
                "predictor": v2v_WITH_SAVPE_DetectionPredictor,
                "trainer": V2V_With_MultiScale_SAVPE_SigLIP2_B_Trainer,
            }
        }
        
class V2V_With_MultiScale_SAVPE_SigLIP2_L(V2V_With_MultiScale_SAVPE):

    def __init__(self, model="yolo11s.pt", task="detect", verbose=False) -> None:
        """
        Initialize V2V with SAVPE model with a pre-trained model file.

        Loads a yolo11 model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        if self.__class__ == V2V_With_MultiScale_SAVPE_SigLIP2_L:
            LOGGER.info(f"☃️☃️ You are using V2V_With_MultiScale_SAVPE_SigLIP2_L ☃️☃️")
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

        # self.model

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": V2V_With_MultiScale_SAVPE_SigLIP2_L_Model,
                "validator": v2v_with_SAVPE_DetectionValidator,
                "predictor": v2v_WITH_SAVPE_DetectionPredictor,
                "trainer": V2V_With_MultiScale_SAVPE_SigLIP2_L_Trainer,
            }
        }

class V2V_With_MultiScale_SAVPE_PE_B16(V2V_With_MultiScale_SAVPE):

    def __init__(self, model="yolo11s.pt", task="detect", verbose=False) -> None:
        """
        Initialize V2V with SAVPE model with a pre-trained model file.

        Loads a yolo11 model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        if self.__class__ == V2V_With_MultiScale_SAVPE_PE_B16:
            LOGGER.info(f"🔱🔱 You are using V2V_With_MultiScale_SAVPE_PE_B16 🔱🔱🪬🪬")
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

        # self.model
    

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": V2V_With_MultiScale_SAVPE_PE_B16_Model,
                "validator": v2v_with_SAVPE_DetectionValidator,
                "predictor": v2v_WITH_SAVPE_DetectionPredictor,
                "trainer": V2V_With_MultiScale_SAVPE_PE_B16_Trainer,
            }
        }
    
class V2V_With_MultiScale_SAVPE_PE_L14(V2V_With_MultiScale_SAVPE_PE_B16):

    def __init__(self, model="yolo11s.pt", task="detect", verbose=False) -> None:
        """
        Initialize V2V with SAVPE model with a pre-trained model file.

        Loads a yolo11 model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        if self.__class__ == V2V_With_MultiScale_SAVPE_PE_L14:
            LOGGER.info(f"🪬🪬 You are using V2V_With_MultiScale_SAVPE_PE_L14 🪬🪬")
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

        # self.model

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": V2V_With_MultiScale_SAVPE_PE_L14_Model,
                "validator": v2v_with_SAVPE_DetectionValidator,
                "predictor": v2v_WITH_SAVPE_DetectionPredictor,
                "trainer": V2V_With_MultiScale_SAVPE_PE_L14_Trainer,
            }
        }

class V2V_With_MultiScale_SAVPE_ObjectOriented(V2V_With_MultiScale_SAVPE):

    def __init__(self, model="yolo11s.pt", task="detect", verbose=False) -> None:
        """
        Initialize V2V with SAVPE model with a pre-trained model file.

        Loads a yolo11 model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        if self.__class__ == V2V_With_MultiScale_SAVPE_ObjectOriented:
            LOGGER.info(f"🐯🐯 You are using V2V_With_MultiScale_SAVPE with Object Oriented Dataset! 🐯🐯")
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

        # self.model

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        
        from v2vdet.v2vdet_ultralytics.models.v2vdet.detect.train_object_oriented import V2V_With_MultiScale_SAVPE_ObjectOriented_Trainer
        from v2vdet.v2vdet_ultralytics.nn.tasks_oo import V2V_With_MultiScale_SAVPE_ObjectOriented_Model
        from v2vdet.v2vdet_ultralytics.models.v2vdet.oo_val import v2v_with_SAVPE_ObjectOriented_DetectionValidator
        
        return {
            "detect": {
                "model": V2V_With_MultiScale_SAVPE_ObjectOriented_Model,
                "validator": v2v_with_SAVPE_ObjectOriented_DetectionValidator,
                "predictor": v2v_WITH_SAVPE_DetectionPredictor,
                "trainer": V2V_With_MultiScale_SAVPE_ObjectOriented_Trainer,
            }
        }

class V2V_With_MultiScale_SAVPE_SigLIP2_B_ObjectOriented(V2V_With_MultiScale_SAVPE_ObjectOriented):

    def __init__(self, model="yolo11s.pt", task="detect", verbose=False) -> None:
        """
        Initialize V2V with SAVPE model with a pre-trained model file.

        Loads a yolo11 model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        if self.__class__ == V2V_With_MultiScale_SAVPE_SigLIP2_B_ObjectOriented:
            LOGGER.info(f"🥶🦧 You are using V2V_With_MultiScale_SAVPE SigLIP2 Base with Object Oriented Dataset! 🥶🦧")
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

        # self.model

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        
        from v2vdet.v2vdet_ultralytics.models.v2vdet.detect.train_object_oriented import V2V_With_MultiScale_SAVPE_SigLIP2_B_ObjectOriented_Trainer as TRAINER
        from v2vdet.v2vdet_ultralytics.nn.tasks_oo import V2V_With_MultiScale_SAVPE_SigLIP2_B_ObjectOriented_Model as MODEL
        from v2vdet.v2vdet_ultralytics.models.v2vdet.oo_val import v2v_with_SAVPE_ObjectOriented_DetectionValidator as VALADATOR
        
        return {
            "detect": {
                "model": MODEL,
                "validator": VALADATOR,
                "predictor": v2v_WITH_SAVPE_DetectionPredictor,
                "trainer": TRAINER,
            }
        }
        
class V2V_With_MultiScale_SAVPE_SigLIP2_L_ObjectOriented(V2V_With_MultiScale_SAVPE_ObjectOriented):

    def __init__(self, model="yolo11s.pt", task="detect", verbose=False) -> None:
        """
        Initialize V2V with SAVPE model with a pre-trained model file.

        Loads a yolo11 model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        if self.__class__ == V2V_With_MultiScale_SAVPE_SigLIP2_L_ObjectOriented:
            LOGGER.info(f"🐋🐋 You are using V2V_With_MultiScale_SAVPE SigLIP2 Large with Object Oriented Dataset! 🐋🐋")
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

        # self.model

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        
        from v2vdet.v2vdet_ultralytics.models.v2vdet.detect.train_object_oriented import V2V_With_MultiScale_SAVPE_SigLIP2_L_ObjectOriented_Trainer as TRAINER
        from v2vdet.v2vdet_ultralytics.nn.tasks_oo import V2V_With_MultiScale_SAVPE_SigLIP2_L_ObjectOriented_Model as MODEL
        from v2vdet.v2vdet_ultralytics.models.v2vdet.oo_val import v2v_with_SAVPE_ObjectOriented_DetectionValidator as VALADATOR
        
        return {
            "detect": {
                "model": MODEL,
                "validator": VALADATOR,
                "predictor": v2v_WITH_SAVPE_DetectionPredictor,
                "trainer": TRAINER,
            }
        }

class V2V_With_MultiScale_SAVPE_DINOv2_B_ObjectOriented(V2V_With_MultiScale_SAVPE_ObjectOriented):

    def __init__(self, model="yolo11s.pt", task="detect", verbose=False) -> None:
        """
        Initialize V2V with SAVPE model with a pre-trained model file.

        Loads a yolo11 model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        if self.__class__ == V2V_With_MultiScale_SAVPE_DINOv2_B_ObjectOriented:
            LOGGER.info(f"🌈🌈 You are using V2V_With_MultiScale_SAVPE DINOv2 Base with Object Oriented Dataset! 🌈🌈")
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

        # self.model

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        
        from v2vdet.v2vdet_ultralytics.models.v2vdet.detect.train_object_oriented import V2V_With_MultiScale_SAVPE_DINO2_B_ObjectOriented_Trainer as TRAINER
        from v2vdet.v2vdet_ultralytics.nn.tasks_oo import V2V_With_MultiScale_SAVPE_DINOv2_B_ObjectOriented_Model as MODEL
        from v2vdet.v2vdet_ultralytics.models.v2vdet.oo_val import v2v_with_SAVPE_ObjectOriented_DetectionValidator as VALADATOR
        
        return {
            "detect": {
                "model": MODEL,
                "validator": VALADATOR,
                "predictor": v2v_WITH_SAVPE_DetectionPredictor,
                "trainer": TRAINER,
            }
        }
        
class V2V_With_MultiScale_SAVPE_DINOv2_L_ObjectOriented(V2V_With_MultiScale_SAVPE_ObjectOriented):

    def __init__(self, model="yolo11s.pt", task="detect", verbose=False) -> None:
        """
        Initialize V2V with SAVPE model with a pre-trained model file.

        Loads a yolo11 model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        if self.__class__ == V2V_With_MultiScale_SAVPE_DINOv2_L_ObjectOriented:
            LOGGER.info(f"🚳🚳 You are using V2V_With_MultiScale_SAVPE DINOv2 Large with Object Oriented Dataset! 🚳🚳")
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

        # self.model

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        
        from v2vdet.v2vdet_ultralytics.models.v2vdet.detect.train_object_oriented import V2V_With_MultiScale_SAVPE_DINO2_L_ObjectOriented_Trainer as TRAINER
        from v2vdet.v2vdet_ultralytics.nn.tasks_oo import V2V_With_MultiScale_SAVPE_DINOv2_L_ObjectOriented_Model as MODEL
        from v2vdet.v2vdet_ultralytics.models.v2vdet.oo_val import v2v_with_SAVPE_ObjectOriented_DetectionValidator as VALADATOR
        
        return {
            "detect": {
                "model": MODEL,
                "validator": VALADATOR,
                "predictor": v2v_WITH_SAVPE_DetectionPredictor,
                "trainer": TRAINER,
            }
        }
        
class V2V_With_MultiScale_SAVPE_PE_B16_ObjectOriented(V2V_With_MultiScale_SAVPE_ObjectOriented):

    def __init__(self, model="yolo11s.pt", task="detect", verbose=False) -> None:
        """
        Initialize V2V with SAVPE model with a pre-trained model file.

        Loads a yolo11 model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        if self.__class__ == V2V_With_MultiScale_SAVPE_PE_B16_ObjectOriented:
            LOGGER.info(f"🐋🐋 You are using V2V_With_MultiScale_SAVPE PE B16 with Object Oriented Dataset! 🐋🐋")
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        
        from v2vdet.v2vdet_ultralytics.models.v2vdet.detect.train_object_oriented import V2V_With_MultiScale_SAVPE_PE_B16_ObjectOriented_Trainer as TRAINER
        from v2vdet.v2vdet_ultralytics.nn.tasks_oo import V2V_With_MultiScale_SAVPE_PE_B16_ObjectOriented_Model as MODEL
        from v2vdet.v2vdet_ultralytics.models.v2vdet.oo_val import v2v_with_SAVPE_ObjectOriented_DetectionValidator as VALADATOR
        
        return {
            "detect": {
                "model": MODEL,
                "validator": VALADATOR,
                "predictor": v2v_WITH_SAVPE_DetectionPredictor,
                "trainer": TRAINER,
            }
        }
        
class V2V_With_MultiScale_SAVPE_PE_L14_ObjectOriented(V2V_With_MultiScale_SAVPE_PE_B16_ObjectOriented):

    def __init__(self, model="yolo11s.pt", task="detect", verbose=False) -> None:
        """
        Initialize V2V with SAVPE model with a pre-trained model file.

        Loads a yolo11 model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        if self.__class__ == V2V_With_MultiScale_SAVPE_PE_L14_ObjectOriented:
            LOGGER.info(f"💼💼 You are using V2V_With_MultiScale_SAVPE PE L14 with Object Oriented Dataset! 💼💼")
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

        # self.model

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        
        from v2vdet.v2vdet_ultralytics.models.v2vdet.detect.train_object_oriented import V2V_With_MultiScale_SAVPE_PE_L14_ObjectOriented_Trainer as TRAINER
        from v2vdet.v2vdet_ultralytics.nn.tasks_oo import V2V_With_MultiScale_SAVPE_PE_L14_ObjectOriented_Model as MODEL
        from v2vdet.v2vdet_ultralytics.models.v2vdet.oo_val import v2v_with_SAVPE_ObjectOriented_DetectionValidator as VALADATOR
        
        return {
            "detect": {
                "model": MODEL,
                "validator": VALADATOR,
                "predictor": v2v_WITH_SAVPE_DetectionPredictor,
                "trainer": TRAINER,
            }
        }

class V2V_With_MultiScale_SAVPE_YOLOE_ObjectOriented(V2V_With_MultiScale_SAVPE_ObjectOriented):

    def __init__(self, model="yolo11s.pt", task="detect", verbose=False) -> None:
        """
        Initialize V2V with SAVPE model with a pre-trained model file.

        Loads a yolo11 model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        if self.__class__ == V2V_With_MultiScale_SAVPE_YOLOE_ObjectOriented:
            LOGGER.info(f"📯📯 You are using V2V_With_MultiScale_SAVPE YOLOE with Object Oriented Dataset! 📯📯")
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names

        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

        # self.model

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        
        from v2vdet.v2vdet_ultralytics.models.v2vdet.detect.train_object_oriented import V2V_With_MultiScale_SAVPE_YOLOE_ObjectOriented_Trainer as TRAINER
        from v2vdet.v2vdet_ultralytics.nn.tasks_oo import V2V_With_MultiScale_SAVPE_YOLOE_ObjectOriented_Model as MODEL
        from v2vdet.v2vdet_ultralytics.models.v2vdet.oo_val import v2v_with_SAVPE_ObjectOriented_DetectionValidator as VALADATOR
        
        return {
            "detect": {
                "model": MODEL,
                "validator": VALADATOR,
                "predictor": v2v_WITH_SAVPE_DetectionPredictor,
                "trainer": TRAINER,
            }
        }