from ultralytics.engine.model import Model
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, yaml_model_load
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
import inspect
from pathlib import Path
from typing import Dict, List, Union

class Model_v2(Model):
  """
  A base class for implementing YOLO models, unifying APIs across different model types.

  This class provides a common interface for various operations related to YOLO models, such as training,
  validation, prediction, exporting, and benchmarking. It handles different types of models, including those
  loaded from local files, Ultralytics HUB, or Triton Server.

  Attributes:
      callbacks (Dict): A dictionary of callback functions for various events during model operations.
      predictor (BasePredictor): The predictor object used for making predictions.
      model (nn.Module): The underlying PyTorch model.
      trainer (BaseTrainer): The trainer object used for training the model.
      ckpt (Dict): The checkpoint data if the model is loaded from a *.pt file.
      cfg (str): The configuration of the model if loaded from a *.yaml file.
      ckpt_path (str): The path to the checkpoint file.
      overrides (Dict): A dictionary of overrides for model configuration.
      metrics (Dict): The latest training/validation metrics.
      session (HUBTrainingSession): The Ultralytics HUB session, if applicable.
      task (str): The type of task the model is intended for.
      model_name (str): The name of the model.

  Methods:
      __call__: Alias for the predict method, enabling the model instance to be callable.
      _new: Initializes a new model based on a configuration file.
      _load: Loads a model from a checkpoint file.
      _check_is_pytorch_model: Ensures that the model is a PyTorch model.
      reset_weights: Resets the model's weights to their initial state.
      load: Loads model weights from a specified file.
      save: Saves the current state of the model to a file.
      info: Logs or returns information about the model.
      fuse: Fuses Conv2d and BatchNorm2d layers for optimized inference.
      predict: Performs object detection predictions.
      track: Performs object tracking.
      val: Validates the model on a dataset.
      benchmark: Benchmarks the model on various export formats.
      export: Exports the model to different formats.
      train: Trains the model on a dataset.
      tune: Performs hyperparameter tuning.
      _apply: Applies a function to the model's tensors.
      add_callback: Adds a callback function for an event.
      clear_callback: Clears all callbacks for an event.
      reset_callbacks: Resets all callbacks to their default functions.

  Examples:
      >>> from ultralytics import YOLO
      >>> model = YOLO("yolo11n.pt")
      >>> results = model.predict("image.jpg")
      >>> model.train(data="coco8.yaml", epochs=3)
      >>> metrics = model.val()
      >>> model.export(format="onnx")
  """
    
  def __init__(
      self,
      model: Union[str, Path] = "yolo11n.pt",
      task: str = None,
      verbose: bool = False,
  ) -> None:
    super().__init__(model=model, task=task, verbose=verbose)
  

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
      self.model, self.ckpt = attempt_load_one_weight(weights)
      self.task = self.model.args["task"]
      self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
      self.ckpt_path = self.model.pt_path
    else:
      weights = checks.check_file(weights)  # runs in all cases, not redundant with above call
      self.model, self.ckpt = weights, None
      self.task = task or guess_model_task(weights)
      self.ckpt_path = weights
    self.overrides["model"] = weights
    self.overrides["task"] = self.task
    self.model_name = weights