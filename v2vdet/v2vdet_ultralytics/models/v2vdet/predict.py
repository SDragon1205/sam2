from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import select_device, smart_inference_mode
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from v2vdet.v2vdet_ultralytics.nn.autobackend import v2vdet_template_feats_AutoBackend
from v2vdet.v2vdet_ultralytics.nn import v2vdet_AutoBackend, v2vdet_template_feats_AutoBackend,v2v_with_SAVPE_AutoBackend

import torch
import cv2
from pathlib import Path

class v2v_DetectionPredictor(DetectionPredictor):
  """
  A class extending the BasePredictor class for prediction based on a detection model.

  Example:
      ```python
      from ultralytics.utils import ASSETS
      from ultralytics.models.yolo.detect import DetectionPredictor

      args = dict(model="yolo11n.pt", source=ASSETS)
      predictor = DetectionPredictor(overrides=args)
      predictor.predict_cli()
      ```
  """

  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    super().__init__(cfg, overrides, _callbacks)

  def setup_model(self, model, verbose=True):
    # """Initialize YOLO model with given parameters and set it to evaluation mode."""

    self.model = v2vdet_AutoBackend(
        weights=model or self.args.model,
        device=select_device(self.args.device, verbose=verbose),
        dnn=self.args.dnn,
        data=self.args.data,
        fp16=self.args.half,
        batch=self.args.batch,
        fuse=True,
        verbose=verbose,
    )

    self.device = self.model.device  # update device
    self.args.half = self.model.fp16  # update half
    self.model.eval()

  @smart_inference_mode()
  def stream_inference(self, source=None, model=None, *args, **kwargs):
    """
    Stream real-time inference on camera feed and save results to file.

    Args:
        source (str | Path | List[str] | List[Path] | List[np.ndarray] | np.ndarray | torch.Tensor | None):
            Source for inference.
        model (str | Path | torch.nn.Module | None): Model for inference.
        *args (Any): Additional arguments for the inference method.
        **kwargs (Any): Additional keyword arguments for the inference method.

    Yields:
        (ultralytics.engine.results.Results): Results objects.
    """
    if self.args.verbose:
      LOGGER.info("")

    # Setup model
    if not self.model:
      self.setup_model(model)

    with self._lock:  # for thread-safe inference
      # Setup source every time predict is called
      self.setup_source(source if source is not None else self.args.source)

      # Check if save_dir/ label file exists
      if self.args.save or self.args.save_txt:
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

      # Warmup model
      # if not self.done_warmup:
      #   self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
      #   self.done_warmup = True

      self.seen, self.windows, self.batch = 0, [], None
      profilers = (
          ops.Profile(device=self.device),
          ops.Profile(device=self.device),
          ops.Profile(device=self.device),
      )
      self.run_callbacks("on_predict_start")
      for self.batch in self.dataset:
        self.run_callbacks("on_predict_batch_start")
        paths, im0s, s = self.batch

        # Preprocess
        # print("im0s.shape:", im0s.shape)
        with profilers[0]:
          im = self.preprocess(im0s)
        # print("im.shape:", im.shape)
        # Inference
        with profilers[1]:
          preds = self.inference(im, *args, **kwargs)
          if self.args.embed:
              yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
              continue

        # Postprocess
        with profilers[2]:
          self.results = self.postprocess(preds, im, im0s)
        self.run_callbacks("on_predict_postprocess_end")

        # Visualize, save, write results
        n = len(im0s)
        for i in range(n):
          self.seen += 1
          self.results[i].speed = {
              "preprocess": profilers[0].dt * 1e3 / n,
              "inference": profilers[1].dt * 1e3 / n,
              "postprocess": profilers[2].dt * 1e3 / n,
          }
          if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
            s[i] += self.write_results(i, Path(paths[i]), im, s)

        # Print batch results
        if self.args.verbose:
            LOGGER.info("\n".join(s))

        self.run_callbacks("on_predict_batch_end")
        yield from self.results

    # Release assets
    for v in self.vid_writer.values():
      if isinstance(v, cv2.VideoWriter):
          v.release()

    # Print final results
    if self.args.verbose and self.seen:
      t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
      LOGGER.info(
          f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
          f"{(min(self.args.batch, self.seen), 3, *im.shape[2:])}" % t
      )
    if self.args.save or self.args.save_txt or self.args.save_crop:
      nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
      s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
      LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
    self.run_callbacks("on_predict_end")


class v2v_WITH_SAVPE_DetectionPredictor(DetectionPredictor):
  """
  A class extending the BasePredictor class for prediction based on a detection model.

  Example:
      ```python
      from ultralytics.utils import ASSETS
      from ultralytics.models.yolo.detect import DetectionPredictor

      args = dict(model="yolo11n.pt", source=ASSETS)
      predictor = DetectionPredictor(overrides=args)
      predictor.predict_cli()
      ```
  """

  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    super().__init__(cfg, overrides, _callbacks)

  def setup_model(self, model, verbose=True):
    # """Initialize YOLO model with given parameters and set it to evaluation mode."""

    self.model = v2v_with_SAVPE_AutoBackend(
        weights=model or self.args.model,
        device=select_device(self.args.device, verbose=verbose),
        dnn=self.args.dnn,
        data=self.args.data,
        fp16=self.args.half,
        batch=self.args.batch,
        fuse=True,
        verbose=verbose,
    )

    self.device = self.model.device  # update device
    self.args.half = self.model.fp16  # update half
    self.model.eval()

class V2V_Template_YOLO_Backbone_Share_Param_DetectionPredictor(DetectionPredictor):
  """
  A class extending the BasePredictor class for prediction based on a detection model.

  Example:
      ```python
      from ultralytics.utils import ASSETS
      from ultralytics.models.yolo.detect import DetectionPredictor

      args = dict(model="yolo11n.pt", source=ASSETS)
      predictor = DetectionPredictor(overrides=args)
      predictor.predict_cli()
      ```
  """

  def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
    super().__init__(cfg, overrides, _callbacks)

  def setup_model(self, model, verbose=True):
    # """Initialize YOLO model with given parameters and set it to evaluation mode."""

    # self.model = AutoBackend(
    #     weights=model or self.args.model,
    #     device=select_device(self.args.device, verbose=verbose),
    #     dnn=self.args.dnn,
    #     data=self.args.data,
    #     fp16=self.args.half,
    #     batch=self.args.batch,
    #     fuse=True,
    #     verbose=verbose,
    # )
    
    self.model = v2vdet_template_feats_AutoBackend(
          weights=model or self.args.model,
          device=select_device(self.args.device, self.args.batch),
          dnn=self.args.dnn,
          data=self.args.data,
          fp16=self.args.half,
      )

    self.device = self.model.device  # update device
    self.args.half = self.model.fp16  # update half
    self.model.eval()
