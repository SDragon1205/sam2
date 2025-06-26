# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import deepcopy
import json

import torch
from torch.nn import functional as F

from ultralytics.data import YOLOConcatDataset, build_dataloader, build_yolo_dataset
from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.nn.modules.head import YOLOEDetect
from ultralytics.nn.tasks import YOLOEModel
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.torch_utils import select_device, smart_inference_mode

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode

from v2vdet.v2vdet_ultralytics.nn import (v2vdet_AutoBackend, 
                                   v2vdet_template_feats_AutoBackend,
                                   v2v_with_SAVPE_AutoBackend)

class YOLOEDetectValidator(DetectionValidator):
    """
    A mixin class for YOLOE model validation that handles both text and visual prompt embeddings.

    This mixin provides functionality to validate YOLOE models using either text or visual prompt embeddings.
    It includes methods for extracting visual prompt embeddings from samples, preprocessing batches, and
    running validation with different prompt types.

    Attributes:
        device (torch.device): The device on which validation is performed.
        args (namespace): Configuration arguments for validation.
        dataloader (DataLoader): DataLoader for validation data.
    """

    @smart_inference_mode()
    def get_visual_pe(self, dataloader, model):
        """
        Extract visual prompt embeddings from training samples.

        This function processes a dataloader to compute visual prompt embeddings for each class
        using a YOLOE model. It normalizes the embeddings and handles cases where no samples
        exist for a class.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader providing training samples.
            model (YOLOEModel): The YOLOE model from which to extract visual prompt embeddings.

        Returns:
            (torch.Tensor): Visual prompt embeddings with shape (1, num_classes, embed_dim).
        """
        assert isinstance(model, YOLOEModel)
        names = [name.split("/")[0] for name in list(dataloader.dataset.data["names"].values())]
        visual_pe = torch.zeros(len(names), model.model[-1].embed, device=self.device)
        cls_visual_num = torch.zeros(len(names))

        desc = "Get visual prompt embeddings from samples"

        for batch in dataloader:
            cls = batch["cls"].squeeze(-1).to(torch.int).unique()
            count = torch.bincount(cls, minlength=len(names))
            cls_visual_num += count

        cls_visual_num = cls_visual_num.to(self.device)

        pbar = TQDM(dataloader, total=len(dataloader), desc=desc)
        for batch in pbar:
            batch = self.preprocess(batch)
            preds = model.get_visual_pe(batch["img"], visual=batch["visuals"])  # (B, max_n, embed_dim)

            batch_idx = batch["batch_idx"]
            for i in range(preds.shape[0]):
                cls = batch["cls"][batch_idx == i].squeeze(-1).to(torch.int).unique(sorted=True)
                pad_cls = torch.ones(preds.shape[1], device=self.device) * -1
                pad_cls[: len(cls)] = cls
                for c in cls:
                    visual_pe[c] += preds[i][pad_cls == c].sum(0) / cls_visual_num[c]

        visual_pe[cls_visual_num != 0] = F.normalize(visual_pe[cls_visual_num != 0], dim=-1, p=2)
        visual_pe[cls_visual_num == 0] = 0
        return visual_pe.unsqueeze(0)

    def preprocess(self, batch):
        """Preprocess batch data, ensuring visuals are on the same device as images."""
        batch = super().preprocess(batch)
        if "visuals" in batch:
            batch["visuals"] = batch["visuals"].to(batch["img"].device)
        return batch
    
    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset for training or validation with visual prompts.

        Args:
            img_path (List[str] | str): Path to the folder containing images or list of paths.
            mode (str): 'train' mode or 'val' mode, allowing customized augmentations for each mode.
            batch (int, optional): Size of batches, used for rectangular training/validation.

        Returns:
            (Dataset): YOLO dataset configured for training or validation, with visual prompts for training mode.
        """
        self.args.rect = False
        self.args.task = 'detect'
        dataset = super().build_dataset(img_path, mode, batch)
        return dataset
    
    def get_dataloader(self, dataset_path, batch_size):
        """
        Construct and return dataloader.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Size of each batch.

        Returns:
            (torch.utils.data.DataLoader): Dataloader for validation.
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        if isinstance(dataset, YOLOConcatDataset):
            for d in dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            dataset.transforms.append(LoadVisualPrompt())
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def get_vpe_dataloader(self, data):
        """
        Create a dataloader for LVIS training visual prompt samples.

        This function prepares a dataloader for visual prompt embeddings (VPE) using the LVIS dataset.
        It applies necessary transformations and configurations to the dataset and returns a dataloader
        for validation purposes.

        Args:
            data (dict): Dataset configuration dictionary containing paths and settings.

        Returns:
            (torch.utils.data.DataLoader): The dataLoader for visual prompt samples.
        """
        dataset = build_yolo_dataset(
            self.args,
            data.get(self.args.split, data.get("val")),
            self.args.batch,
            data,
            mode="val",
            rect=False,
        )
        if isinstance(dataset, YOLOConcatDataset):
            for d in dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            dataset.transforms.append(LoadVisualPrompt())
        return build_dataloader(
            dataset,
            self.args.batch,
            self.args.workers,
            shuffle=False,
            rank=-1,
        )

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None, refer_data=None, load_vp=False):
        """
        Run validation on the model using either text or visual prompt embeddings.

        This method validates the model using either text prompts or visual prompts, depending
        on the `load_vp` flag. It supports validation during training (using a trainer object)
        or standalone validation with a provided model.

        Args:
            trainer (object, optional): Trainer object containing the model and device.
            model (YOLOEModel, optional): Model to validate. Required if `trainer` is not provided.
            refer_data (str, optional): Path to reference data for visual prompts.
            load_vp (bool): Whether to load visual prompts. If False, text prompts are used.

        Returns:
            (dict): Validation statistics containing metrics computed during validation.
        """
        if trainer is not None:
            self.device = trainer.device
            model = trainer.ema.ema
            names = [name.split("/")[0] for name in list(self.dataloader.dataset.data["names"].values())]

            if load_vp:
                LOGGER.info("Validate using the visual prompt.")
                self.args.half = False
                # Directly use the same dataloader for visual embeddings extracted during training
                vpe = self.get_visual_pe(self.dataloader, model)
                model.set_classes(names, vpe)
            else:
                LOGGER.info("Validate using the text prompt.")
                tpe = model.get_text_pe(names)
                model.set_classes(names, tpe)
            stats = super().__call__(trainer, model)
        else:
            if refer_data is not None:
                assert load_vp, "Refer data is only used for visual prompt validation."
            self.device = select_device(self.args.device)

            if isinstance(model, str):
                from ultralytics.nn.tasks import attempt_load_weights

                model = attempt_load_weights(model, device=self.device, inplace=True)
            model.eval().to(self.device)
            data = check_det_dataset(refer_data or self.args.data)
            names = [name.split("/")[0] for name in list(data["names"].values())]

            if load_vp:
                LOGGER.info("Validate using the visual prompt.")
                self.args.half = False
                # TODO: need to check if the names from refer data is consistent with the evaluated dataset
                # could use same dataset or refer to extract visual prompt embeddings
                # dataloader = self.get_vpe_dataloader(data)
                # vpe = self.get_visual_pe(dataloader, model)
                # model.set_classes(names, vpe)
                stats = self.__BaseValidator__call__(model=deepcopy(model))
            elif isinstance(model.model[-1], YOLOEDetect) and hasattr(model.model[-1], "lrpc"):  # prompt-free
                return self.__BaseValidator__call__(trainer, model)
            else:
                LOGGER.info("Validate using the text prompt.")
                tpe = model.get_text_pe(names)
                model.set_classes(names, tpe)
                stats = self.__BaseValidator__call__(model=deepcopy(model))
        return stats

    def __BaseValidator__call__(self, trainer=None, model=None):
        """
        Execute validation process, running inference on dataloader and computing performance metrics.

        Args:
            trainer (object, optional): Trainer object that contains the model to validate.
            model (nn.Module, optional): Model to validate if not using a trainer.

        Returns:
            stats (dict): Dictionary containing validation statistics.
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # Force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            # self.model = model
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not (pt or jit or getattr(model, "dynamic", False)):
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            elif self.args.task == "detect":
                self.data = check_det_dataset(self.args.data['val']['yolo_data'][0])
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not (pt or getattr(model, "dynamic", False)):
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        
        names = [name.split("/")[0] for name in list(self.dataloader.dataset.data["names"].values())]
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for batch_i, batch in enumerate(bar):
                self.run_callbacks("on_val_batch_start")
                self.batch_i = batch_i
                # Preprocess
                with dt[0]:
                    batch = self.preprocess(batch)

                # Inference
                with dt[1]:
                    cls_visual_num = torch.zeros(len(names))
                    cls_visual_num = cls_visual_num.to(self.device)
                    # visual_pe = torch.zeros(len(names), model.model.model[-1].embed, device=self.device)
                    
                    origin_vpe = model.model.get_visual_pe(batch["img"], visual=batch["visuals"])  # (B, max_n, embed_dim)
                    vpe = torch.zeros(origin_vpe.shape[0], len(names), origin_vpe.shape[-1], device=self.device)
                    class_order, origin = batch["cls"].unique(sorted=True, return_inverse=True)
                    
                    class_order_list = class_order.to(int).tolist()
                    for gg_idx in range(len(names)):
                        if gg_idx in class_order_list:
                            vpe[0][gg_idx] = origin_vpe[0][class_order_list.index(gg_idx)]
                    
                    # visual_pe = torch.zeros((len(names), preds[0].shape[-1]), device=self.device)
                    # batch_idx = batch["batch_idx"]
                    
                    # cls = batch["cls"].squeeze(-1).to(torch.int).unique()
                    # count = torch.bincount(cls, minlength=len(names))
                    # cls_visual_num += count
                    
                    # for i in range(preds[0].shape[0]):
                    #     cls = batch["cls"][batch_idx == i].squeeze(-1).to(torch.int).unique(sorted=True)
                    #     pad_cls = torch.ones(preds[0].shape[1], device=self.device) * -1
                    #     pad_cls[: len(cls)] = cls
                        
                    #     for c in cls:
                    #         visual_pe[c] = preds[i][0][pad_cls == c].sum(0)
                    #     # for c in cls:
                    #     #     visual_pe[c] += preds[i][pad_cls == c].sum(0) / cls_visual_num[c]
                    # visual_pe[cls_visual_num != 0] = F.normalize(visual_pe[cls_visual_num != 0], dim=-1, p=2)
                    # visual_pe[cls_visual_num == 0] = 0
                    # vpe = visual_pe.unsqueeze(0)
                    
                    model.model.set_classes(names, vpe)
                    preds = model(batch["img"], augment=augment)

                # Loss
                with dt[2]:
                    if self.training:
                        self.loss += model.loss(batch, preds)[1]

                # Postprocess
                with dt[3]:
                    preds = self.postprocess(preds)

                self.update_metrics(preds, batch)
                if self.args.plots and batch_i < 3:
                    self.plot_val_samples(batch, batch_i)
                    self.plot_predictions(batch, preds, batch_i)

                self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats


class YOLOESegValidator(YOLOEDetectValidator, SegmentationValidator):
    """YOLOE segmentation validator that supports both text and visual prompt embeddings."""

    pass
