import numpy as np
import torch
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
import sys

class maP50_Validator:
    def __init__(self, names, save_dir=None, imgsz=1024, conf=0.001, iou=0.7, max_det=300, agnostic=False):
        self.nc = len(names)
        # self.names = names
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.agnostic = agnostic

        self.metrics = DetMetrics(save_dir=save_dir, on_plot=BaseValidator.on_plot)
        self.metrics.names = names
        self.metrics.plot = False
        # self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=0.001)
        self.lb = []
        # self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.imgsz = imgsz
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

    def init_metrics(self):
        """Initialize evaluation metrics for YOLO."""
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.conf)
        # print("self.nc:", self.nc)
        # print("self.args.conf:", self.args.conf)
        self.seen = 0
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        # self.device = device

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.conf,
            self.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.agnostic,
            max_det=self.max_det,
        )
    
    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou.detach())

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = (1024, 1024)
        # ratio_pad = batch["ratio_pad"][si]
        # print("ori_shape:", ori_shape, type(ori_shape))
        # print("imgsz:", imgsz)#, imgsz.shape)
        # print("ratio_pad:", ratio_pad)#, ratio_pad.shape)
        # sys.exit()
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape)#, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz}#, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"] #, ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    # if self.args.plots:
                    #     self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            # if self.args.single_cls:
            #     pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
            # if self.args.plots:
            #     self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])
    
    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        return self.metrics.results_dict
    
    def check_stats(self, stats):
        """Checks statistics."""
        pass

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)
    
    def print_results(self):
        """Prints training/validation set metrics per class."""
        # print("self.seen:", self.seen)
        # print("self.nt_per_class.sum(): ", self.nt_per_class.sum())
        # print("*self.metrics.mean_results(): ", *self.metrics.mean_results())
        result = self.metrics.mean_results()
        print(f"Images:{self.seen}, Instances:{self.nt_per_class.sum()}, Box(P:{result[0]}, R:{result[1]}, mAP50:{result[2]}, mAP50-95:{result[3]})")
        # sys.exit()