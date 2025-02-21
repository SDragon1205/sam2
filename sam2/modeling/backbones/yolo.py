import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO

# class yolo(YOLO):
#     def __init__(self, model="yolo11n.pt", task=None, verbose=False):
#         super().__init__(model=model, task=task, verbose=verbose)

#     def forward_backbone(self, x: torch.Tensor):
#         """
#         Perform a forward pass through the network.

#         Args:
#             x (torch.Tensor): The input tensor to the model.

#         Returns:
#             (torch.Tensor): The last output of the model.
#         """
#         y = []  # outputs
#         backbone_fpn = []
#         for i_m in range(10):
#             m = self.detection_model.model[i_m]
#             if m.f != -1:  # if not from previous layer
#                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
#             x = m(x)  # run
#             y.append(x if m.i in self.detection_model.save else None)  # save output
#             if m.i in self.detection_model.save:
#                 backbone_fpn.append(x)
#         output = {
#             "backbone_fpn": backbone_fpn,
#             # "vision_pos_enc": pos,
#         }

#         # for i_y in range(len(y)):
#         #     if y[i_y] == None:
#         #         print(f"y[{i_y}]: None")
#         #     else:
#         #         print(f"y[{i_y}]: {y[i_y].shape}")
#         # for i_b in range(len(backbone_fpn)):
#         #     if backbone_fpn[i_b] == None:
#         #         print(f"backbone_fpn[{i_b}]: None")
#         #     else:
#         #         print(f"backbone_fpn[{i_b}]: {backbone_fpn[i_b].shape}")
#         # sys.exit()

#         return output

#     def forward_neck_head(self, x_list):
#         """
#         Perform a forward pass through the network.

#         Args:
#             y (torch.Tensor): The input tensor to the model.

#         Returns:
#             (torch.Tensor): The last output of the model.
#         """
#         y = []
#         i_x = 0
#         for i_m in range(10):
#             m = self.detection_model.model[i_m]
#             if m.i in self.detection_model.save:
#                 y.append(x_list[i_x])
#                 i_x = i_x + 1
#             else:
#                 y.append(None)
#         x = x_list[2]
#         for i_m in range(10, 23):
#             m = self.detection_model.model[i_m]
#             if m.f != -1:  # if not from previous layer
#                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
#             x = m(x)  # run
#             y.append(x if m.i in self.detection_model.save else None)  # save output

#         return x

class yolo(nn.Module):
    def __init__(
        self,
        cfg: str,#="yolov8s.yaml",
        nc: int,#=10,
        position_encoding: nn.Module,
    ):
        super().__init__()
        self.detection_model = DetectionModel(cfg=cfg, ch=3, nc=nc, verbose=True)
        self.position_encoding = position_encoding

    def forward_backbone(self, x: torch.Tensor):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y = []  # outputs
        pos = []
        backbone_fpn = []
        for i_m in range(10):
            m = self.detection_model.model[i_m]
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.detection_model.save else None)  # save output
            if m.i in self.detection_model.save:
                backbone_fpn.append(x)
                pos.append(self.position_encoding(x).to(x.dtype))
        output = {
            "backbone_fpn": backbone_fpn,
            "vision_pos_enc": pos,
        }

        # for i_y in range(len(y)):
        #     if y[i_y] == None:
        #         print(f"y[{i_y}]: None")
        #     else:
        #         print(f"y[{i_y}]: {y[i_y].shape}")
        # for i_b in range(len(backbone_fpn)):
        #     if backbone_fpn[i_b] == None:
        #         print(f"backbone_fpn[{i_b}]: None")
        #         print(f"backbone_fpn[{i_b}]: None")
        #     else:
        #         print(f"backbone_fpn[{i_b}]: {backbone_fpn[i_b].shape}")
        #         print(f"pos[{i_b}]: {pos[i_b].shape}")
        # sys.exit()

        return output

    def forward_neck_head(self, x_list):
        """
        Perform a forward pass through the network.

        Args:
            y (torch.Tensor): The input tensor to the model.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y = []
        i_x = 0
        for i_m in range(10):
            m = self.detection_model.model[i_m]
            if m.i in self.detection_model.save:
                y.append(x_list[i_x])
                i_x = i_x + 1
            else:
                y.append(None)
        x = x_list[2]
        for i_m in range(10, 23):
            m = self.detection_model.model[i_m]
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.detection_model.save else None)  # save output

        return x