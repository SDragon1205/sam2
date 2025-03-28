import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
from typing import List, Optional, Tuple, Type

class yolo(nn.Module):
    def __init__(
        self,
        cfg: str,#="yolov8s.yaml",
        nc: int,#=10,
        position_encoding: nn.Module,
        has_output_upscaling: bool = False,
        transformer_dim: int = 512,
        norm_type: str = "",
        activation: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.detection_model = DetectionModel(cfg=cfg, ch=3, nc=nc, verbose=True)
        self.position_encoding = position_encoding
        self.has_output_upscaling = has_output_upscaling
        if has_output_upscaling:
            if norm_type == "group":
                self.output_upscaling = nn.Sequential(
                    nn.ConvTranspose2d(
                        transformer_dim, transformer_dim // 2, kernel_size=2, stride=2
                    ),
                    nn.GroupNorm(num_groups=32, num_channels=transformer_dim // 2),
                    activation(),
                    nn.ConvTranspose2d(
                        transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2
                    ),
                    activation(),
                )
            elif norm_type == "batch":
                self.output_upscaling = nn.Sequential(
                    nn.ConvTranspose2d(
                        transformer_dim, transformer_dim // 2, kernel_size=2, stride=2
                    ),
                    nn.BatchNorm2d(transformer_dim // 2),
                    activation(),
                    nn.ConvTranspose2d(
                        transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2
                    ),
                    activation(),
                )
            else:
                print("yolo norm_type error!")
                sys.exit()

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
    
    def forward_backbone_neck(self, x: torch.Tensor):
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
        for i_m in range(23):
            # print("i_m:", i_m)
            m = self.detection_model.model[i_m]
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if i_m == 22:
                    # print("x.shape:", x[0].shape, x[1].shape, x[2].shape)
                    # print("type(x)", type(x))
                    for xi in x:
                        # backbone_fpn.append(xi)
                        pos.append(self.position_encoding(xi).to(xi.dtype))
                    # print("pos.shape:", pos[0].shape, pos[1].shape, pos[2].shape)
                    # sys.exit()
                    output = {
                        "backbone_fpn": x,
                        "vision_pos_enc": pos,
                    }
                    return output
            x = m(x)  # run
            y.append(x if m.i in self.detection_model.save else None)  # save output
            # if m.i in self.detection_model.save:
            #     backbone_fpn.append(x)
            #     pos.append(self.position_encoding(x).to(x.dtype))
        output = {
            "backbone_fpn": backbone_fpn,
            "vision_pos_enc": pos,
        }

        return output
    
    def forward_head(self, x_list):
        # print("x_list.shape:", x_list[0].shape, x_list[1].shape, x_list[2].shape)
        m = self.detection_model.model[22]
        if self.has_output_upscaling:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            # x_list_2 = x_list[2].clone()
            upscaled_embedding1 = act1(ln1(dc1(x_list[2]) + x_list[1]))
            upscaled_embedding2 = act2(dc2(upscaled_embedding1) + x_list[0])
            # print("is x_list_2 same", x_list_2 == x_list[2])
            # print("input shape:", upscaled_embedding2.shape, upscaled_embedding1.shape, x_list[2].shape)
            output = m([upscaled_embedding2, upscaled_embedding1, x_list[2]])
        else:
            output = m(x_list)
        # print("output.shape:", output[0].shape, output[1].shape, output[2].shape)
        # sys.exit()
        return output
    
    def forward_backbone_15(self, x: torch.Tensor):
        y = []  # outputs
        pos = []
        backbone_fpn = []
        for i_m in range(16):
            m = self.detection_model.model[i_m]
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.detection_model.save else None)  # save output
            if (m.i in self.detection_model.save) and (i_m == 9 or i_m == 12 or i_m == 15):
                backbone_fpn.append(x)
                pos.append(self.position_encoding(x).to(x.dtype))
                # print(f"{i_m}: {x.shape}")
        backbone_fpn[0], backbone_fpn[2] = backbone_fpn[2], backbone_fpn[0]
        pos[0], pos[2] = pos[2], pos[0]
        # print("backbone_fpn:", backbone_fpn[0].shape, backbone_fpn[1].shape, backbone_fpn[2].shape)
        # print("pos:", pos[0].shape, pos[1].shape, pos[2].shape)
        output = {
            "backbone_fpn": backbone_fpn,
            "vision_pos_enc": pos,
        }
        # for i_y in range(len(y)):
        #     if y[i_y] == None:
        #         print(f"y[{i_y}]: None")
        #     else:
        #         print(f"y[{i_y}]: {y[i_y].shape}")
        # sys.exit()
        return output
    
    def forward_16_head(self, x_list):
        y = []
        for i_m in range(16):
            # m = self.detection_model.model[i_m]
            if i_m == 9:
                y.append(x_list[2])
            elif i_m == 12:
                y.append(x_list[1])
            elif i_m == 15:
                y.append(x_list[0])
            else:
                y.append(None)
        # print("after")
        # for i_y in range(len(y)):
        #     if y[i_y] == None:
        #         print(f"y[{i_y}]: None")
        #     else:
        #         print(f"y[{i_y}]: {y[i_y].shape}")
        x = x_list[0]
        for i_m in range(16, 23):
            m = self.detection_model.model[i_m]
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.detection_model.save else None)  # save output
        # sys.exit()
        return x
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