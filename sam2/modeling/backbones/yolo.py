import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from ultralytics.nn.tasks import DetectionModel, WorldModel
from ultralytics import YOLO
from typing import List, Optional, Tuple, Type
from ultralytics.nn.modules import (
    C2fAttn,
    ImagePoolingAttn,
    WorldDetect,
    YOLOEDetect,
)
from ultralytics.utils.plotting import feature_visualization
import numpy as np
from v2vdet.v2vdet_ultralytics.nn.tasks_oo import V2V_With_MultiScale_SAVPE_ObjectOriented_Model, V2V_With_MultiScale_SAVPE_SigLIP2_B_ObjectOriented_Model
import torchvision.transforms as transforms
from PIL import Image
# from ultralytics.data.augment import LoadVisualPrompt
import v2vdet.v2vdet_ultralytics.perception_models.core.vision_encoder.pe as pe
from ultralytics.utils.ops import xywh2xyxy
# from training.model.yolom import select_frame_gtdata, select_batch
from training.trainer_yolo import visualize_batched_video

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
        world: bool = False,
        oo: bool = False,
    ):
        super().__init__()
        self.nc = nc

        if oo == True:
            detection_model_nc = self.nc
            if cfg == "/home/user/sdragon/sam2/v2vdet/v2vdet_ultralytics/cfg/models/v2v/11/yolo11n-v2v-multiscale_1_3_5.yaml":
                detection_model_nc = 80
            self.detection_model = V2V_With_MultiScale_SAVPE_SigLIP2_B_ObjectOriented_Model(cfg=cfg, ch=3, nc=detection_model_nc, verbose=True)
        elif world == True:
            self.detection_model = WorldModel(cfg=cfg, ch=3, nc=nc, verbose=True)
        else:
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

    def select_frame_gtdata(self, gtdata, select_frame):
        """
        保留 batch_idx 為 select_frame 的資料，並將其 batch_idx 改為 0。
        
        :param gtdata: Dict，包含 batch_idx, cls, bboxes, ori_shape
        :param select_frame: int，要保留的 frame 編號（batch_idx）
        :return: 過濾後的 gtdata
        """
        # 取得 batch_idx == select_frame 的 mask
        mask = gtdata["batch_idx"] == select_frame

        # 過濾資料，並將 batch_idx 改成 0
        gtdata_filtered = {
            "batch_idx": gtdata["batch_idx"][mask] * 0,  # or .fill(0)
            "cls": gtdata["cls"][mask],
            "bboxes": gtdata["bboxes"][mask],
            "ori_shape": [shape for i, shape in enumerate(gtdata["ori_shape"]) if gtdata["batch_idx"][i] == select_frame]
        }

        # 檢查長度一致性
        lengths = {
            "batch_idx": len(gtdata_filtered["batch_idx"]),
            "cls": len(gtdata_filtered["cls"]),
            "bboxes": len(gtdata_filtered["bboxes"]),
            "ori_shape": len(gtdata_filtered["ori_shape"])
        }
        
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Mismatch in lengths: {lengths}")
        
        return gtdata_filtered

    def select_img_batch(self, all_frame_outputs, select_frame):
        """
        保留 all_frame_outputs 中 batch index 為 select_frame 的資料
        並將其當作 batch_idx=0 處理（即剩下的 shape 是 batch size 1）

        :param all_frame_outputs: Tuple or List，第一個 tensor 是 (B, 14, 8400)，
                                第二個是 list of tensors，每個是 (B, 74, h, w)
        :param select_frame: int，要保留的 batch index
        :return: 過濾後的 all_frame_outputs，只保留 batch_idx == select_frame 的資料
        """
        batch_size = all_frame_outputs.shape[0]
        if select_frame >= batch_size:
            raise ValueError(f"select_frame={select_frame} exceeds batch_size={batch_size}")

        # 取出 batch_idx == select_frame 的資料
        selected_outputs = all_frame_outputs[select_frame:select_frame+1]  # shape: (1, 14, 8400)

        return selected_outputs

    def inference_set_classes(self, imgs, gtdata, batch_idx):
        device = next(self.detection_model.vision_encoder.parameters()).device

        transform_vision_encoder = transforms.Compose([
            transforms.Resize((self.detection_model.vision_encoder_patch_size, self.detection_model.vision_encoder_patch_size), antialias=True) #,
            # transforms.ToTensor()
        ])
        transform_visual_prompt = transforms.Compose([
            transforms.Resize((640, 640), antialias=True),
            # transforms.ToTensor()
        ])

        imgs_list = []
        visual_prompt_list = []
        cls_unique_list = []
        for idx in batch_idx:
            pil_img = self.select_img_batch(imgs, idx).squeeze(0)
            # def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
            #     if tensor.ndim != 3 or tensor.shape[0] != 3:
            #         raise ValueError("Expected RGB image tensor with shape [3, H, W]")
            #     img = tensor.permute(1, 2, 0)  # [H, W, C]
            #     img = (img * 255).clamp(0, 255).byte()
            #     return Image.fromarray(img.cpu().numpy())
            # pil_img = tensor_to_pil_image(img)
            # print(f"self.select_img_batch(imgs, {idx}).squeeze(0):", img.shape)
            imgs_list.append(pil_img)
            gt = self.select_frame_gtdata(gtdata, idx)
            # print("gt:", gt)
            visual_prompt_label = {
                'img': transform_visual_prompt(pil_img),
                'bboxes': gt['bboxes'],
                'cls': gt['cls'],
            }

            load_visual_prompt_func = LoadVisualPrompt()
            visual_prompt_label, cls_unique = load_visual_prompt_func(labels=visual_prompt_label, nc=self.nc)
            # visual_prompt_label = load_visual_prompt_func(labels=visual_prompt_label)
            visual_prompt_list.append(visual_prompt_label['visuals'])
            cls_unique_list.append(cls_unique)
            # print("visual_prompt_label['visuals']:", visual_prompt_label['visuals'].shape)

            # from torchvision.transforms.functional import to_pil_image
            # to_pil_image(pil_img.cpu()).save("set_class/img_0.png")
            # visuals = visual_prompt_label['visuals']
            # torch.save(visual_prompt_label['visuals'], "set_class/visuals.pt")
            # for i in range(visuals.shape[0]):
            #     mask = visuals[i]  # shape: [80, 80]
            #     print(f"visuals[{i}]", torch.unique(mask))
            #     pil_img = to_pil_image(mask.cpu().unsqueeze(0))  # 加上 channel 維度，變 [1, 80, 80]
            #     pil_img.save(f"set_class/visual_mask_{i}.png")

            # print("img:", img.shape, torch.max(img), torch.min(img))
            # print("gt", gt)
            # pil_img.save("tmp/inference_set_classes.jpg")  # 儲存圖片
            # print("visual_prompt_label['visuals']:", visual_prompt_label['visuals'].shape)
            # print("visual_prompt_label['visuals'].shape[0]:", visual_prompt_label['visuals'].shape[0])
            # for i_v in range(visual_prompt_label['visuals'].shape[0]):
            #     print(f"visual_prompt_label['visuals'][{i_v}]:", visual_prompt_label['visuals'][i_v].shape, torch.max(visual_prompt_label['visuals'][i_v]), torch.min(visual_prompt_label['visuals'][i_v]))
        # sys.exit()
        visuals_tensor = torch.stack(visual_prompt_list)
        # print("visuals_tensor:", visuals_tensor.shape)

        img_tensor_list = [transform_vision_encoder(img) for img in imgs_list]
        batch_tensor = torch.stack(img_tensor_list)
        # print("batch_tensor:", batch_tensor.shape)

        # For Vision Encoder
        vision_encoder_inputs = {'pixel_values': batch_tensor.to(device)}

        # print("isinstance(self.detection_model.vision_encoder, pe.VisionTransformer):", isinstance(self.detection_model.vision_encoder, pe.VisionTransformer), self.detection_model.vision_encoder)
        if isinstance(self.detection_model.vision_encoder, pe.VisionTransformer):
            vision_model_output = self.detection_model.vision_encoder(vision_encoder_inputs['pixel_values'], output_hidden_list=self.detection_model.want_layers, strip_cls_token=True)
            savpe_input = vision_model_output['hidden_states'].permute(1, 0, 2, 3)
            self.detection_model.vpe = self.detection_model.patch_emb_savpe(savpe_input, visuals_tensor)
        else:
            vision_model_output = self.detection_model.vision_encoder(**vision_encoder_inputs, output_hidden_states=True, return_dict=True)
            hidden_states_list = [[] for _ in range(len(self.detection_model.want_layers))]
            for idx, layer_num in enumerate(self.detection_model.want_layers):
                hidden_states_list[idx] = vision_model_output['hidden_states'][layer_num]
            hidden_states_tensor = torch.stack(tensors=hidden_states_list, dim=1)
            # print("hidden_states_tensor:", hidden_states_tensor.shape)
            self.detection_model.vpe = self.detection_model.patch_emb_savpe(hidden_states_tensor, visuals_tensor)
            # print("self.detection_model.vpe:", self.detection_model.vpe.shape)
        # print("cls_unique:", cls_unique)
        # for i_v in range(self.detection_model.vpe.shape[0]):
        #     for j_v in range(self.detection_model.vpe.shape[1]):
        #         print(f"self.detection_model.vpe[i_v][{j_v}]:", self.detection_model.vpe[i_v][j_v].shape, torch.max(self.detection_model.vpe[i_v][j_v]), torch.min(self.detection_model.vpe[i_v][j_v]))
        
        for i in range(self.detection_model.vpe.shape[0]):
            for j in range(self.detection_model.vpe.shape[1]):  # loop over class indices
                if j not in cls_unique_list[i]:
                    self.detection_model.vpe[i][j] = 0
        
        # for i_v in range(self.detection_model.vpe.shape[0]):
        #     for j_v in range(self.detection_model.vpe.shape[1]):
        #         print(f"after self.detection_model.vpe[i_v][{j_v}]:", self.detection_model.vpe[i_v][j_v].shape, torch.max(self.detection_model.vpe[i_v][j_v]), torch.min(self.detection_model.vpe[i_v][j_v]))

        # print("self.detection_model.vpe.shape[1]:", self.detection_model.vpe.shape[1])
        self.detection_model.nc = self.detection_model.vpe.shape[1]
        self.detection_model.model[-1].nc = self.detection_model.nc
        # sys.exit()

    def forward_backbone_oo(self, x: torch.Tensor, profile=False, visualize=False, augment=False, embed=None, vpe=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        # if vpe is None:
        #     if hasattr(self, "vpe"):
        #         vpe = self.detection_model.vpe
        #     else:
        #         vpe = torch.zeros(1, self.nc, 512)  # features placeholder

        # b = x.shape[0]
        # print("self.detection_model.vision_encoder_patch_size:", self.detection_model.vision_encoder_patch_size)
        # print("self.detection_model.criterion:", self.detection_model.init_criterion())
        y, dt, embeddings = [], [], []  # outputs
        pos = []
        backbone_fpn = []
        # print("forward_backbone_oo x:", x.shape)
        for i_m in range(11):
            m = self.detection_model.model[i_m]
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self.detection_model._profile_one_layer(m, x, dt)

            # if isinstance(m, WorldDetect) or isinstance(m, YOLOEDetect):
            #     cls_pe = self.get_cls_pe(vpe).to(device=x[0].device, dtype=x[0].dtype)
            #     if cls_pe.shape[0] != b or m.export:
            #         cls_pe = cls_pe.expand(b, -1, -1)
            #     x = m(x, cls_pe)
            #     print("error")
            # else:
            x = m(x)  # run
            # print(f"{i_m} x:", x.shape)

            y.append(x if m.i in self.detection_model.save else None)  # save output
            if m.i in self.detection_model.save:
                backbone_fpn.append(x)
                pos.append(self.position_encoding(x).to(x.dtype))
                # print("save")

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        output = {
            "backbone_fpn": backbone_fpn,
            "vision_pos_enc": pos,
        }
        # sys.exit()
        return output
    
    def forward_neck_head_oo(self, x_list, profile=False, visualize=False, augment=False, embed=None, vpe=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if vpe is None:
            # print("hasattr(self, vpe)", hasattr(self, "vpe"))
            # print("hasattr(self.detection_model, vpe)", hasattr(self.detection_model, "vpe"))
            if hasattr(self.detection_model, "vpe"):
                vpe = self.detection_model.vpe
            else:
                vpe = torch.zeros(1, self.nc, 512)  # features placeholder

        y, dt, embeddings = [], [], []  # outputs
        pos = []
        backbone_fpn = []
        i_x = 0
        for i_m in range(11):
            m = self.detection_model.model[i_m]
            if m.i in self.detection_model.save:
                y.append(x_list[i_x])
                i_x = i_x + 1
            else:
                y.append(None)
        x = x_list[2]
        b = x.shape[0]
        # print("x_list[2]:", x_list[2].shape)

        for i_m in range(11, 24):
            m = self.detection_model.model[i_m]
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self.detection_model._profile_one_layer(m, x, dt)

            if isinstance(m, WorldDetect) or isinstance(m, YOLOEDetect):
                cls_pe = self.detection_model.get_cls_pe(vpe).to(device=x[0].device, dtype=x[0].dtype)
                if cls_pe.shape[0] != b or m.export:
                    cls_pe = cls_pe.expand(b, -1, -1)
                # print("isinstance(m, WorldDetect):", isinstance(m, WorldDetect))
                # print("isinstance(m, YOLOEDetect):", isinstance(m, YOLOEDetect))
                # print("m.nc:", m.nc)
                # print("WorldDetect")
                # print("x:", len(x), x[0].shape, x[1].shape, x[2].shape)
                # print("ori_txt_feats:", cls_pe.shape, cls_pe)
                x = m(x, cls_pe)
                # print("mx[0]:", x[0].shape)
                # print("mx[1]:", x[1][0].shape, x[1][1].shape, x[1][2].shape)
                # print(f"{i_m} x:", len(x), x[0].shape, x[1].shape, x[2].shape)
            else:
                x = m(x)  # run
                # print(f"{i_m} x:", x.shape)

            y.append(x if m.i in self.detection_model.save else None)  # save output
            if m.i in self.detection_model.save:
                backbone_fpn.append(x)
                pos.append(self.position_encoding(x).to(x.dtype))

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        # sys.exit()
        return x
    
    def forward_backbone_world(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            txt_feats (torch.Tensor, optional): The text features, use it if it's given.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        # txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        # if len(txt_feats) != len(x) or self.model[-1].export:
        #     txt_feats = txt_feats.expand(x.shape[0], -1, -1)
        # ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        pos = []
        backbone_fpn = []
        for i_m in range(10):
            m = self.detection_model.model[i_m]
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self.detection_model._profile_one_layer(m, x, dt)
            # if isinstance(m, C2fAttn):
            #     print("C2fAttn")
            #     x = m(x, txt_feats)
            # elif isinstance(m, WorldDetect):
            #     print("C2fAttn")
            #     x = m(x, ori_txt_feats)
            # elif isinstance(m, ImagePoolingAttn):
            #     print("C2fAttn")
            #     txt_feats = m(x, txt_feats)
            # else:
            x = m(x)  # run

            y.append(x if m.i in self.detection_model.save else None)  # save output
            if m.i in self.detection_model.save:
                backbone_fpn.append(x)
                pos.append(self.position_encoding(x).to(x.dtype))

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        output = {
            "backbone_fpn": backbone_fpn,
            "vision_pos_enc": pos,
        }
        return output
    
    def forward_neck_head_world(self, x_list, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            txt_feats (torch.Tensor, optional): The text features, use it if it's given.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        # txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x_list[0].device, dtype=x_list[0].dtype)
        if txt_feats is None:
            raise ValueError('txt_feats is None in forward_neck_head_world')
        txt_feats = txt_feats.to(device=x_list[0].device, dtype=x_list[0].dtype)
        # print("txt_feats:", txt_feats.shape, txt_feats)
        # # print("self.detection_model.model[-1].export:", self.detection_model.model[-1].export)
        # sys.exit()
        if len(txt_feats) != len(x_list[0]) or self.detection_model.model[-1].export:
            print("len(txt_feats):", len(txt_feats))
            print("len(x_list[0]):", len(x_list[0]))
            raise ValueError('len(txt_feats) != len(x_list[0]) or self.detection_model.model[-1].export: use clone to input txt_feats on forward_neck_head_world function, 因為其他frame的txt_feats和ori_txt_feats會被影響')
            txt_feats = txt_feats.expand(x_list[0].shape[0], -1, -1)
        ori_txt_feats = txt_feats.clone()
        # print("ori_txt_feats:", ori_txt_feats.shape)
        # sys.exit()

        y, dt, embeddings = [], [], []  # outputs
        i_x = 0
        for i_m in range(10):
            m = self.detection_model.model[i_m]
            if m.i in self.detection_model.save:
                y.append(x_list[i_x])
                i_x = i_x + 1
            else:
                y.append(None)
        x = x_list[2]
        # print("x_list[2]:", x_list[2].shape)

        for i_m in range(10, 23):
            m = self.detection_model.model[i_m]
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self.detection_model._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                # print("C2fAttn")
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                print("WorldDetect")
                print("x:", len(x), x[0].shape, x[1].shape, x[2].shape)
                print("ori_txt_feats:", ori_txt_feats.shape)
                x = m(x, ori_txt_feats)
                print("mx[0]:", x[0].shape)
                print("mx[1]:", x[1][0].shape, x[1][1].shape, x[1][2].shape)
                sys.exit()
            elif isinstance(m, ImagePoolingAttn):
                # print("ImagePoolingAttn")
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.detection_model.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

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

class LoadVisualPrompt:
    """Create visual prompts from bounding boxes or masks for model input."""

    def __init__(self, scale_factor=1 / 8):
        self.scale_factor = scale_factor

    def make_mask(self, boxes, h, w):
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
        r = torch.arange(w)[None, None, :]
        c = torch.arange(h)[None, :, None]
        return (r >= x1) * (r < x2) * (c >= y1) * (c < y2)

    def __call__(self, labels, nc=None):
        imgsz = labels["img"].shape[1:]
        bboxes, masks = None, None
        if "bboxes" in labels:
            bboxes = labels["bboxes"]
            bboxes = xywh2xyxy(bboxes) * torch.tensor(imgsz)[[1, 0, 1, 0]]  # denormalize boxes

        cls = labels["cls"].squeeze(-1).to(torch.int)
        visuals, cls_unique = self.get_visuals(cls, imgsz, bboxes=bboxes, masks=masks, nc=nc)
        labels["visuals"] = visuals
        return labels, cls_unique

    def get_visuals(self, category, shape, bboxes=None, masks=None, nc=None):
        masksz = (int(shape[0] * self.scale_factor), int(shape[1] * self.scale_factor))

        if bboxes is not None:
            if isinstance(bboxes, np.ndarray):
                bboxes = torch.from_numpy(bboxes)
            bboxes *= self.scale_factor
            masks = self.make_mask(bboxes, *masksz).float()
        elif masks is not None:
            if isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks)
            masks = F.interpolate(masks.unsqueeze(1), masksz, mode="nearest").squeeze(1).float()
        else:
            raise ValueError("LoadVisualPrompt must have bboxes or masks in the label")

        if not isinstance(category, torch.Tensor):
            category = torch.tensor(category, dtype=torch.int)

        cls_unique, inverse_indices = torch.unique(category, sorted=True, return_inverse=True)
        # print("cls_unique:", cls_unique)

        # 建立 [nc, H, W] 視覺遮罩（包含未出現的類別也會保留為全 0）
        if nc is not None:
            visuals = torch.zeros(nc, *masksz, dtype=masks.dtype)
            for i, mask in zip(inverse_indices, masks):
                cls_id = cls_unique[i]
                visuals[cls_id] = torch.logical_or(visuals[cls_id], mask)
        else:
            visuals = torch.zeros(len(cls_unique), *masksz, dtype=masks.dtype)
            for i, mask in zip(inverse_indices, masks):
                visuals[i] = torch.logical_or(visuals[i], mask)

        return visuals, cls_unique