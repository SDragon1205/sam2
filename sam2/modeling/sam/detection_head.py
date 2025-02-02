# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.sam2_utils import LayerNorm2d, MLP
from torch.nn import BatchNorm2d
from ultralytics.nn.modules.head import Detect

import sys

class Detection_head(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        activation: Type[nn.Module] = nn.GELU,
        nc=10,
        ch=[256, 64, 32],
        stride=[16., 8., 4.],
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          activation (nn.Module): the type of activation to use when
            upscaling masks
        """
        super().__init__()
        self.detect = Detect(nc=nc, ch=ch)
        self.detect.stride = stride
        # self.detect.export = True
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            BatchNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        
    def forward(
        self,
        image_embeddings: torch.Tensor,
        # image_pe: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings

        Returns:
          List[torch.Tensor]: 3*[(nc+reg_max*4)*H*W] batched undecoded bbox and class
          reg_max: 16
        """
        preds = self.predict(
            image_embeddings=image_embeddings,
            # image_pe=image_pe,
            high_res_features=high_res_features,
        )

        return preds

    def predict(
        self,
        image_embeddings: torch.Tensor,
        # image_pe: torch.Tensor,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        src = image_embeddings
        # assert (
        #     image_pe.size(0) == 1
        # ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        # pos_src = torch.repeat_interleave(image_pe, src.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        # hs, src = self.transformer(src, pos_src, tokens)
        #src = src + pos_src

        # Upscale mask embeddings and predict masks using the mask tokens
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = high_res_features
        upscaled_embedding1 = act1(ln1(dc1(src) + feat_s1))
        upscaled_embedding2 = act2(dc2(upscaled_embedding1) + feat_s0)

        head_input = [src, upscaled_embedding1, upscaled_embedding2]
        # for i_head in range(len(head_input)):
        #     print(f"head_input[{i_head}]: {head_input[i_head].shape}")
        head_output = self.detect(head_input)
        # for i_head in range(len(head_output)):
        #     print(f"head_output[{i_head}]: {head_output[i_head].shape}")
        # sys.exit()
        
        return head_output