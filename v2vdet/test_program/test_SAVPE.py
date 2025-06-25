from ultralytics.nn.modules.block import SAVPE
import torch

ch = [256, 512, 512]
c3 = 256
embed=512
bs = 1
image_size = 80
x = [torch.zeros(bs, 256 if i==0 else 512, image_size//(2**i), image_size//(2**i)) for i in range(3)]
vp = torch.zeros(bs, 10, image_size, image_size)

savpe_layer = SAVPE(ch = [256, 512, 512], c3=256, embed=512)
savpe_layer(x, vp)