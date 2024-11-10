import torch
from omegaconf import OmegaConf

from qam.modules.encoder import ConfEncoderWithClassificationHeads

hparams = OmegaConf.load("")
model = ConfEncoderWithClassificationHeads(**hparams.model)
model.to(torch.float32)
op, l = model(torch.randn(2, 512, 9).to(torch.float32))
print(op.shape, l)
