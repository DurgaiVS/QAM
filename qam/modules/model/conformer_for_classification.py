import hydra
import torch
from nemo.collections.asr.modules import ConformerEncoder, ConvASRDecoder


class ConformerForClassification(torch.nn.Module):
    def __init__(self, encoder: ConformerEncoder, decoder: ConvASRDecoder):
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_tensor, input_length):
        return self.decoder(self.encoder(input_tensor, input_length)[0])

    @classmethod
    def from_cfg(cls, cfg):
        encoder = hydra.utils.instantiate(cfg.encoder)
        decoder = hydra.utils.instantiate(cfg.decoder)

        return cls(encoder, decoder)
