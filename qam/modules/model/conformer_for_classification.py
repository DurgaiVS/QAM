import hydra
import torch
from nemo.collections.asr.modules import ConformerEncoder, ConvASRDecoderClassification


class ConformerForClassification(torch.nn.Module):
    def __init__(
        self, encoder: ConformerEncoder, decoder: ConvASRDecoderClassification
    ):
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_tensor, input_length):
        enc_op, *_ = self.encoder(input_tensor, input_length)
        return self.decoder(enc_op)

    @classmethod
    def from_cfg(cls, cfg):
        encoder = hydra.utils.instantiate(cfg.encoder)
        decoder = hydra.utils.instantiate(cfg.decoder)

        return cls(encoder, decoder)
