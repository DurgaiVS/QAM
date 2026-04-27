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

    def sample_input(self, batch_size, model_max_length):
        sample_input = torch.randn(batch_size, model_max_length, self.encoder._feat_in)
        sample_input_length = torch.zeros(batch_size, dtype=torch.int32).fill_(
            model_max_length
        )
        return sample_input, sample_input_length

    def export_onnx(self, save_path, batch_size, model_max_length):
        torch.onnx.export(
            self,
            f=save_path,
            args=self.sample_input(batch_size, model_max_length),
            input_names=["input", "input_length"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "ip_seq_len"},
                "input_length": {0: "batch_size"},
                "logits": {0: "batch_size", 1: "label_probs"},
            },
            verify=True,
            optimize=False,
        )

    def get_torchscript_model(self, batch_size, model_max_length):
        scripted_model = torch.jit.trace(
            self, example_inputs=self.sample_input(batch_size, model_max_length)
        )
        return scripted_model
