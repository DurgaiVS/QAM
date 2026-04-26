import hydra
import torch
from nemo.collections.asr.modules import ConformerEncoder, ConvASRDecoderClassification

from ..nn.multi_layer_perceptron import MultiLayerPerceptron


class PolicyNetwork(torch.nn.Module):
    def __init__(
        self,
        front_encoder: ConformerEncoder,
        back_encoder: ConformerEncoder,
        mlp_head: MultiLayerPerceptron,
        policy_head: ConvASRDecoderClassification,
    ):
        super().__init__()
        self.front_encoder = front_encoder
        self.back_encoder = back_encoder
        self.mlp_head = mlp_head
        self.policy_head = policy_head

    def forward(self, input_tensor, input_length, state_point):
        enc_op, op_len, *_ = self.front_encoder(
            audio_signal=input_tensor.transpose(1, 2), length=input_length
        )

        mlp_op = self.mlp_head(state_point)
        mask = (state_point != 0).any(dim=-1)
        mlp_op *= mask.unsqueeze(-1)

        ####################################
        # TODO: Explore better ways to combine
        #       front encoder and MLP output.
        enc_op = enc_op + mlp_op.sum(dim=1).unsqueeze(1).transpose(1, 2)
        ####################################

        enc_op, *_ = self.back_encoder(audio_signal=enc_op, length=op_len)
        policy_output = self.policy_head(enc_op)

        return policy_output

    @classmethod
    def from_cfg(cls, cfg):
        front_encoder = hydra.utils.instantiate(cfg.front_encoder)
        back_encoder = hydra.utils.instantiate(cfg.back_encoder)
        mlp_head = hydra.utils.instantiate(cfg.mlp_head)
        policy_head = hydra.utils.instantiate(cfg.policy_head)

        return cls(front_encoder, back_encoder, mlp_head, policy_head)

    def sample_input(self, batch_size, model_max_length):
        sample_input = torch.randn(
            batch_size, model_max_length, self.front_encoder._feat_in
        )
        sample_input_length = torch.zeros(
            batch_size, torch.randint(0, 10, (1)), dtype=torch.int32
        ).fill_(model_max_length)
        sample_state_point = torch.randn(batch_size, self.mlp_head._feat_in)
        return sample_input, sample_input_length, sample_state_point

    def export_onnx(self, save_path, batch_size, model_max_length):
        torch.onnx.export(
            self,
            f=save_path,
            args=self.sample_input(batch_size, model_max_length),
            input_names=["inputs", "ip_lengths", "state_point"],
            output_names=["logits"],
            dynamic_axes={
                "inputs": {0: "batch_size", 1: "ip_seq_len", 2: "embed_dim"},
                "ip_lengths": {0: "ip_seq_len"},
                "state_point": {0: "batch_size", 1: "buy_points", 2: "state_point_dim"},
                "logits": {0: "batch_size", 1: "label_probs"},
            },
        )

    def get_torchscript_model(self, batch_size, model_max_length):
        scripted_model = torch.jit.trace(
            self, example_inputs=self.sample_input(batch_size, model_max_length)
        )
        return scripted_model
