import hydra
import torch
from nemo.collections.asr.modules import ConformerEncoder, ConvASRDecoderClassification

from ..nn.multi_layer_perceptron import MultiLayerPerceptron


class RLNetwork(torch.nn.Module):
    def __init__(
        self,
        front_encoder: ConformerEncoder,
        back_encoder: ConformerEncoder,
        mlp_head: MultiLayerPerceptron,
        policy_head: ConvASRDecoderClassification,
        value_head: ConvASRDecoderClassification,
    ):
        super().__init__()
        self.front_encoder = front_encoder
        self.back_encoder = back_encoder
        self.mlp_head = mlp_head
        self.policy_head = policy_head
        self.value_head = value_head

    def forward(self, input_tensor, input_length, state_point):
        enc_op, op_len, *_ = self.front_encoder(input_tensor, input_length)
        mlp_op = self.mlp_head(state_point)

        ####################################
        # TODO: Explore better ways to combine
        #       the encoder output and the MLP
        #       output.
        enc_op = enc_op + mlp_op.unsqueeze(1)
        ####################################

        enc_op, *_ = self.back_encoder(enc_op, op_len)
        policy_output = self.policy_head(enc_op)
        value_output = self.value_head(enc_op)

        return policy_output, value_output

    @classmethod
    def from_cfg(cls, cfg):
        front_encoder = hydra.utils.instantiate(cfg.front_encoder)
        back_encoder = hydra.utils.instantiate(cfg.back_encoder)
        mlp_head = hydra.utils.instantiate(cfg.mlp_head)
        policy_head = hydra.utils.instantiate(cfg.policy_head)
        value_head = hydra.utils.instantiate(cfg.value_head)

        return cls(front_encoder, back_encoder, mlp_head, policy_head, value_head)
