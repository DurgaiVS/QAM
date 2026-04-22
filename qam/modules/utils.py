import tarfile
from pathlib import Path
from typing import Union

import torch
from omegaconf import OmegaConf

from .model.conformer_for_classification import ConformerForClassification
from .model.ppo_network import PPONetwork


def load_rl_network_from_ssl_conformer(
    pt_modelpath: Union[str, Path], ppo_network: PPONetwork
):
    """
    Loads the weights of a pre-trained SSL Conformer model into the PPONetwork. Provided
    the number of layers in the front and back encoders of the PPONetwork sum up to the total
    number of layers in the SSL model.

    Parameters
    ----------
    pt_modelpath: Union[str, Path]
        Path to the pre-trained SSL Conformer model. The tar output generated from the
        training script of SSL Conformer.
    rl_network: RLNetwork
        The PPONetwork into which the weights of the SSL model will be loaded.
    """
    with tarfile.open(pt_modelpath, "r:gz") as tar:
        cfg = OmegaConf.load(tar.extractfile("hparams.yaml"))
        weights = torch.load(tar.extractfile("model.pt"), map_location="cpu")

    ssl_model = ConformerForClassification.from_cfg(cfg.model)
    ssl_model.load_state_dict(weights)

    front_encoder_length = len(ppo_network.front_encoder.layers)
    back_encoder_length = len(ppo_network.back_encoder.layers)

    assert front_encoder_length + back_encoder_length == len(
        ssl_model.encoder.layers
    ), (
        f"Total number of layers in RL network ({front_encoder_length + back_encoder_length}) does not match "
        f"the number of layers in SSL model ({len(ssl_model.encoder.layers)})."
    )

    for _rl_sub, _ssl_sub in zip(
        ppo_network.front_encoder.layers, ssl_model.encoder.layers
    ):
        _rl_sub.load_state_dict(_ssl_sub.state_dict())

    for _rl_sub, _ssl_sub in zip(
        ppo_network.back_encoder.layers, ssl_model.encoder.layers[front_encoder_length:]
    ):
        _rl_sub.load_state_dict(_ssl_sub.state_dict())
