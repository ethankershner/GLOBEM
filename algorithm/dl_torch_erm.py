"""
PyTorch ERM (Empirical Risk Minimization) algorithm for depression detection.

Implements classification-only model:
- M2 (dl_torch_erm_transformer): Proper transformer (4 layers, d=64)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithm.dl_torch_base import (
    DepressionDetectionAlgorithm_DL_torch,
    DepressionDetectionClassifier_DL_torch,
)
from utils.network_torch import BehavioralTransformerEncoder, LabelHead


class DepressionDetectionClassifier_DL_torch_erm(
    DepressionDetectionClassifier_DL_torch
):
    """PyTorch ERM classifier (classification-only, no auxiliary heads)."""

    def _build_model_parts(self):
        mp = self.model_params
        n_features = mp["input_shape"][1]

        backbone = BehavioralTransformerEncoder(
            n_features=n_features,
            d_model=mp.get("d_model", 64),
            num_heads=mp.get("num_heads", 8),
            ff_dim=mp.get("ff_dim", 128),
            num_layers=mp.get("num_layers", 4),
            dropout=mp.get("dropout", 0.1),
        )
        cls_head = LabelHead(mp.get("d_model", 64), num_classes=2)

        return {"backbone": backbone, "cls_head": cls_head}


class DepressionDetectionAlgorithm_DL_torch_erm(DepressionDetectionAlgorithm_DL_torch):
    """PyTorch ERM algorithm."""

    def __init__(self, config_dict=None, config_name="dl_torch_erm_transformer"):
        super().__init__(config_dict=config_dict, config_name=config_name)

    def prep_model(self, data_train=None, criteria="balanced_acc"):
        self.config["model_params"].update({"input_shape": self.input_shape})
        return DepressionDetectionClassifier_DL_torch_erm(config=self.config)
