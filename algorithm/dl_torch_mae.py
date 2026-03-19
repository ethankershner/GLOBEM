"""
PyTorch MAE (Masked Autoencoder) algorithm for depression detection.

Implements masked modality reconstruction models:
- M4a (dl_torch_mae_simultaneous): Simultaneous classification + reconstruction
- M4b (dl_torch_mae_staged): Pretrain on reconstruction, finetune on classification

To be implemented in Phase 5.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithm.dl_torch_base import (
    DepressionDetectionAlgorithm_DL_torch,
    DepressionDetectionClassifier_DL_torch,
)


class DepressionDetectionClassifier_DL_torch_mae(
    DepressionDetectionClassifier_DL_torch
):
    """PyTorch MAE classifier. To be implemented in Phase 5."""

    def _build_model_parts(self):
        raise NotImplementedError("MAE classifier not yet implemented (Phase 5)")


class DepressionDetectionAlgorithm_DL_torch_mae(DepressionDetectionAlgorithm_DL_torch):
    """PyTorch MAE algorithm. To be implemented in Phase 5."""

    def __init__(self, config_dict=None, config_name="dl_torch_mae_simultaneous"):
        super().__init__(config_dict=config_dict, config_name=config_name)

    def prep_model(self, data_train=None, criteria="balanced_acc"):
        self.config["model_params"].update({"input_shape": self.input_shape})
        return DepressionDetectionClassifier_DL_torch_mae(config=self.config)
