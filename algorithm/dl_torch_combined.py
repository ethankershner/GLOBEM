"""
PyTorch Combined (Reorder + MAE) algorithm for depression detection.

Implements the three-head model:
- M5 (dl_torch_combined): Classification + reorder + masked reconstruction

To be implemented in Phase 6.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithm.dl_torch_base import (
    DepressionDetectionAlgorithm_DL_torch,
    DepressionDetectionClassifier_DL_torch,
)


class DepressionDetectionClassifier_DL_torch_combined(
    DepressionDetectionClassifier_DL_torch
):
    """PyTorch Combined classifier. To be implemented in Phase 6."""

    def _build_model_parts(self):
        raise NotImplementedError("Combined classifier not yet implemented (Phase 6)")


class DepressionDetectionAlgorithm_DL_torch_combined(
    DepressionDetectionAlgorithm_DL_torch
):
    """PyTorch Combined algorithm. To be implemented in Phase 6."""

    def __init__(self, config_dict=None, config_name="dl_torch_combined"):
        super().__init__(config_dict=config_dict, config_name=config_name)

    def prep_model(self, data_train=None, criteria="balanced_acc"):
        self.config["model_params"].update({"input_shape": self.input_shape})
        return DepressionDetectionClassifier_DL_torch_combined(config=self.config)
