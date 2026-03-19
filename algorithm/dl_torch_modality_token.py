"""
PyTorch Modality-Token Hybrid algorithm for depression detection.

Implements per-modality CNN encoders with cross-modality attention:
- M7  (dl_torch_modality_token): ERM classification only
- M7r (dl_torch_modality_token_reorder): classification + reorder

The architecture splits the 54-feature input into 6 modality sub-sequences,
encodes each with an independent small CNN, then fuses via transformer
cross-attention over the 6 modality tokens.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader

from algorithm.dl_torch_base import (
    DepressionDetectionAlgorithm_DL_torch,
    DepressionDetectionClassifier_DL_torch,
)
from utils.network_torch import (
    ModalityTokenBackbone,
    LabelHead,
    ReorderHead,
)
from utils.torch_training import (
    MODALITY_INDICES,
    ReorderDataset,
    TorchTrainer,
)


class DepressionDetectionClassifier_DL_torch_modality_token(
    DepressionDetectionClassifier_DL_torch
):
    """Modality-Token Hybrid classifier."""

    def _build_model_parts(self):
        mp = self.model_params
        modality_indices = mp.get("modality_indices", MODALITY_INDICES)
        d_token = mp.get("d_token", 32)

        backbone = ModalityTokenBackbone(
            modality_indices=modality_indices,
            conv_channels=mp.get("conv_channels", 8),
            d_token=d_token,
            num_cross_layers=mp.get("num_cross_layers", 2),
            num_heads=mp.get("num_heads", 4),
            ff_dim=mp.get("ff_dim", 64),
            dropout=mp.get("cross_dropout", 0.1),
        )
        cls_head = LabelHead(d_token, num_classes=2)

        return {
            "backbone": backbone,
            "cls_head": cls_head,
        }


class DepressionDetectionClassifier_DL_torch_modality_token_reorder(
    DepressionDetectionClassifier_DL_torch
):
    """Modality-Token Hybrid classifier with reorder auxiliary task."""

    def _build_model_parts(self):
        mp = self.model_params
        modality_indices = mp.get("modality_indices", MODALITY_INDICES)
        d_token = mp.get("d_token", 32)

        backbone = ModalityTokenBackbone(
            modality_indices=modality_indices,
            conv_channels=mp.get("conv_channels", 8),
            d_token=d_token,
            num_cross_layers=mp.get("num_cross_layers", 2),
            num_heads=mp.get("num_heads", 4),
            ff_dim=mp.get("ff_dim", 64),
            dropout=mp.get("cross_dropout", 0.1),
        )
        num_reorder_classes = mp.get("num_reorder_class", 200)
        label_head = LabelHead(d_token, num_classes=2)
        reorder_head = ReorderHead(d_token, num_reorder_classes)

        return {
            "backbone": backbone,
            "label_head": label_head,
            "reorder_head": reorder_head,
        }

    def _make_train_loader(self, training_data):
        mp = self.model_params
        dataset = ReorderDataset(
            X=training_data.train_X,
            y=training_data.train_y,
            mixup_alpha=self.config["data_loader"].get("mixup_alpha"),
            num_reorder_classes=mp.get("num_reorder_class", 200),
            rate_of_reorder=mp.get("rate_of_reorder", 0.7),
            augmentation=self.config.get("augmentation"),
        )
        return DataLoader(
            dataset,
            batch_size=self.config["data_loader"].get("batch_size", 512),
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )

    def _train(self, training_data):
        train_loader = self._make_train_loader(training_data)
        eval_val = (training_data.val_X, training_data.val_y)
        eval_test = None
        if training_data.test_X is not None:
            eval_test = (training_data.test_X, training_data.test_y)

        self.trainer = TorchTrainer(
            model_parts=self.model_parts,
            training_params=self.training_params,
            device=self.device,
        )

        weight_of_reorder = self.model_params.get("weight_of_reorder", 0.2)
        return self.trainer.train_reorder(
            train_loader,
            weight_of_reorder=weight_of_reorder,
            eval_data_val=eval_val,
            eval_data_test=eval_test,
        )


class DepressionDetectionAlgorithm_DL_torch_modality_token(
    DepressionDetectionAlgorithm_DL_torch
):
    """PyTorch Modality-Token ERM algorithm."""

    def __init__(self, config_dict=None, config_name="dl_torch_modality_token"):
        super().__init__(config_dict=config_dict, config_name=config_name)

    def prep_model(self, data_train=None, criteria="balanced_acc"):
        self.config["model_params"].update({"input_shape": self.input_shape})
        return DepressionDetectionClassifier_DL_torch_modality_token(config=self.config)


class DepressionDetectionAlgorithm_DL_torch_modality_token_reorder(
    DepressionDetectionAlgorithm_DL_torch
):
    """PyTorch Modality-Token + Reorder algorithm."""

    def __init__(self, config_dict=None, config_name="dl_torch_modality_token_reorder"):
        super().__init__(config_dict=config_dict, config_name=config_name)

    def prep_model(self, data_train=None, criteria="balanced_acc"):
        self.config["model_params"].update({
            "input_shape": self.input_shape,
            "flag_return_embedding": True,
            "flag_embedding_norm": False,
            "flag_input_dict": True,
        })
        return DepressionDetectionClassifier_DL_torch_modality_token_reorder(
            config=self.config
        )
