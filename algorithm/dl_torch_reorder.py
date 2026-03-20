"""
PyTorch Reorder algorithm for depression detection.

Implements the Reorder multi-task learning approach using PyTorch:
- M0 (dl_torch_reorder_cnn): CNN backbone, replicating original published Reorder
- M3 (dl_torch_reorder_transformer): Transformer backbone, same Reorder strategy
- M6 (dl_torch_reorder_deep): Deeper transformer, same Reorder strategy

The algorithm trains jointly on classification + temporal reorder prediction.
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
    CNN1D_Backbone,
    BehavioralTransformerEncoder,
    LabelHead,
    ReorderHead,
    build_reorder_model,
)
from utils.torch_training import ReorderDataset, TorchTrainer


class DepressionDetectionClassifier_DL_torch_reorder(
    DepressionDetectionClassifier_DL_torch
):
    """PyTorch Reorder classifier with multi-task classification + reorder heads."""

    def _build_model_parts(self):
        mp = self.model_params
        n_features = mp["input_shape"][1]
        arch = mp.get("arch", "1dCNN")

        if arch == "1dCNN":
            backbone = CNN1D_Backbone(
                n_features=n_features,
                conv_shapes=mp.get("conv_shapes", [8, 8, 8]),
                embedding_size=mp.get("embedding_size", 16),
                dropout=mp.get("cnn_dropout", 0.25),
            )
            embed_dim = mp.get("embedding_size", 16)
        elif arch == "Transformer":
            backbone = BehavioralTransformerEncoder(
                n_features=n_features,
                d_model=mp.get("d_model", 64),
                num_heads=mp.get("num_heads", 8),
                ff_dim=mp.get("ff_dim", 128),
                num_layers=mp.get("num_layers", 4),
                dropout=mp.get("dropout", 0.1),
            )
            embed_dim = mp.get("d_model", 64)
        else:
            raise ValueError(f"Unsupported arch: {arch}")

        num_reorder_classes = mp.get("num_reorder_class", 200)
        label_head = LabelHead(embed_dim, num_classes=2)
        reorder_head = ReorderHead(embed_dim, num_reorder_classes)

        return {
            "backbone": backbone,
            "label_head": label_head,
            "reorder_head": reorder_head,
        }

    def _make_train_loader(self, training_data):
        """Create a DataLoader with ReorderDataset for reorder augmentation."""
        mp = self.model_params
        dataset = ReorderDataset(
            X=training_data.train_X,
            y=training_data.train_y,
            mixup_alpha=self.config["data_loader"].get("mixup_alpha"),
            num_reorder_classes=mp.get("num_reorder_class", 200),
            rate_of_reorder=mp.get("rate_of_reorder", 0.7),
            augmentation=self.config.get("augmentation"),
        )
        batch_size = min(
            self.config["data_loader"].get("batch_size", 512),
            len(dataset),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(dataset) > batch_size,
            num_workers=0,
            pin_memory=True,
        )

    def _train(self, training_data):
        """Train with multi-task reorder loop."""
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


class DepressionDetectionAlgorithm_DL_torch_reorder(
    DepressionDetectionAlgorithm_DL_torch
):
    """PyTorch Reorder algorithm. Extends DL_torch with reorder-specific model building."""

    def __init__(self, config_dict=None, config_name="dl_torch_reorder_cnn"):
        super().__init__(config_dict=config_dict, config_name=config_name)

    def prep_model(self, data_train=None, criteria="balanced_acc"):
        self.config["model_params"].update(
            {
                "input_shape": self.input_shape,
                "flag_return_embedding": True,
                "flag_embedding_norm": False,
                "flag_input_dict": True,
            }
        )
        return DepressionDetectionClassifier_DL_torch_reorder(config=self.config)
