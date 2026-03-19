"""
PyTorch Combined (Reorder + MAE) algorithm for depression detection.

Implements the three-head model:
- M5 (dl_torch_combined): Classification + reorder + masked reconstruction

Loss: L = Lc + alpha * Lreorder + beta * Lrecon
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
    BehavioralTransformerEncoder,
    LabelHead,
    ReorderHead,
    MaskedReconstructionHead,
    ModalityMaskEmbedding,
)
from utils.torch_training import (
    CombinedDataset,
    MODALITY_INDICES,
    TorchTrainer,
)


def _get_modality_dims(modality_indices):
    """Get {modality_name: n_features} from the index mapping."""
    return {name: len(indices) for name, indices in modality_indices.items()}


class DepressionDetectionClassifier_DL_torch_combined(
    DepressionDetectionClassifier_DL_torch
):
    """PyTorch Combined classifier with classification + reorder + reconstruction heads."""

    def __init__(self, config):
        self.modality_indices = config.get("modality_indices", MODALITY_INDICES)
        self.modality_dims = _get_modality_dims(self.modality_indices)
        super().__init__(config)
        self.mask_embeddings = ModalityMaskEmbedding(self.modality_dims).to(self.device)

    def _build_model_parts(self):
        mp = self.model_params
        n_features = mp["input_shape"][1]
        d_model = mp.get("d_model", 64)

        backbone = BehavioralTransformerEncoder(
            n_features=n_features,
            d_model=d_model,
            num_heads=mp.get("num_heads", 8),
            ff_dim=mp.get("ff_dim", 128),
            num_layers=mp.get("num_layers", 4),
            dropout=mp.get("dropout", 0.1),
        )

        num_reorder_classes = mp.get("num_reorder_class", 200)
        label_head = LabelHead(d_model, num_classes=2)
        reorder_head = ReorderHead(d_model, num_reorder_classes)
        recon_head = MaskedReconstructionHead(
            d_model=d_model,
            modality_dims=self.modality_dims,
            hidden_dim=mp.get("recon_hidden_dim", 64),
        )

        return {
            "backbone": backbone,
            "label_head": label_head,
            "reorder_head": reorder_head,
            "recon_head": recon_head,
        }

    def _make_train_loader(self, training_data):
        """Create DataLoader with combined reorder + modality masking."""
        mp = self.model_params
        dataset = CombinedDataset(
            X=training_data.train_X,
            y=training_data.train_y,
            mixup_alpha=self.config["data_loader"].get("mixup_alpha"),
            num_reorder_classes=mp.get("num_reorder_class", 200),
            rate_of_reorder=mp.get("rate_of_reorder", 0.7),
            modality_indices=self.modality_indices,
            max_masked=mp.get("max_masked_modalities", 2),
        )
        return DataLoader(
            dataset,
            batch_size=self.config["data_loader"].get("batch_size", 32),
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )

    def _train(self, training_data):
        """Train with combined three-head loop."""
        train_loader = self._make_train_loader(training_data)
        eval_val = (training_data.val_X, training_data.val_y)
        eval_test = None
        if training_data.test_X is not None:
            eval_test = (training_data.test_X, training_data.test_y)

        self.trainer = TorchTrainer(
            model_parts=self.model_parts,
            training_params=self.training_params,
            device=self.device,
            extra_params=self.mask_embeddings.parameters(),
        )

        weight_of_reorder = self.model_params.get("weight_of_reorder", 0.2)
        beta = self.model_params.get("beta", 0.2)

        return self.trainer.train_combined(
            train_loader,
            weight_of_reorder=weight_of_reorder,
            beta=beta,
            mask_embeddings=self.mask_embeddings,
            modality_indices=self.modality_indices,
            eval_data_val=eval_val,
            eval_data_test=eval_test,
        )


class DepressionDetectionAlgorithm_DL_torch_combined(
    DepressionDetectionAlgorithm_DL_torch
):
    """PyTorch Combined algorithm."""

    def __init__(self, config_dict=None, config_name="dl_torch_combined"):
        super().__init__(config_dict=config_dict, config_name=config_name)

    def prep_model(self, data_train=None, criteria="balanced_acc"):
        self.config["model_params"].update({"input_shape": self.input_shape})
        return DepressionDetectionClassifier_DL_torch_combined(config=self.config)
