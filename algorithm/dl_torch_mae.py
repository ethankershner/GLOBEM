"""
PyTorch MAE (Masked Autoencoder) algorithm for depression detection.

Implements masked modality reconstruction with staged training:
- M4b (dl_torch_mae_staged): Pretrain on reconstruction, finetune on classification
  Phase 1: Lmask only; Phase 2: Lc only
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
    MaskedReconstructionHead,
    ModalityMaskEmbedding,
)
from utils.torch_training import (
    ModalityMaskingDataset,
    MODALITY_INDICES,
    TorchTrainer,
)


def _get_modality_dims(modality_indices):
    """Get {modality_name: n_features} from the index mapping."""
    return {name: len(indices) for name, indices in modality_indices.items()}


class DepressionDetectionClassifier_DL_torch_mae(
    DepressionDetectionClassifier_DL_torch
):
    """PyTorch MAE classifier with classification + masked reconstruction heads."""

    def __init__(self, config):
        self.modality_indices = config.get("modality_indices", MODALITY_INDICES)
        self.modality_dims = _get_modality_dims(self.modality_indices)
        super().__init__(config)
        # Mask embeddings are separate from model_parts (not needed at inference)
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
        cls_head = LabelHead(d_model, num_classes=2)
        recon_head = MaskedReconstructionHead(
            d_model=d_model,
            modality_dims=self.modality_dims,
            hidden_dim=mp.get("recon_hidden_dim", 64),
        )

        return {
            "backbone": backbone,
            "cls_head": cls_head,
            "recon_head": recon_head,
        }

    def _make_train_loader(self, training_data):
        """Create DataLoader with modality masking."""
        mp = self.model_params
        dataset = ModalityMaskingDataset(
            X=training_data.train_X,
            y=training_data.train_y,
            mixup_alpha=self.config["data_loader"].get("mixup_alpha"),
            modality_indices=self.modality_indices,
            max_masked=mp.get("max_masked_modalities", 2),
            augmentation=self.config.get("augmentation"),
        )
        batch_size = min(
            self.config["data_loader"].get("batch_size", 32),
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
        """Train with staged MAE: pretrain reconstruction, finetune classification."""
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

        return self.trainer.train_mae_staged(
            train_loader,
            mask_embeddings=self.mask_embeddings,
            modality_indices=self.modality_indices,
            eval_data_val=eval_val,
            eval_data_test=eval_test,
            pretrain_epochs=self.training_params.get("pretrain_epochs"),
            pretrain_patience=self.training_params.get("pretrain_patience", 10),
        )


class DepressionDetectionAlgorithm_DL_torch_mae(DepressionDetectionAlgorithm_DL_torch):
    """PyTorch MAE algorithm."""

    def __init__(self, config_dict=None, config_name="dl_torch_mae_transformer"):
        super().__init__(config_dict=config_dict, config_name=config_name)

    def prep_model(self, data_train=None, criteria="balanced_acc"):
        self.config["model_params"].update({"input_shape": self.input_shape})
        return DepressionDetectionClassifier_DL_torch_mae(config=self.config)
