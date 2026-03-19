"""
PyTorch MAE-CNN algorithm for depression detection.

Implements staged masked reconstruction with CNN backbone:
- M4c (dl_torch_mae_cnn_staged): CNN encoder-decoder pretrain, then CNN classification finetune

Phase 1: Train CNN encoder + decoder on masked modality reconstruction
Phase 2: Train CNN encoder + classification head on depression labels
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from algorithm.dl_torch_base import (
    DepressionDetectionAlgorithm_DL_torch,
    DepressionDetectionClassifier_DL_torch,
)
from utils.network_torch import (
    CNN1D_Backbone,
    CNN1D_Decoder,
    LabelHead,
    ModalityMaskEmbedding,
)
from utils.torch_training import (
    ModalityMaskingDataset,
    MODALITY_INDICES,
    TorchTrainer,
    _masked_recon_loss,
    build_cosine_warmup_scheduler,
    build_cosine_restarts_scheduler,
)

import pandas as pd


def _get_modality_dims(modality_indices):
    return {name: len(indices) for name, indices in modality_indices.items()}


class DepressionDetectionClassifier_DL_torch_mae_cnn(
    DepressionDetectionClassifier_DL_torch
):
    """PyTorch MAE-CNN classifier: staged reconstruction pretraining with CNN backbone."""

    def __init__(self, config):
        self.modality_indices = config.get("modality_indices", MODALITY_INDICES)
        self.modality_dims = _get_modality_dims(self.modality_indices)
        super().__init__(config)
        self.mask_embeddings = ModalityMaskEmbedding(self.modality_dims).to(self.device)

    def _build_model_parts(self):
        mp = self.model_params
        n_features = mp["input_shape"][1]
        embed_size = mp.get("embedding_size", 16)

        backbone = CNN1D_Backbone(
            n_features=n_features,
            conv_shapes=mp.get("conv_shapes", [8, 8, 8]),
            embedding_size=embed_size,
            dropout=mp.get("cnn_dropout", 0.25),
        )
        decoder = CNN1D_Decoder(
            n_features=n_features,
            conv_shapes=mp.get("conv_shapes", [8, 8, 8]),
            embedding_size=embed_size,
        )
        label_head = LabelHead(embed_size, num_classes=2)

        return {
            "backbone": backbone,
            "decoder": decoder,
            "label_head": label_head,
        }

    def _make_train_loader(self, training_data):
        mp = self.model_params
        dataset = ModalityMaskingDataset(
            X=training_data.train_X,
            y=training_data.train_y,
            mixup_alpha=self.config["data_loader"].get("mixup_alpha"),
            modality_indices=self.modality_indices,
            max_masked=mp.get("max_masked_modalities", 2),
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
        """Staged training: pretrain reconstruction, finetune classification."""
        train_loader = self._make_train_loader(training_data)
        eval_val = (training_data.val_X, training_data.val_y)
        eval_test = None
        if training_data.test_X is not None:
            eval_test = (training_data.test_X, training_data.test_y)

        backbone = self.model_parts["backbone"]
        decoder = self.model_parts["decoder"]
        label_head = self.model_parts["label_head"]

        tp = self.training_params
        epochs = tp.get("epochs", 200)
        steps_per_epoch = tp.get("steps_per_epoch", 100)
        pretrain_epochs = tp.get("pretrain_epochs", epochs)
        pretrain_patience = tp.get("pretrain_patience", 10)
        lr = tp.get("learning_rate", 0.001)
        grad_clip = tp.get("grad_clip", 0.0)
        verbose = tp.get("verbose", 0)

        # ── Phase 1: Pretrain reconstruction ──
        if verbose > 0:
            print("=== Phase 1: CNN masked reconstruction pretraining ===")

        pretrain_params = (list(backbone.parameters()) +
                          list(decoder.parameters()) +
                          list(self.mask_embeddings.parameters()))

        optimizer_type = tp.get("optimizer", "Adam")
        if optimizer_type == "Adam":
            optimizer = torch.optim.Adam(pretrain_params, lr=lr)
        else:
            weight_decay = tp.get("weight_decay", 1e-4)
            optimizer = torch.optim.AdamW(pretrain_params, lr=lr,
                                          weight_decay=weight_decay)

        scheduler_type = tp.get("scheduler", "cosine_restarts")
        total_steps = pretrain_epochs * steps_per_epoch
        if scheduler_type == "cosine_restarts":
            scheduler = build_cosine_restarts_scheduler(
                optimizer, tp.get("cos_annealing_step", 20),
                tp.get("cos_annealing_decay", 0.95),
                tp.get("cos_annealing_alpha", 0.01),
                tp.get("cos_annealing_t_mul", 2.0))
        else:
            warmup_steps = tp.get("warmup_steps", steps_per_epoch * 5)
            scheduler = build_cosine_warmup_scheduler(
                optimizer, warmup_steps, total_steps)

        best_recon_loss = float("inf")
        patience_counter = 0
        best_pretrain_state = None

        for epoch in range(pretrain_epochs):
            backbone.train()
            decoder.train()
            self.mask_embeddings.train()
            epoch_recon_loss = 0.0

            train_iter = iter(train_loader)
            for step in range(steps_per_epoch):
                try:
                    X_masked, y_batch, X_target, feat_mask = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    X_masked, y_batch, X_target, feat_mask = next(train_iter)

                X_masked = X_masked.to(self.device)
                X_target = X_target.to(self.device)
                feat_mask = feat_mask.to(self.device)

                # Forward: mask -> encode -> decode -> reconstruct
                X_input = self.mask_embeddings(X_masked, feat_mask)
                embedding = backbone(X_input)  # (batch, embed_size)
                X_recon = decoder(embedding)    # (batch, 28, n_features)

                # Reconstruction loss: MSE on masked positions only
                recon_loss = _cnn_masked_recon_loss(
                    X_recon, X_target, feat_mask, self.modality_indices)

                optimizer.zero_grad()
                recon_loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(pretrain_params, grad_clip)
                optimizer.step()
                scheduler.step()

                epoch_recon_loss += recon_loss.item()

            avg_recon = epoch_recon_loss / steps_per_epoch

            if verbose > 0:
                print(f"Pretrain epoch {epoch} -- recon_loss {avg_recon:.4f}")

            if avg_recon < best_recon_loss:
                best_recon_loss = avg_recon
                patience_counter = 0
                best_pretrain_state = {
                    "backbone": backbone.state_dict(),
                    "decoder": decoder.state_dict(),
                }
            else:
                patience_counter += 1
                if patience_counter >= pretrain_patience:
                    if verbose > 0:
                        print(f"Pretrain early stopping at epoch {epoch}")
                    break

        # Restore best pretrain backbone weights
        if best_pretrain_state is not None:
            backbone.load_state_dict(best_pretrain_state["backbone"])

        # ── Phase 2: Finetune classification ──
        if verbose > 0:
            print("=== Phase 2: Classification fine-tuning ===")

        # Create TorchTrainer for finetune phase (handles eval, early stopping, etc.)
        finetune_parts = {"backbone": backbone, "label_head": label_head}
        self.trainer = TorchTrainer(
            model_parts=finetune_parts,
            training_params=tp,
            device=self.device,
        )

        # Use the reorder training loop (label_head has same interface)
        # but with weight_of_reorder=0 we effectively just train classification
        # Actually, use train_erm-style: override _evaluate to use label_head
        # Simpler: just run the finetune loop directly

        finetune_params = (list(backbone.parameters()) +
                          list(label_head.parameters()))

        if optimizer_type == "Adam":
            ft_optimizer = torch.optim.Adam(finetune_params, lr=lr)
        else:
            weight_decay = tp.get("weight_decay", 1e-4)
            ft_optimizer = torch.optim.AdamW(finetune_params, lr=lr,
                                             weight_decay=weight_decay)

        ft_total_steps = epochs * steps_per_epoch
        if scheduler_type == "cosine_restarts":
            ft_scheduler = build_cosine_restarts_scheduler(
                ft_optimizer, tp.get("cos_annealing_step", 20),
                tp.get("cos_annealing_decay", 0.95),
                tp.get("cos_annealing_alpha", 0.01),
                tp.get("cos_annealing_t_mul", 2.0))
        else:
            warmup_steps = tp.get("warmup_steps", steps_per_epoch * 5)
            ft_scheduler = build_cosine_warmup_scheduler(
                ft_optimizer, warmup_steps, ft_total_steps)

        self.trainer.optimizer = ft_optimizer
        self.trainer.scheduler = ft_scheduler

        # Finetune loop
        results_record = []
        model_repo = {}
        best_val_metric = -1.0
        patience_counter = 0

        for epoch in range(epochs):
            backbone.train()
            label_head.train()
            epoch_cls_loss = 0.0

            train_iter = iter(train_loader)
            for step in range(steps_per_epoch):
                try:
                    X_masked, y_batch, X_target, feat_mask = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    X_masked, y_batch, X_target, feat_mask = next(train_iter)

                # No masking during finetune — use clean input
                X_batch = X_target.to(self.device)
                y_batch = y_batch.to(self.device)

                embedding = backbone(X_batch)
                cls_logits = label_head(embedding)
                cls_loss = F.cross_entropy(cls_logits, y_batch)

                ft_optimizer.zero_grad()
                cls_loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(finetune_params, grad_clip)
                ft_optimizer.step()
                ft_scheduler.step()

                epoch_cls_loss += cls_loss.item()

            avg_cls = epoch_cls_loss / steps_per_epoch

            # Save model state
            model_repo[epoch] = {
                "backbone": backbone.state_dict(),
                "label_head": label_head.state_dict(),
            }

            epoch_results = {"epoch": epoch, "logs_loss": avg_cls, "logs_cls_loss": avg_cls}
            if eval_val is not None:
                metrics_val = self.trainer._evaluate(eval_val[0], eval_val[1])
                epoch_results.update({f"{k}_val": v for k, v in metrics_val.items()})
            if eval_test is not None:
                metrics_test = self.trainer._evaluate(eval_test[0], eval_test[1])
                epoch_results.update({f"{k}_test": v for k, v in metrics_test.items()})

            results_record.append(epoch_results)

            if verbose > 0:
                print(f"Finetune epoch {epoch} -- cls_loss {avg_cls:.4f} "
                      f"Val balacc {epoch_results.get('balanced_acc_val', 0):.3f}", end="")
                if eval_test is not None:
                    print(f" | Test balacc {epoch_results.get('balanced_acc_test', 0):.3f}", end="")
                print()

            # Early stopping on validation balanced accuracy
            val_metric = epoch_results.get("balanced_acc_val", 0)
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= tp.get("patience", 10):
                    if verbose > 0:
                        print(f"Finetune early stopping at epoch {epoch}")
                    break

        # Store results for best epoch selection
        self.trainer.results_record = results_record
        self.trainer.model_repo_dict = model_repo

        return pd.DataFrame(results_record)


def _cnn_masked_recon_loss(X_recon, X_target, feat_mask, modality_indices):
    """Compute MSE loss on masked feature positions for CNN decoder output.

    Args:
        X_recon: (batch, 28, N_features) reconstructed output
        X_target: (batch, 28, N_features) original clean input
        feat_mask: (batch, N_features) boolean mask (True = masked)
        modality_indices: dict mapping modality name -> feature index list
    """
    total_loss = 0.0
    n_masked = 0

    for mod_name, col_indices in modality_indices.items():
        mod_masked = feat_mask[:, col_indices[0]]  # (batch,)
        if not mod_masked.any():
            continue

        for ci in col_indices:
            pred = X_recon[mod_masked, :, ci]    # (n_masked_samples, 28)
            target = X_target[mod_masked, :, ci]  # (n_masked_samples, 28)
            total_loss += F.mse_loss(pred, target, reduction="sum")
            n_masked += pred.numel()

    if n_masked == 0:
        return torch.tensor(0.0, device=X_recon.device, requires_grad=True)
    return total_loss / n_masked


class DepressionDetectionAlgorithm_DL_torch_mae_cnn(DepressionDetectionAlgorithm_DL_torch):
    """PyTorch MAE-CNN algorithm."""

    def __init__(self, config_dict=None, config_name="dl_torch_mae_cnn_staged"):
        super().__init__(config_dict=config_dict, config_name=config_name)

    def prep_model(self, data_train=None, criteria="balanced_acc"):
        self.config["model_params"].update({"input_shape": self.input_shape})
        return DepressionDetectionClassifier_DL_torch_mae_cnn(config=self.config)
