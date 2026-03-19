"""
PyTorch training infrastructure for GLOBEM depression detection.

Provides:
- MixupDataset: PyTorch Dataset with optional mixup augmentation
- ReorderDataset: Extends MixupDataset with temporal reordering
- TorchTrainer: Training loop with early stopping, checkpointing, evaluation
"""

import os
import math
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, confusion_matrix

from utils import path_definitions


# ─── Datasets ────────────────────────────────────────────────────────────

def apply_augmentation(X, modality_indices, aug_config):
    """Apply time-series augmentations to a single sample tensor.

    Applied after mixup, before any task-specific transforms (reorder, masking).

    Args:
        X: torch.Tensor of shape (28, F)
        modality_indices: dict mapping modality name -> list of feature column indices
        aug_config: dict with keys:
            mag_scale_range: (lo, hi) for per-modality magnitude scaling
            feat_dropout_p: probability of dropping each modality entirely
            window_slice_min: minimum sub-window length (days)

    Returns:
        X: augmented tensor (possibly shorter if window slicing applied, then zero-padded)
    """
    # Magnitude scaling: per-modality random scale factor
    if "mag_scale_range" in aug_config:
        lo, hi = aug_config["mag_scale_range"]
        for mod_name, col_indices in modality_indices.items():
            scale = np.random.uniform(lo, hi)
            for ci in col_indices:
                X[:, ci] = X[:, ci] * scale

    # Feature dropout: zero entire modalities with probability p
    if "feat_dropout_p" in aug_config:
        p = aug_config["feat_dropout_p"]
        for mod_name, col_indices in modality_indices.items():
            if np.random.random() < p:
                for ci in col_indices:
                    X[:, ci] = 0.0

    # Window slicing: random sub-window, zero-pad back to 28
    if "window_slice_min" in aug_config:
        min_len = aug_config["window_slice_min"]
        seq_len = X.shape[0]  # 28
        win_len = np.random.randint(min_len, seq_len + 1)
        start = np.random.randint(0, seq_len - win_len + 1)
        sliced = X[start:start + win_len].clone()
        X = torch.zeros_like(X)
        X[:win_len] = sliced

    return X


class MixupDataset(Dataset):
    """PyTorch Dataset wrapping DataRepo_np arrays with optional mixup.

    Replicates the mixup logic from MultiSourceDataGenerator.__mixup__().

    Args:
        X: np.ndarray of shape (N, 28, F)
        y: np.ndarray of shape (N, 2) one-hot labels
        mixup_alpha: Beta distribution alpha for mixup. None to disable.
        augmentation: dict of augmentation config (see apply_augmentation). None to disable.
        modality_indices: required if augmentation is not None
    """

    def __init__(self, X, y, mixup_alpha=None, augmentation=None, modality_indices=None):
        self.X = X.astype(np.float32)
        # Convert one-hot to class index if needed
        if y.ndim > 1 and y.shape[1] == 2:
            self.y_onehot = y.astype(np.float32)
            self.y = np.argmax(y, axis=1)
        else:
            self.y = y.astype(np.int64)
            self.y_onehot = np.eye(2, dtype=np.float32)[self.y]
        self.mixup_alpha = mixup_alpha
        self.augmentation = augmentation
        self.aug_modality_indices = modality_indices or MODALITY_INDICES

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y_onehot[idx]

        if self.mixup_alpha is not None and self.mixup_alpha > 0:
            # Sample a random partner and mixup
            idx2 = np.random.randint(0, len(self.X))
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            X = lam * X + (1 - lam) * self.X[idx2]
            y = lam * y + (1 - lam) * self.y_onehot[idx2]

        X = torch.from_numpy(X)

        if self.augmentation:
            X = apply_augmentation(X, self.aug_modality_indices, self.augmentation)

        return X, torch.from_numpy(y)


class ReorderDataset(MixupDataset):
    """Extends MixupDataset with temporal reordering augmentation.

    Replicates MultiSourceDataGeneratorReorder logic:
    - With probability `rate_of_reorder`, apply a random permutation to the time steps
    - The reorder label is a one-hot vector over (num_reorder_classes + 1) classes
      where class 0 = no reorder

    Args:
        X, y, mixup_alpha: same as MixupDataset
        num_reorder_classes: number of pre-determined reorder permutations
        rate_of_reorder: fraction of samples that get reordered
        permutation_list: np.ndarray of shape (num_reorder_classes, 28) — precomputed
            index permutations expanded from the 10-segment Hamming-maximized permutations
    """

    def __init__(self, X, y, mixup_alpha=None, num_reorder_classes=200,
                 rate_of_reorder=0.7, permutation_list=None,
                 augmentation=None, modality_indices=None):
        super().__init__(X, y, mixup_alpha, augmentation=augmentation,
                         modality_indices=modality_indices)
        self.num_reorder_classes = num_reorder_classes
        self.rate_of_reorder = rate_of_reorder
        self.noshuffle_idx = np.arange(28)

        if permutation_list is not None:
            self.permutation_list = permutation_list
        else:
            # Load and expand permutations (same logic as dl_reorder.py)
            perm_raw = np.load(
                os.path.join(path_definitions.TMP_PATH,
                             f"reorder/permutations_hamming_max_{num_reorder_classes}.npy"))
            self.permutation_list = self._expand_permutations(perm_raw)

    @staticmethod
    def _expand_permutations(perm_raw):
        """Expand 10-segment permutations to 28-day index permutations."""
        expanded = []
        for p in perm_raw:
            d = []
            for idx in p:
                if idx == 9:
                    d += [idx * 3]
                else:
                    d += [idx * 3, idx * 3 + 1, idx * 3 + 2]
            expanded.append(np.array(d))
        return np.array(expanded)

    def __getitem__(self, idx):
        X, y = super().__getitem__(idx)

        # Decide whether to reorder
        if np.random.random() < self.rate_of_reorder:
            reorder_class = np.random.randint(1, self.num_reorder_classes + 1)
            perm = self.permutation_list[reorder_class - 1]
            X = X[perm, :]  # permute time steps
        else:
            reorder_class = 0

        reorder_label = torch.zeros(self.num_reorder_classes + 1, dtype=torch.float32)
        reorder_label[reorder_class] = 1.0

        return X, y, reorder_label


class InferenceDataset(Dataset):
    """Simple dataset for inference — no augmentation."""

    def __init__(self, X, y=None, pids=None):
        self.X = torch.from_numpy(X.astype(np.float32))
        if y is not None:
            if y.ndim > 1 and y.shape[1] == 2:
                self.y = np.argmax(y, axis=1)
            else:
                self.y = y
        else:
            self.y = None
        self.pids = pids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


# Default modality-to-feature-index mapping for the Reorder feature set
# (feature_list_more_feat_types in dl_feat_prep.yaml, 54 features total).
MODALITY_INDICES = {
    "screen":    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16, 38,39,40,41,42,43,44,45],
    "sleep":     [17,18,19,20,21,22,23,24,25,26, 46,47,48],
    "steps":     [27,28,29,30, 49,50,51],
    "location":  [33,34,35,36,37],
    "bluetooth": [31, 52],
    "call":      [32, 53],
}


class ModalityMaskingDataset(MixupDataset):
    """Extends MixupDataset with modality masking for MAE training.

    At each sample, randomly masks K ~ Uniform{1, 2} modalities by zeroing
    their feature columns. Returns the original (unmasked) features as the
    reconstruction target and a boolean mask over feature columns.

    Note: The learned [MASK] embedding replacement is applied inside the model,
    not here. This dataset just decides *which* modalities to mask.

    Args:
        X, y, mixup_alpha: same as MixupDataset
        modality_indices: dict mapping modality name -> list of feature column indices
        max_masked: maximum number of modalities to mask per sample (K ~ U{1, max_masked})
    """

    def __init__(self, X, y, mixup_alpha=None, modality_indices=None,
                 max_masked=2, augmentation=None):
        super().__init__(X, y, mixup_alpha, augmentation=augmentation,
                         modality_indices=modality_indices)
        self.modality_indices = modality_indices or MODALITY_INDICES
        self.modality_names = list(self.modality_indices.keys())
        self.n_modalities = len(self.modality_names)
        self.n_features = X.shape[-1]
        self.max_masked = max_masked

    def __getitem__(self, idx):
        X, y = super().__getitem__(idx)

        # Keep unmasked copy as reconstruction target
        X_target = X.clone()

        # Choose K ~ Uniform{1, max_masked} modalities to mask
        K = np.random.randint(1, self.max_masked + 1)
        masked_mods = np.random.choice(self.n_modalities, size=K, replace=False)

        # Build feature-level mask (True = masked)
        feat_mask = torch.zeros(self.n_features, dtype=torch.bool)
        for mod_idx in masked_mods:
            mod_name = self.modality_names[mod_idx]
            for fi in self.modality_indices[mod_name]:
                feat_mask[fi] = True

        # Zero out masked features (model will replace with learned embeddings)
        X[:, feat_mask] = 0.0

        return X, y, X_target, feat_mask


class CombinedDataset(ReorderDataset):
    """Extends ReorderDataset with modality masking for combined training.

    Each sample gets both a reorder augmentation (with probability rate_of_reorder)
    AND a modality masking (always masks 1-max_masked modalities).

    Returns: (X_masked, y, reorder_label, X_target, feat_mask)
    """

    def __init__(self, X, y, mixup_alpha=None, num_reorder_classes=200,
                 rate_of_reorder=0.7, permutation_list=None,
                 modality_indices=None, max_masked=2, augmentation=None):
        super().__init__(X, y, mixup_alpha, num_reorder_classes,
                         rate_of_reorder, permutation_list,
                         augmentation=augmentation, modality_indices=modality_indices)
        self.modality_indices = modality_indices or MODALITY_INDICES
        self.modality_names = list(self.modality_indices.keys())
        self.n_modalities = len(self.modality_names)
        self.n_features = X.shape[-1]
        self.max_masked = max_masked

    def __getitem__(self, idx):
        # Get reorder-augmented sample: (X_reordered, y, reorder_label)
        X, y, reorder_label = super().__getitem__(idx)

        # Keep unmasked copy as reconstruction target
        X_target = X.clone()

        # Choose K ~ Uniform{1, max_masked} modalities to mask
        K = np.random.randint(1, self.max_masked + 1)
        masked_mods = np.random.choice(self.n_modalities, size=K, replace=False)

        # Build feature-level mask (True = masked)
        feat_mask = torch.zeros(self.n_features, dtype=torch.bool)
        for mod_idx in masked_mods:
            mod_name = self.modality_names[mod_idx]
            for fi in self.modality_indices[mod_name]:
                feat_mask[fi] = True

        # Zero out masked features
        X[:, feat_mask] = 0.0

        return X, y, reorder_label, X_target, feat_mask


# ─── LR Scheduling ──────────────────────────────────────────────────────

def build_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine annealing to 0."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_cosine_restarts_scheduler(optimizer, first_decay_steps, t_mul=2.0,
                                    m_mul=0.95, alpha=0.01):
    """Replicate TF CosineDecayRestarts schedule.

    Each cycle i has length first_decay_steps * t_mul^i steps.
    Peak LR decays by m_mul at each restart. Minimum LR factor is alpha.
    """
    def lr_lambda(step):
        completed = 0
        cycle = 0
        cycle_len = first_decay_steps
        while completed + cycle_len <= step:
            completed += cycle_len
            cycle += 1
            cycle_len = int(first_decay_steps * (t_mul ** cycle))
        progress = (step - completed) / max(1, cycle_len)
        decay = m_mul ** cycle
        return decay * (alpha + (1 - alpha) * 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── MAE Loss ────────────────────────────────────────────────────────────

def _masked_recon_loss(recon_outputs, X_target, feat_mask, modality_indices):
    """Compute MSE reconstruction loss on masked modality features only.

    Args:
        recon_outputs: dict mapping modality name -> (batch, 28, n_mod_feats) predictions
        X_target: (batch, 28, N_features) original unmasked features
        feat_mask: (batch, N_features) boolean mask (True = masked)
        modality_indices: dict mapping modality name -> list of feature column indices

    Returns:
        scalar MSE loss averaged over all masked features
    """
    total_loss = 0.0
    total_count = 0
    for mod_name, pred in recon_outputs.items():
        col_indices = modality_indices[mod_name]
        # Check which samples have this modality masked
        mod_masked = feat_mask[:, col_indices[0]]  # (batch,) — all cols in a modality share mask
        if not mod_masked.any():
            continue
        target = X_target[mod_masked][:, :, col_indices]  # (n_masked, 28, n_mod_feats)
        pred_masked = pred[mod_masked]                      # (n_masked, 28, n_mod_feats)
        total_loss += F.mse_loss(pred_masked, target, reduction="sum")
        total_count += target.numel()
    if total_count == 0:
        return torch.tensor(0.0, device=X_target.device, requires_grad=True)
    return total_loss / total_count


# ─── Evaluation ──────────────────────────────────────────────────────────

def evaluate_predictions(y_true, y_pred, y_proba):
    """Compute evaluation metrics matching the existing GLOBEM pipeline.

    Returns dict with: acc, balanced_acc, roc_auc, cfmtx
    """
    results = {}
    results["acc"] = accuracy_score(y_true, y_pred)
    results["balanced_acc"] = balanced_accuracy_score(y_true, y_pred)
    try:
        results["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
    except (ValueError, IndexError):
        results["roc_auc"] = float("nan")
    results["cfmtx"] = confusion_matrix(y_true, y_pred).tolist()
    return results


# ─── Trainer ─────────────────────────────────────────────────────────────

class TorchTrainer:
    """Manages the PyTorch training loop.

    Replaces Keras model.fit() + callbacks with:
    - AdamW optimizer with cosine annealing + linear warmup
    - Gradient clipping
    - Early stopping (patience on validation balanced accuracy)
    - Model checkpointing (in-memory state_dict snapshots per epoch)
    - Per-epoch evaluation on val/test sets
    - Best epoch selection (matching existing 'direct' strategy)

    Args:
        model_parts: dict with keys like 'backbone', 'cls_head', etc. — all nn.Modules
        training_params: dict from config YAML
        device: torch device
    """

    def __init__(self, model_parts, training_params, device=None, extra_params=None):
        self.model_parts = model_parts
        self.training_params = training_params
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = training_params.get("verbose", 0)

        # Move all model parts to device
        for name, module in self.model_parts.items():
            module.to(self.device)

        # Collect all parameters (including extra_params for e.g. mask embeddings)
        all_params = []
        for module in self.model_parts.values():
            all_params += list(module.parameters())
        if extra_params:
            all_params += list(extra_params)

        # Optimizer (configurable: Adam or AdamW)
        lr = training_params.get("learning_rate", 1e-3)
        optimizer_type = training_params.get("optimizer", "AdamW")
        if optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(all_params, lr=lr)
        else:
            weight_decay = training_params.get("weight_decay", 1e-4)
            self.optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

        # LR scheduler (configurable: cosine_warmup or cosine_restarts)
        epochs = training_params.get("epochs", 200)
        steps_per_epoch = training_params.get("steps_per_epoch", 100)
        scheduler_type = training_params.get("scheduler", "cosine_warmup")
        if scheduler_type == "cosine_restarts":
            self.scheduler = build_cosine_restarts_scheduler(
                self.optimizer,
                first_decay_steps=training_params.get("cos_annealing_step", 20),
                t_mul=training_params.get("cos_annealing_t_mul", 2.0),
                m_mul=training_params.get("cos_annealing_decay", 0.95),
                alpha=training_params.get("cos_annealing_alpha", 0.01),
            )
        else:
            total_steps = epochs * steps_per_epoch
            warmup_steps = training_params.get("warmup_steps", steps_per_epoch * 5)
            self.scheduler = build_cosine_warmup_scheduler(
                self.optimizer, warmup_steps, total_steps)

        self.grad_clip = training_params.get("grad_clip", 1.0)
        self.patience = training_params.get("patience", 10)

        # Checkpointing
        self.model_repo_dict = {}
        self.results_record = []

    def _get_all_state_dicts(self):
        return {name: deepcopy(module.state_dict()) for name, module in self.model_parts.items()}

    def _load_all_state_dicts(self, state_dicts):
        for name, module in self.model_parts.items():
            module.load_state_dict(state_dicts[name])

    def _set_train(self):
        for module in self.model_parts.values():
            module.train()

    def _set_eval(self):
        for module in self.model_parts.values():
            module.eval()

    def train_erm(self, train_loader, eval_data_val=None, eval_data_test=None):
        """Train an ERM (classification-only) model.

        Args:
            train_loader: DataLoader yielding (X, y_onehot) batches
            eval_data_val: (X_np, y_np) tuple for validation evaluation
            eval_data_test: (X_np, y_np) tuple for test evaluation, or None

        Returns:
            pd.DataFrame of per-epoch results
        """
        backbone = self.model_parts["backbone"]
        cls_head = self.model_parts["cls_head"]
        epochs = self.training_params.get("epochs", 200)
        steps_per_epoch = self.training_params.get("steps_per_epoch", 100)

        for epoch in range(epochs):
            self._set_train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            train_iter = iter(train_loader)
            for step in range(steps_per_epoch):
                try:
                    X_batch, y_batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    X_batch, y_batch = next(train_iter)

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                embeddings = backbone(X_batch)
                logits = cls_head(embeddings)
                loss = F.cross_entropy(logits, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        list(backbone.parameters()) + list(cls_head.parameters()),
                        self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                preds = logits.argmax(dim=1)
                labels = y_batch.argmax(dim=1) if y_batch.dim() > 1 else y_batch
                epoch_correct += (preds == labels).sum().item()
                epoch_total += len(labels)

            avg_loss = epoch_loss / steps_per_epoch
            avg_acc = epoch_correct / max(epoch_total, 1)

            # Checkpoint
            self.model_repo_dict[epoch] = self._get_all_state_dicts()

            # Evaluate
            epoch_results = {"epoch": epoch, "logs_loss": avg_loss, "logs_acc": avg_acc}
            if eval_data_val is not None:
                metrics_val = self._evaluate(eval_data_val[0], eval_data_val[1])
                epoch_results.update({f"{k}_val": v for k, v in metrics_val.items()})
            if eval_data_test is not None:
                metrics_test = self._evaluate(eval_data_test[0], eval_data_test[1])
                epoch_results.update({f"{k}_test": v for k, v in metrics_test.items()})

            self.results_record.append(epoch_results)

            if self.verbose > 0 and eval_data_val is not None:
                print(f"Epoch {epoch} -- loss {avg_loss:.4f} "
                      f"Val: balacc {epoch_results.get('balanced_acc_val', 0):.3f} "
                      f"auc {epoch_results.get('roc_auc_val', 0):.3f}", end="")
                if eval_data_test is not None:
                    print(f" | Test: balacc {epoch_results.get('balanced_acc_test', 0):.3f} "
                          f"auc {epoch_results.get('roc_auc_test', 0):.3f}", end="")
                print()

            # Early stopping check
            if self._check_early_stop(epoch):
                if self.verbose > 0:
                    print(f"Early stopping at epoch {epoch}")
                break

        return pd.DataFrame(self.results_record)

    def train_reorder(self, train_loader, weight_of_reorder,
                      eval_data_val=None, eval_data_test=None):
        """Train a reorder multi-task model.

        Args:
            train_loader: DataLoader yielding (X, y_onehot, reorder_label) batches
            weight_of_reorder: loss weight for the reorder task
            eval_data_val, eval_data_test: same as train_erm

        Returns:
            pd.DataFrame of per-epoch results
        """
        backbone = self.model_parts["backbone"]
        label_head = self.model_parts["label_head"]
        reorder_head = self.model_parts["reorder_head"]
        epochs = self.training_params.get("epochs", 200)
        steps_per_epoch = self.training_params.get("steps_per_epoch", 100)

        for epoch in range(epochs):
            self._set_train()
            epoch_loss = 0.0
            epoch_cls_loss = 0.0
            epoch_reorder_loss = 0.0

            train_iter = iter(train_loader)
            for step in range(steps_per_epoch):
                try:
                    X_batch, y_batch, reorder_batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    X_batch, y_batch, reorder_batch = next(train_iter)

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                reorder_batch = reorder_batch.to(self.device)

                embeddings = backbone(X_batch)
                cls_logits = label_head(embeddings)
                reorder_logits = reorder_head(embeddings)

                cls_loss = F.cross_entropy(cls_logits, y_batch)
                reorder_loss = F.cross_entropy(reorder_logits, reorder_batch)
                loss = cls_loss + weight_of_reorder * reorder_loss

                self.optimizer.zero_grad()
                loss.backward()
                all_params = (list(backbone.parameters()) +
                              list(label_head.parameters()) +
                              list(reorder_head.parameters()))
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(all_params, self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                epoch_cls_loss += cls_loss.item()
                epoch_reorder_loss += reorder_loss.item()

            avg_loss = epoch_loss / steps_per_epoch

            # Checkpoint
            self.model_repo_dict[epoch] = self._get_all_state_dicts()

            # Evaluate
            epoch_results = {
                "epoch": epoch,
                "logs_loss": avg_loss,
                "logs_cls_loss": epoch_cls_loss / steps_per_epoch,
                "logs_reorder_loss": epoch_reorder_loss / steps_per_epoch,
            }
            if eval_data_val is not None:
                metrics_val = self._evaluate(eval_data_val[0], eval_data_val[1])
                epoch_results.update({f"{k}_val": v for k, v in metrics_val.items()})
            if eval_data_test is not None:
                metrics_test = self._evaluate(eval_data_test[0], eval_data_test[1])
                epoch_results.update({f"{k}_test": v for k, v in metrics_test.items()})

            self.results_record.append(epoch_results)

            if self.verbose > 0 and eval_data_val is not None:
                print(f"Epoch {epoch} -- loss {avg_loss:.4f} "
                      f"cls_loss {epoch_results['logs_cls_loss']:.4f} "
                      f"reorder_loss {epoch_results['logs_reorder_loss']:.4f} "
                      f"Val balacc {epoch_results.get('balanced_acc_val', 0):.3f}", end="")
                if eval_data_test is not None:
                    print(f" | Test balacc {epoch_results.get('balanced_acc_test', 0):.3f}", end="")
                print()

            if self._check_early_stop(epoch):
                if self.verbose > 0:
                    print(f"Early stopping at epoch {epoch}")
                break

        return pd.DataFrame(self.results_record)

    def train_mae_simultaneous(self, train_loader, beta, mask_embeddings,
                               modality_indices, eval_data_val=None,
                               eval_data_test=None):
        """Train MAE model with simultaneous classification + reconstruction.

        Loss: L = Lc + beta * Lmask

        Args:
            train_loader: DataLoader yielding (X_masked, y, X_target, feat_mask)
            beta: weight for reconstruction loss
            mask_embeddings: nn.Module with learned mask embedding parameters
            modality_indices: dict mapping modality name -> feature index list
            eval_data_val, eval_data_test: same as train_erm
        """
        backbone = self.model_parts["backbone"]
        cls_head = self.model_parts["cls_head"]
        recon_head = self.model_parts["recon_head"]
        epochs = self.training_params.get("epochs", 200)
        steps_per_epoch = self.training_params.get("steps_per_epoch", 100)

        for epoch in range(epochs):
            self._set_train()
            mask_embeddings.train()
            epoch_loss = 0.0
            epoch_cls_loss = 0.0
            epoch_recon_loss = 0.0

            train_iter = iter(train_loader)
            for step in range(steps_per_epoch):
                try:
                    X_masked, y_batch, X_target, feat_mask = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    X_masked, y_batch, X_target, feat_mask = next(train_iter)

                X_masked = X_masked.to(self.device)
                y_batch = y_batch.to(self.device)
                X_target = X_target.to(self.device)
                feat_mask = feat_mask.to(self.device)

                # Classification forward on CLEAN input (matches eval distribution)
                cls_embeddings = backbone(X_target)
                cls_logits = cls_head(cls_embeddings)
                cls_loss = F.cross_entropy(cls_logits, y_batch)

                # Reconstruction forward on MASKED input
                X_input = mask_embeddings(X_masked, feat_mask)
                seq_out = backbone(X_input, return_sequence=True)
                recon_outputs = recon_head(seq_out)
                recon_loss = _masked_recon_loss(
                    recon_outputs, X_target, feat_mask, modality_indices)

                loss = cls_loss + beta * recon_loss

                self.optimizer.zero_grad()
                loss.backward()
                all_params = (list(backbone.parameters()) +
                              list(cls_head.parameters()) +
                              list(recon_head.parameters()) +
                              list(mask_embeddings.parameters()))
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(all_params, self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                epoch_cls_loss += cls_loss.item()
                epoch_recon_loss += recon_loss.item()

            avg_loss = epoch_loss / steps_per_epoch

            self.model_repo_dict[epoch] = self._get_all_state_dicts()

            epoch_results = {
                "epoch": epoch,
                "logs_loss": avg_loss,
                "logs_cls_loss": epoch_cls_loss / steps_per_epoch,
                "logs_recon_loss": epoch_recon_loss / steps_per_epoch,
            }
            if eval_data_val is not None:
                metrics_val = self._evaluate(eval_data_val[0], eval_data_val[1])
                epoch_results.update({f"{k}_val": v for k, v in metrics_val.items()})
            if eval_data_test is not None:
                metrics_test = self._evaluate(eval_data_test[0], eval_data_test[1])
                epoch_results.update({f"{k}_test": v for k, v in metrics_test.items()})

            self.results_record.append(epoch_results)

            if self.verbose > 0:
                print(f"Epoch {epoch} -- loss {avg_loss:.4f} "
                      f"cls {epoch_results['logs_cls_loss']:.4f} "
                      f"recon {epoch_results['logs_recon_loss']:.4f} "
                      f"Val balacc {epoch_results.get('balanced_acc_val', 0):.3f}", end="")
                if eval_data_test is not None:
                    print(f" | Test balacc {epoch_results.get('balanced_acc_test', 0):.3f}", end="")
                print()

            if self._check_early_stop(epoch):
                if self.verbose > 0:
                    print(f"Early stopping at epoch {epoch}")
                break

        return pd.DataFrame(self.results_record)

    def train_mae_staged(self, train_loader, mask_embeddings, modality_indices,
                         eval_data_val=None, eval_data_test=None,
                         pretrain_epochs=None, pretrain_patience=10):
        """Train MAE model with staged pretrain (reconstruction) then finetune (classification).

        Phase 1: Train backbone + recon_head on Lmask only
        Phase 2: Train backbone + cls_head on Lc only (recon_head discarded)

        Args:
            train_loader: DataLoader yielding (X_masked, y, X_target, feat_mask)
            mask_embeddings: nn.Module with learned mask embedding parameters
            modality_indices: dict mapping modality name -> feature index list
            pretrain_epochs: max epochs for pretraining phase (defaults to training_params epochs)
            pretrain_patience: early stopping patience for pretraining (on recon loss)
        """
        backbone = self.model_parts["backbone"]
        cls_head = self.model_parts["cls_head"]
        recon_head = self.model_parts["recon_head"]
        epochs = self.training_params.get("epochs", 200)
        steps_per_epoch = self.training_params.get("steps_per_epoch", 100)
        if pretrain_epochs is None:
            pretrain_epochs = epochs

        # ── Phase 1: Pretrain on reconstruction ──
        if self.verbose > 0:
            print("=== Phase 1: Masked reconstruction pretraining ===")

        pretrain_records = []
        best_recon_loss = float("inf")
        patience_counter = 0

        for epoch in range(pretrain_epochs):
            self._set_train()
            mask_embeddings.train()
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

                X_input = mask_embeddings(X_masked, feat_mask)
                seq_out = backbone(X_input, return_sequence=True)
                recon_outputs = recon_head(seq_out)
                recon_loss = _masked_recon_loss(
                    recon_outputs, X_target, feat_mask, modality_indices)

                self.optimizer.zero_grad()
                recon_loss.backward()
                pretrain_params = (list(backbone.parameters()) +
                                   list(recon_head.parameters()) +
                                   list(mask_embeddings.parameters()))
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(pretrain_params, self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                epoch_recon_loss += recon_loss.item()

            avg_recon = epoch_recon_loss / steps_per_epoch
            pretrain_records.append({"epoch": epoch, "phase": "pretrain",
                                     "logs_recon_loss": avg_recon})

            if self.verbose > 0:
                print(f"Pretrain epoch {epoch} -- recon_loss {avg_recon:.4f}")

            # Early stopping on reconstruction loss
            if avg_recon < best_recon_loss:
                best_recon_loss = avg_recon
                patience_counter = 0
                best_pretrain_state = self._get_all_state_dicts()
            else:
                patience_counter += 1
                if patience_counter >= pretrain_patience:
                    if self.verbose > 0:
                        print(f"Pretrain early stopping at epoch {epoch}")
                    break

        # Restore best pretrain weights
        if best_pretrain_state is not None:
            self._load_all_state_dicts(best_pretrain_state)

        # ── Phase 2: Finetune on classification ──
        if self.verbose > 0:
            print("=== Phase 2: Classification fine-tuning ===")

        # Reset optimizer and scheduler for finetune phase
        all_params = []
        for module in self.model_parts.values():
            all_params += list(module.parameters())
        lr = self.training_params.get("learning_rate", 1e-4)
        optimizer_type = self.training_params.get("optimizer", "AdamW")
        if optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(all_params, lr=lr)
        else:
            weight_decay = self.training_params.get("weight_decay", 1e-4)
            self.optimizer = torch.optim.AdamW(all_params, lr=lr,
                                               weight_decay=weight_decay)
        total_steps = epochs * steps_per_epoch
        warmup_steps = self.training_params.get("warmup_steps",
                                                steps_per_epoch * 5)
        self.scheduler = build_cosine_warmup_scheduler(
            self.optimizer, warmup_steps, total_steps)

        # Reset early stopping state
        self.results_record = []
        self.model_repo_dict = {}

        for epoch in range(epochs):
            self._set_train()
            epoch_cls_loss = 0.0

            train_iter = iter(train_loader)
            for step in range(steps_per_epoch):
                try:
                    X_masked, y_batch, X_target, feat_mask = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    X_masked, y_batch, X_target, feat_mask = next(train_iter)

                # No masking during finetune — use original features
                X_batch = X_target.to(self.device)
                y_batch = y_batch.to(self.device)

                embeddings = backbone(X_batch)
                cls_logits = cls_head(embeddings)
                cls_loss = F.cross_entropy(cls_logits, y_batch)

                self.optimizer.zero_grad()
                cls_loss.backward()
                finetune_params = (list(backbone.parameters()) +
                                   list(cls_head.parameters()))
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(finetune_params, self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                epoch_cls_loss += cls_loss.item()

            avg_cls = epoch_cls_loss / steps_per_epoch

            self.model_repo_dict[epoch] = self._get_all_state_dicts()

            epoch_results = {
                "epoch": epoch,
                "logs_loss": avg_cls,
                "logs_cls_loss": avg_cls,
            }
            if eval_data_val is not None:
                metrics_val = self._evaluate(eval_data_val[0], eval_data_val[1])
                epoch_results.update({f"{k}_val": v for k, v in metrics_val.items()})
            if eval_data_test is not None:
                metrics_test = self._evaluate(eval_data_test[0], eval_data_test[1])
                epoch_results.update({f"{k}_test": v for k, v in metrics_test.items()})

            self.results_record.append(epoch_results)

            if self.verbose > 0:
                print(f"Finetune epoch {epoch} -- cls_loss {avg_cls:.4f} "
                      f"Val balacc {epoch_results.get('balanced_acc_val', 0):.3f}", end="")
                if eval_data_test is not None:
                    print(f" | Test balacc {epoch_results.get('balanced_acc_test', 0):.3f}", end="")
                print()

            if self._check_early_stop(epoch):
                if self.verbose > 0:
                    print(f"Finetune early stopping at epoch {epoch}")
                break

        return pd.DataFrame(self.results_record)

    def train_combined(self, train_loader, weight_of_reorder, beta,
                       mask_embeddings, modality_indices,
                       eval_data_val=None, eval_data_test=None):
        """Train combined three-head model: classification + reorder + reconstruction.

        Loss: L = Lc + alpha * Lreorder + beta * Lrecon

        Classification is computed on clean (unmasked) input to match eval distribution.
        Reorder and reconstruction are computed on the masked/reordered input.

        Args:
            train_loader: DataLoader yielding (X_masked, y, reorder_label, X_target, feat_mask)
            weight_of_reorder: loss weight for the reorder task (alpha)
            beta: loss weight for the reconstruction task
            mask_embeddings: ModalityMaskEmbedding module
            modality_indices: dict mapping modality name -> list of feature column indices
            eval_data_val, eval_data_test: (X_np, y_np) tuples

        Returns:
            pd.DataFrame of per-epoch results
        """
        backbone = self.model_parts["backbone"]
        label_head = self.model_parts["label_head"]
        reorder_head = self.model_parts["reorder_head"]
        recon_head = self.model_parts["recon_head"]
        epochs = self.training_params.get("epochs", 200)
        steps_per_epoch = self.training_params.get("steps_per_epoch", 100)

        for epoch in range(epochs):
            self._set_train()
            mask_embeddings.train()
            epoch_loss = 0.0
            epoch_cls_loss = 0.0
            epoch_reorder_loss = 0.0
            epoch_recon_loss = 0.0

            train_iter = iter(train_loader)
            for step in range(steps_per_epoch):
                try:
                    X_masked, y_batch, reorder_batch, X_target, feat_mask = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    X_masked, y_batch, reorder_batch, X_target, feat_mask = next(train_iter)

                X_masked = X_masked.to(self.device)
                y_batch = y_batch.to(self.device)
                reorder_batch = reorder_batch.to(self.device)
                X_target = X_target.to(self.device)
                feat_mask = feat_mask.to(self.device)

                # Classification on clean input (matches eval distribution)
                cls_embeddings = backbone(X_target)
                cls_logits = label_head(cls_embeddings)
                cls_loss = F.cross_entropy(cls_logits, y_batch)

                # Reorder on clean input (reorder is temporal, independent of masking)
                reorder_logits = reorder_head(cls_embeddings)
                reorder_loss = F.cross_entropy(reorder_logits, reorder_batch)

                # Reconstruction on masked input
                X_input = mask_embeddings(X_masked, feat_mask)
                seq_out = backbone(X_input, return_sequence=True)
                recon_outputs = recon_head(seq_out)
                recon_loss = _masked_recon_loss(
                    recon_outputs, X_target, feat_mask, modality_indices)

                loss = cls_loss + weight_of_reorder * reorder_loss + beta * recon_loss

                self.optimizer.zero_grad()
                loss.backward()
                all_params = (list(backbone.parameters()) +
                              list(label_head.parameters()) +
                              list(reorder_head.parameters()) +
                              list(recon_head.parameters()) +
                              list(mask_embeddings.parameters()))
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(all_params, self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                epoch_cls_loss += cls_loss.item()
                epoch_reorder_loss += reorder_loss.item()
                epoch_recon_loss += recon_loss.item()

            avg_loss = epoch_loss / steps_per_epoch

            self.model_repo_dict[epoch] = self._get_all_state_dicts()

            epoch_results = {
                "epoch": epoch,
                "logs_loss": avg_loss,
                "logs_cls_loss": epoch_cls_loss / steps_per_epoch,
                "logs_reorder_loss": epoch_reorder_loss / steps_per_epoch,
                "logs_recon_loss": epoch_recon_loss / steps_per_epoch,
            }
            if eval_data_val is not None:
                metrics_val = self._evaluate(eval_data_val[0], eval_data_val[1])
                epoch_results.update({f"{k}_val": v for k, v in metrics_val.items()})
            if eval_data_test is not None:
                metrics_test = self._evaluate(eval_data_test[0], eval_data_test[1])
                epoch_results.update({f"{k}_test": v for k, v in metrics_test.items()})

            self.results_record.append(epoch_results)

            if self.verbose > 0:
                print(f"Epoch {epoch} -- loss {avg_loss:.4f} "
                      f"cls {epoch_results['logs_cls_loss']:.4f} "
                      f"reorder {epoch_results['logs_reorder_loss']:.4f} "
                      f"recon {epoch_results['logs_recon_loss']:.4f} "
                      f"Val balacc {epoch_results.get('balanced_acc_val', 0):.3f}", end="")
                if eval_data_test is not None:
                    print(f" | Test balacc {epoch_results.get('balanced_acc_test', 0):.3f}", end="")
                print()

            if self._check_early_stop(epoch):
                if self.verbose > 0:
                    print(f"Early stopping at epoch {epoch}")
                break

        return pd.DataFrame(self.results_record)

    def _evaluate(self, X_np, y_np):
        """Run inference and compute metrics.

        Args:
            X_np: np.ndarray (N, 28, F)
            y_np: np.ndarray (N,) class labels or (N, 2) one-hot

        Returns:
            dict of metrics
        """
        self._set_eval()
        if y_np.ndim > 1 and y_np.shape[1] == 2:
            y_true = np.argmax(y_np, axis=1)
        else:
            y_true = y_np

        y_proba = self.predict_proba_np(X_np)
        y_pred = np.argmax(y_proba, axis=1)
        return evaluate_predictions(y_true, y_pred, y_proba)

    @torch.no_grad()
    def predict_proba_np(self, X_np):
        """Run inference on numpy array, return (N, 2) probabilities."""
        self._set_eval()
        backbone = self.model_parts["backbone"]

        # Use cls_head or label_head depending on model type
        if "cls_head" in self.model_parts:
            head = self.model_parts["cls_head"]
        elif "label_head" in self.model_parts:
            head = self.model_parts["label_head"]
        else:
            raise ValueError("No classification head found in model_parts")

        X_tensor = torch.from_numpy(X_np.astype(np.float32)).to(self.device)
        # Process in chunks to avoid OOM
        batch_size = 512
        all_proba = []
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size]
            embeddings = backbone(batch)
            logits = head(embeddings)
            proba = F.softmax(logits, dim=1)
            all_proba.append(proba.cpu().numpy())
        return np.concatenate(all_proba, axis=0)

    def _check_early_stop(self, current_epoch):
        """Check if training should stop based on validation balanced accuracy."""
        if self.patience <= 0 or current_epoch < self.patience:
            return False
        df = pd.DataFrame(self.results_record)
        if "balanced_acc_val" not in df.columns:
            return False
        recent = df["balanced_acc_val"].iloc[-self.patience:]
        best_overall = df["balanced_acc_val"].max()
        # Stop if no improvement in the last `patience` epochs
        return recent.max() < best_overall

    def find_best_epoch(self, df_results_record, strategy="direct"):
        """Find best epoch matching GLOBEM's existing logic.

        'direct': pick epoch with best validation balanced accuracy (or min loss as fallback)
        'on_test': same as existing find_best_epoch_on_test
        """
        df = df_results_record.reset_index(drop=True)
        if strategy == "direct":
            if "balanced_acc_val" in df.columns and not pd.isna(df["balanced_acc_val"]).any():
                return int(df.loc[df["balanced_acc_val"].idxmax(), "epoch"])
            else:
                return int(df.loc[df["logs_loss"].idxmin(), "epoch"])
        elif strategy == "on_test":
            # Replicate find_best_epoch_on_test logic
            metrics_col = "balanced_acc_val" if ("balanced_acc_val" in df.columns and
                          not pd.isna(df["balanced_acc_val"]).any()) else "logs_loss"
            flag_flip = -1 if "loss" in metrics_col else 1
            test_col = "balanced_acc_test" if "balanced_acc_test" in df.columns else "balanced_acc_val"

            best_train = flag_flip * df.iloc[0][metrics_col]
            current_test = df.iloc[0][test_col]
            best_test_list = []
            for _, row in df.iterrows():
                if flag_flip * row[metrics_col] > best_train:
                    best_train = flag_flip * row[metrics_col]
                    current_test = row[test_col]
                best_test_list.append(current_test)
            df["_best_test"] = best_test_list
            return int(df.loc[df["_best_test"].idxmax(), "epoch"])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def save_epoch_results(self, df_results_record, save_folder, config_name=""):
        """Save results and model checkpoints to disk."""
        save_obj = {
            "df_results_record": df_results_record,
            "model_repo_dict": self.model_repo_dict,
        }
        file_path = os.path.join(save_folder, "results_saving.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(save_obj, f)
