"""
PyTorch network architectures for behavioral time series depression detection.

Provides:
- BehavioralTransformerEncoder: Transformer encoder with learned day-of-week positional encoding
- CNN1D_Backbone: 1D-CNN matching the original GLOBEM TF architecture
- ClassificationHead: Linear(d_model -> 2) for depression classification
- ReorderHead: Dense(32, relu) -> Dense(num_classes+1, softmax) for reorder task
- MaskedReconstructionHead: Per-modality 2-layer MLP decoder for MAE
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions (0..seq_len-1)."""

    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class PreLNTransformerBlock(nn.Module):
    """Single Pre-LayerNorm Transformer encoder block."""

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop1(attn_out)

        # Pre-LN feedforward
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + self.drop2(ff_out)
        return x


class BehavioralTransformerEncoder(nn.Module):
    """Transformer encoder for behavioral time series.

    Input: (batch, 28, N_features)
    Output: (batch, d_model) after mean pooling

    If return_sequence=True, returns (batch, 28, d_model) before pooling
    (needed for masked reconstruction head).
    """

    def __init__(self, n_features, d_model=64, num_heads=8, ff_dim=128,
                 num_layers=4, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=64)

        self.blocks = nn.ModuleList([
            PreLNTransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, return_sequence=False):
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)       # (batch, seq_len, d_model)
        x = self.pos_enc(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)       # (batch, seq_len, d_model)

        if return_sequence:
            return x

        # Mean pooling over the sequence dimension
        return x.mean(dim=1)          # (batch, d_model)


class CNN1D_Backbone(nn.Module):
    """1D-CNN matching the original GLOBEM TF architecture.

    TF architecture (from network.py build_1dCNN):
      For each conv layer i in conv_shapes:
        Conv1D(filters, kernel=3, relu, padding=same, he_uniform, l2=2e-4)
        BatchNorm
        MaxPool1D(2) if i < 2
        Dropout(0.25)
      Flatten
      Dense(embedding_size, relu)

    Input:  (batch, 28, N_features)
    Output: (batch, embedding_size)
    """

    def __init__(self, n_features, conv_shapes=(8, 8, 8), embedding_size=16, dropout=0.25):
        super().__init__()
        layers = []
        in_channels = n_features
        for i, out_channels in enumerate(conv_shapes):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_channels))
            if i < 2:
                layers.append(nn.MaxPool1d(kernel_size=2))
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Compute flattened size: input seq_len=28, two MaxPool1d(2) -> 28 // 4 = 7
        self._flat_size = conv_shapes[-1] * 7
        self.fc = nn.Linear(self._flat_size, embedding_size)
        self.embedding_size = embedding_size

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, seq_len=28, n_features) — channels-last from data pipeline
        x = x.transpose(1, 2)        # (batch, n_features, seq_len) — Conv1d expects channels first
        x = self.conv_layers(x)       # (batch, out_channels, reduced_seq)
        x = x.reshape(x.size(0), -1) # flatten
        x = F.relu(self.fc(x))
        return x                      # (batch, embedding_size)


class ClassificationHead(nn.Module):
    """Binary classification head: Linear -> softmax (applied at loss time)."""

    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)  # raw logits; softmax applied in loss or predict


class ReorderHead(nn.Module):
    """Reorder prediction head matching TF architecture:
    Dense(32, relu) -> Dense(num_classes + 1, softmax)
    """

    def __init__(self, input_dim, num_reorder_classes=200):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, num_reorder_classes + 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # raw logits


class LabelHead(nn.Module):
    """Label prediction head matching TF reorder architecture:
    Dense(16, relu) -> Dense(2, softmax)
    """

    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # raw logits


class MaskedReconstructionHead(nn.Module):
    """Per-modality 2-layer MLP decoder for masked autoencoder reconstruction.

    Operates on pre-pooled sequence output (batch, seq_len, d_model).
    Reconstructs masked features for each modality independently.

    Args:
        d_model: transformer hidden dimension
        modality_dims: dict mapping modality name -> number of features in that modality
        hidden_dim: hidden layer size in each per-modality MLP
    """

    def __init__(self, d_model, modality_dims, hidden_dim=64):
        super().__init__()
        self.modality_heads = nn.ModuleDict()
        for name, n_feats in modality_dims.items():
            self.modality_heads[name] = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_feats),
            )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model) — sequence output from transformer

        Returns:
            dict mapping modality name -> (batch, seq_len, n_feats) reconstructions
        """
        return {name: head(x) for name, head in self.modality_heads.items()}


class ModalityMaskEmbedding(nn.Module):
    """Learned mask embeddings for modality masking.

    For each masked modality, replaces its feature columns with a learned
    embedding vector (broadcast across all 28 days).

    Args:
        modality_dims: dict mapping modality name -> number of features
    """

    def __init__(self, modality_dims):
        super().__init__()
        self.mask_params = nn.ParameterDict()
        for name, n_feats in modality_dims.items():
            self.mask_params[name] = nn.Parameter(torch.zeros(n_feats))

    def forward(self, X, feat_mask, modality_indices=None):
        """Replace masked feature columns with learned embeddings.

        Args:
            X: (batch, 28, N_features) — already zeroed at masked positions
            feat_mask: (batch, N_features) boolean mask
            modality_indices: dict mapping modality name -> feature indices
                (required if not stored; typically passed from dataset config)

        Returns:
            X with masked positions filled by learned embeddings
        """
        if modality_indices is None:
            from utils.torch_training import MODALITY_INDICES
            modality_indices = MODALITY_INDICES

        X_out = X.clone()
        for mod_name, param in self.mask_params.items():
            col_indices = modality_indices[mod_name]
            # Check which samples have this modality masked
            mod_masked = feat_mask[:, col_indices[0]]  # (batch,)
            if mod_masked.any():
                # Broadcast learned embedding: (n_feats,) -> (n_masked, 28, n_feats)
                X_out[mod_masked, :, col_indices[0]:col_indices[-1]+1] = 0.0
                # Handle non-contiguous indices
                for i, ci in enumerate(col_indices):
                    X_out[mod_masked, :, ci] = param[i]
        return X_out


# ─── Composite model builders ───────────────────────────────────────────

def build_erm_model(backbone, input_dim):
    """Build an ERM (classification-only) model.

    Args:
        backbone: nn.Module returning (batch, embed_dim) embeddings
        input_dim: embedding dimension from backbone

    Returns:
        (backbone, cls_head) tuple
    """
    cls_head = ClassificationHead(input_dim, num_classes=2)
    return backbone, cls_head


def build_reorder_model(backbone, input_dim, num_reorder_classes=200):
    """Build a reorder model with label + reorder heads.

    Returns:
        (backbone, label_head, reorder_head) tuple
    """
    label_head = LabelHead(input_dim, num_classes=2)
    reorder_head = ReorderHead(input_dim, num_reorder_classes)
    return backbone, label_head, reorder_head
