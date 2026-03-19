"""
PyTorch network architectures for behavioral time series depression detection.

Provides:
- BehavioralTransformerEncoder: Transformer encoder with learned day-of-week positional encoding
- CNN1D_Backbone: 1D-CNN matching the original GLOBEM TF architecture
- LabelHead: Dense(d_model -> 16, relu) -> Dense(16 -> 2) for depression classification
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


class CNN1D_Decoder(nn.Module):
    """1D-CNN decoder that mirrors CNN1D_Backbone for reconstruction.

    Takes the CNN embedding and reconstructs the original (28, N_features) input
    using transposed convolutions.

    Input:  (batch, embedding_size)
    Output: (batch, 28, N_features)
    """

    def __init__(self, n_features, conv_shapes=(8, 8, 8), embedding_size=16):
        super().__init__()
        # Reverse of encoder: embedding -> unflatten -> deconv layers
        # Encoder: 28 -> MaxPool(2) -> 14 -> MaxPool(2) -> 7 -> conv -> 7
        # Decoder: 7 -> upsample -> 14 -> upsample -> 28
        self._seq_len_reduced = 7
        self._last_channels = conv_shapes[-1]

        self.fc = nn.Linear(embedding_size, self._last_channels * self._seq_len_reduced)

        layers = []
        rev_shapes = list(reversed(conv_shapes))
        for i, in_channels in enumerate(rev_shapes):
            if i < len(rev_shapes) - 1:
                out_channels = rev_shapes[i + 1]
            else:
                out_channels = n_features

            if i < 2:  # mirror the 2 MaxPool layers
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, padding=1))
            if i < len(rev_shapes) - 1:
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(out_channels))

        self.deconv_layers = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, embedding_size)
        x = F.relu(self.fc(x))
        x = x.view(x.size(0), self._last_channels, self._seq_len_reduced)  # (batch, C, 7)
        x = self.deconv_layers(x)  # (batch, n_features, 28)
        x = x.transpose(1, 2)  # (batch, 28, n_features)
        return x


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


class ModalityCNNEncoder(nn.Module):
    """Small 1D-CNN encoder for a single modality's temporal features.

    Input:  (batch, 28, n_feats) — one modality's features over 28 days
    Output: (batch, d_token) — modality embedding
    """

    def __init__(self, n_feats, conv_channels=8, d_token=32, dropout=0.25):
        super().__init__()
        layers = []
        in_ch = n_feats
        for i in range(3):
            layers.append(nn.Conv1d(in_ch, conv_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(conv_channels))
            if i < 2:
                layers.append(nn.MaxPool1d(kernel_size=2))
            layers.append(nn.Dropout(dropout))
            in_ch = conv_channels

        self.conv_layers = nn.Sequential(*layers)
        # After 2 MaxPool(2): 28 -> 14 -> 7
        self._flat_size = conv_channels * 7
        self.fc = nn.Linear(self._flat_size, d_token)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, 28, n_feats)
        x = x.transpose(1, 2)        # (batch, n_feats, 28)
        x = self.conv_layers(x)       # (batch, conv_channels, 7)
        x = x.reshape(x.size(0), -1)  # (batch, flat_size)
        x = F.relu(self.fc(x))        # (batch, d_token)
        return x


class ModalityTokenBackbone(nn.Module):
    """Modality-Token Hybrid backbone: per-modality CNN encoders + cross-attention.

    Splits input into per-modality sub-sequences, encodes each with an independent
    small CNN, then applies transformer cross-attention over the 6 modality tokens.

    Input:  (batch, 28, 54)
    Output: (batch, d_token) after mean pooling over modality tokens
    """

    def __init__(self, modality_indices, conv_channels=8, d_token=32,
                 num_cross_layers=2, num_heads=4, ff_dim=64, dropout=0.1):
        super().__init__()
        self.modality_indices = modality_indices
        self.d_token = d_token

        # Per-modality CNN encoders
        self.modality_encoders = nn.ModuleDict()
        for name, indices in modality_indices.items():
            n_feats = len(indices)
            self.modality_encoders[name] = ModalityCNNEncoder(
                n_feats=n_feats,
                conv_channels=conv_channels,
                d_token=d_token,
                dropout=0.25,  # CNN dropout matches M0
            )

        # Cross-modality transformer
        self.cross_blocks = nn.ModuleList([
            PreLNTransformerBlock(d_token, num_heads, ff_dim, dropout)
            for _ in range(num_cross_layers)
        ])
        self.final_norm = nn.LayerNorm(d_token, eps=1e-6)

    def forward(self, x, mask_modalities=None, return_tokens=False):
        """Forward pass through modality-token backbone.

        Args:
            x: (batch, 28, 54) input features
            mask_modalities: optional set of modality names to skip encoding.
                If provided, only visible (unmasked) modalities are encoded.
            return_tokens: if True, return (batch, n_tokens, d_token) before pooling

        Returns:
            (batch, d_token) pooled embedding, or (batch, n_tokens, d_token) if return_tokens
        """
        # x: (batch, 28, 54)
        # Encode each modality independently
        tokens = []
        for name, encoder in self.modality_encoders.items():
            if mask_modalities and name in mask_modalities:
                continue
            indices = self.modality_indices[name]
            x_mod = x[:, :, indices]                     # (batch, 28, n_feats)
            tok = encoder(x_mod)                         # (batch, d_token)
            tokens.append(tok)

        # Stack into (batch, n_visible, d_token)
        x_tokens = torch.stack(tokens, dim=1)

        # Cross-modality attention
        for block in self.cross_blocks:
            x_tokens = block(x_tokens)

        x_tokens = self.final_norm(x_tokens)

        if return_tokens:
            return x_tokens

        # Mean pool over modality tokens
        return x_tokens.mean(dim=1)  # (batch, d_token)


class ModalityCrossAttentionDecoder(nn.Module):
    """Cross-attention decoder for modality-token MAE.

    For each masked modality, uses a learned query token that cross-attends
    to the visible modality tokens, then decodes back to (28, n_feats) via
    a per-modality CNN decoder.

    This is the architecturally principled MAE: masking = dropping tokens,
    reconstruction = cross-attending from masked queries to visible tokens.
    """

    def __init__(self, modality_indices, d_token=32, conv_channels=8,
                 num_decoder_layers=1, num_heads=4, ff_dim=64, dropout=0.1):
        super().__init__()
        self.modality_indices = modality_indices
        self.d_token = d_token

        # Learned query tokens — one per modality
        self.query_tokens = nn.ParameterDict()
        for name in modality_indices:
            self.query_tokens[name] = nn.Parameter(torch.randn(d_token) * 0.02)

        # Cross-attention layers (query attends to visible tokens)
        self.cross_attn_layers = nn.ModuleList()
        self.cross_norms_q = nn.ModuleList()
        self.cross_norms_kv = nn.ModuleList()
        self.cross_ff = nn.ModuleList()
        self.cross_ff_norms = nn.ModuleList()
        for _ in range(num_decoder_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(d_token, num_heads, dropout=dropout, batch_first=True))
            self.cross_norms_q.append(nn.LayerNorm(d_token, eps=1e-6))
            self.cross_norms_kv.append(nn.LayerNorm(d_token, eps=1e-6))
            self.cross_ff.append(nn.Sequential(
                nn.Linear(d_token, ff_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(ff_dim, d_token)))
            self.cross_ff_norms.append(nn.LayerNorm(d_token, eps=1e-6))

        # Per-modality CNN decoders: d_token -> (28, n_feats)
        self.modality_decoders = nn.ModuleDict()
        for name, indices in modality_indices.items():
            n_feats = len(indices)
            self.modality_decoders[name] = ModalityCNNDecoder(
                n_feats=n_feats, conv_channels=conv_channels, d_token=d_token)

    def forward(self, visible_tokens, masked_modality_names):
        """Reconstruct masked modalities from visible tokens.

        Args:
            visible_tokens: (batch, n_visible, d_token) from backbone
            masked_modality_names: list of modality names to reconstruct

        Returns:
            dict mapping modality name -> (batch, 28, n_feats) reconstructions
        """
        batch_size = visible_tokens.size(0)

        # Build query tokens for masked modalities: (batch, n_masked, d_token)
        queries = torch.stack(
            [self.query_tokens[name] for name in masked_modality_names], dim=0)
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)  # (B, n_masked, d)

        # Cross-attention: queries attend to visible tokens
        for attn, norm_q, norm_kv, ff, ff_norm in zip(
                self.cross_attn_layers, self.cross_norms_q, self.cross_norms_kv,
                self.cross_ff, self.cross_ff_norms):
            q = norm_q(queries)
            kv = norm_kv(visible_tokens)
            attn_out, _ = attn(q, kv, kv)
            queries = queries + attn_out
            ff_out = ff(ff_norm(queries))
            queries = queries + ff_out

        # Decode each masked modality
        recon = {}
        for i, name in enumerate(masked_modality_names):
            token = queries[:, i, :]  # (batch, d_token)
            recon[name] = self.modality_decoders[name](token)  # (batch, 28, n_feats)

        return recon


class ModalityCNNDecoder(nn.Module):
    """Small 1D-CNN decoder for a single modality's temporal reconstruction.

    Mirrors ModalityCNNEncoder: d_token -> unflatten -> ConvTranspose1d -> (28, n_feats)

    Input:  (batch, d_token)
    Output: (batch, 28, n_feats)
    """

    def __init__(self, n_feats, conv_channels=8, d_token=32):
        super().__init__()
        self._seq_len_reduced = 7
        self._conv_channels = conv_channels

        self.fc = nn.Linear(d_token, conv_channels * self._seq_len_reduced)

        self.deconv = nn.Sequential(
            # 7 -> 14
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(conv_channels),
            # 14 -> 28
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(conv_channels),
            # final conv to output features
            nn.ConvTranspose1d(conv_channels, n_feats, kernel_size=3, padding=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, d_token)
        x = F.relu(self.fc(x))
        x = x.view(x.size(0), self._conv_channels, self._seq_len_reduced)  # (B, C, 7)
        x = self.deconv(x)  # (B, n_feats, 28)
        x = x.transpose(1, 2)  # (B, 28, n_feats)
        return x


# ─── Composite model builders ───────────────────────────────────────────

def build_erm_model(backbone, input_dim):
    """Build an ERM (classification-only) model.

    Args:
        backbone: nn.Module returning (batch, embed_dim) embeddings
        input_dim: embedding dimension from backbone

    Returns:
        (backbone, cls_head) tuple
    """
    cls_head = LabelHead(input_dim, num_classes=2)
    return backbone, cls_head


def build_reorder_model(backbone, input_dim, num_reorder_classes=200):
    """Build a reorder model with label + reorder heads.

    Returns:
        (backbone, label_head, reorder_head) tuple
    """
    label_head = LabelHead(input_dim, num_classes=2)
    reorder_head = ReorderHead(input_dim, num_reorder_classes)
    return backbone, label_head, reorder_head
