"""
PyTorch base classes for depression detection algorithms.

Provides:
- DataRepo_torch: PyTorch-compatible data repository (replaces DataRepo_tf)
- DepressionDetectionClassifier_DL_torch: PyTorch classifier with sklearn interface
- DepressionDetectionAlgorithm_DL_torch: Algorithm class extending DL_erm for data loading

These classes reuse the existing data loading pipeline (DataRepo_np, feature selection,
train/val splitting) but bypass TensorFlow entirely for model training and inference.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from copy import deepcopy
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from algorithm.dl_erm import DepressionDetectionAlgorithm_DL_erm
from algorithm.base import DepressionDetectionClassifierBase
from data_loader.data_loader_ml import DatasetDict, DataRepo
from utils import path_definitions
from utils.torch_training import (
    MixupDataset,
    TorchTrainer,
    load_nan_masks,
    AutoencoderImputer,
)


# ─── Data Repository ────────────────────────────────────────────────────


class TrainingData:
    """Holds numpy arrays for the training pipeline:
    - train_X, train_y: training split
    - val_X, val_y: validation split (one random sample per person)
    - whole_X, whole_y: entire training set (for evaluation callbacks)
    - test_X, test_y: optional test set (for on_test best epoch selection)
    """

    def __init__(self, train_X, train_y, val_X, val_y, whole_X, whole_y,
                 test_X=None, test_y=None):
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y
        self.whole_X = whole_X
        self.whole_y = whole_y
        self.test_X = test_X
        self.test_y = test_y


class DataRepo_torch:
    """PyTorch-compatible data repository replacing DataRepo_tf.

    For flag_train=True:
        - X: TrainingData container (not dict, not FlatMapDataset)
        - y: None
        - pids: None

    For flag_train=False:
        - X: np.ndarray of shape (N, 28, F)
        - y: np.ndarray of shape (N,) — class labels
        - pids: np.ndarray of shape (N,)

    Implements get_eval_repo() for the harness compatibility hook in
    train_eval_pipeline.py.
    """

    def __init__(self, X, y=None, pids=None, eval_repo=None):
        self.X = X
        self.y = y
        self.pids = pids
        self._eval_repo = eval_repo

    def get_eval_repo(self):
        """Return an evaluation-mode DataRepo_torch for the training data.
        """
        if self._eval_repo is not None:
            return self._eval_repo
        # Fallback: if TrainingData, use whole set
        if isinstance(self.X, TrainingData):
            y_labels = np.argmax(self.X.whole_y, axis=1) if self.X.whole_y.ndim > 1 else self.X.whole_y
            return DataRepo_torch(X=self.X.whole_X, y=y_labels, pids=None)
        return self


# ─── Classifier ──────────────────────────────────────────────────────────


class DepressionDetectionClassifier_DL_torch(DepressionDetectionClassifierBase):
    """PyTorch classifier with sklearn-compatible fit/predict/predict_proba interface.

    Subclasses should override _build_model_parts() to define the specific
    architecture (backbone + heads) and _train() to run the appropriate
    training loop (ERM, reorder, MAE, etc.).
    """

    def __init__(self, config):
        self.config = config
        self.model_params = config["model_params"]
        self.training_params = config["training_params"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model parts (backbone + heads)
        self.model_parts = self._build_model_parts()
        self.trainer = None

        # Results save folder (same pattern as dl_erm.py)
        folder_name = self.config["data_loader"]["training_dataset_key"].replace(":", "_")
        self.results_save_folder = os.path.join(
            path_definitions.TMP_PATH,
            self.config["name"],
            "input_shape-" + "_".join(str(i) for i in self.model_params["input_shape"]),
            self.config["training_params"].get("eval_task", ""),
            folder_name,
        )
        os.makedirs(self.results_save_folder, exist_ok=True)

    def _build_model_parts(self):
        """Build and return a dict of nn.Module model parts.

        Must be overridden by subclasses. Example return:
            {"backbone": BehavioralTransformerEncoder(...),
             "cls_head": LabelHead(...)}
        """
        raise NotImplementedError

    def _make_train_loader(self, training_data):
        """Create a DataLoader for training from TrainingData.

        Can be overridden by subclasses (e.g., ReorderDataset needs different Dataset).
        """
        dataset = MixupDataset(
            X=training_data.train_X,
            y=training_data.train_y,
            mixup_alpha=self.config["data_loader"].get("mixup_alpha"),
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
        """Run the training loop. Override for different training modes.

        Args:
            training_data: TrainingData container

        Returns:
            pd.DataFrame of per-epoch results
        """
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
        return self.trainer.train_erm(
            train_loader, eval_data_val=eval_val, eval_data_test=eval_test
        )

    def fit(self, X, y=None):
        """Train the model. Called by the harness as clf.fit(X=data_repo.X, y=data_repo.y).

        Args:
            X: TrainingData container (from DataRepo_torch with flag_train=True)
            y: unused (labels are inside X)
        """
        import random
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        assert isinstance(X, TrainingData), (
            f"Expected TrainingData, got {type(X)}. "
            "Make sure prep_data_repo(flag_train=True) returns DataRepo_torch."
        )

        skip_training = self.training_params.get("skip_training", False)

        if not skip_training:
            df_results_record = self._train(X)

            # Find best epoch and restore weights
            strategy = self.training_params.get("best_epoch_strategy", "direct")
            best_epoch = self.trainer.find_best_epoch(df_results_record, strategy=strategy)
            self.trainer._load_all_state_dicts(self.trainer.model_repo_dict[best_epoch])

            if self.training_params.get("verbose", 0) > 0:
                print(f"Best epoch: {best_epoch}")

            # Save results
            self.trainer.save_epoch_results(df_results_record, self.results_save_folder)
        else:
            df_results_record = self._fit_skip_training()

        return df_results_record

    def _fit_skip_training(self):
        """Load previously saved results and restore best weights."""
        file_path = os.path.join(self.results_save_folder, "results_saving.pkl")
        with open(file_path, "rb") as f:
            save_obj = pickle.load(f)

        df_results_record = save_obj["df_results_record"]
        self.trainer = TorchTrainer(
            model_parts=self.model_parts,
            training_params=self.training_params,
            device=self.device,
        )
        self.trainer.model_repo_dict = save_obj["model_repo_dict"]

        strategy = self.training_params.get("best_epoch_strategy", "direct")
        best_epoch = self.trainer.find_best_epoch(df_results_record, strategy=strategy)
        self.trainer._load_all_state_dicts(self.trainer.model_repo_dict[best_epoch])

        if self.training_params.get("verbose", 0) > 0:
            print(f"Skip training, best epoch: {best_epoch}")

        return df_results_record

    def predict(self, X, y=None):
        """Predict class labels. X is a numpy array (N, 28, F)."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X, y=None):
        """Predict class probabilities. X is a numpy array (N, 28, F).

        Returns np.ndarray of shape (N, 2).
        """
        if self.trainer is None:
            # Build a trainer just for inference
            self.trainer = TorchTrainer(
                model_parts=self.model_parts,
                training_params=self.training_params,
                device=self.device,
            )
        return self.trainer.predict_proba_np(X)


# ─── Algorithm ───────────────────────────────────────────────────────────


class DepressionDetectionAlgorithm_DL_torch(DepressionDetectionAlgorithm_DL_erm):
    """PyTorch algorithm extending DL_erm to reuse data loading.

    Overrides prep_data_repo to produce DataRepo_torch instead of DataRepo_tf,
    and prep_model to return a PyTorch classifier.

    The train/val splitting logic is replicated from DL_erm.prep_data_repo()
    to produce numpy arrays directly, bypassing the TF data pipeline.
    """

    def __init__(self, config_dict=None, config_name="dl_torch_erm"):
        super().__init__(config_dict=config_dict, config_name=config_name)
        # We don't need the TF data generator objects
        self.data_generator_obj = None
        self.data_generator_additional_args = {"train": {}, "nontrain": {}}

    def prep_data_repo(self, dataset: DatasetDict, flag_train: bool = True):
        """Prepare data repo for PyTorch models.

        Replicates the data selection and train/val splitting from
        DepressionDetectionAlgorithm_DL_erm.prep_data_repo() but returns
        DataRepo_torch instead of DataRepo_tf.
        """
        # We still need the parent's data repo setup for config/key tracking,
        # but we skip the TF-specific parts by calling the grandparent's prep_data_repo
        # and then doing our own splitting.
        from algorithm.base import DepressionDetectionAlgorithmBase
        DepressionDetectionAlgorithmBase.prep_data_repo(self, flag_train=flag_train)

        np.random.seed(42)

        self.config["data_loader"]["training_dataset_key"] = dataset.key
        ds_keys = dataset.key.split(":")

        pred_target = dataset.prediction_target
        input_shape = self.input_shape_dict[pred_target]
        ds_key_all = self.data_repo_np_dict[pred_target].keys()
        self.input_shape = input_shape

        if hasattr(dataset, "eval_task"):
            self.config["training_params"]["eval_task"] = dataset.eval_task
        else:
            self.config["training_params"]["eval_task"] = ""

        # ── Filtering logic (replicated from dl_erm.py) ──

        flag_overlap_filter = hasattr(dataset, "flag_overlap_filter")
        if flag_overlap_filter:
            overlap_group, overlap_ds_key_train = dataset.flag_overlap_filter.split(":")
            assert overlap_group in ["test", "train"]
            assert len(ds_keys) == 1

        flag_split_filter = hasattr(dataset, "flag_split_filter")
        if flag_split_filter:
            split_group, split_fold = dataset.flag_split_filter.split(":")
            assert split_group in ["test", "train"]

        flag_single_within_user_split = hasattr(dataset, "flag_single_within_user_split")
        if flag_single_within_user_split:
            within_split_group, within_split_ratio = dataset.flag_single_within_user_split.split(":")
            assert within_split_group in ["train", "test"]
            if within_split_group == "train":
                within_split_ratio = float(within_split_ratio)
            else:
                within_split_ratio = 1 - float(within_split_ratio)
            assert 0 < within_split_ratio < 1
            assert len(ds_keys) == 1

        assert sum(1 for flag in [flag_overlap_filter, flag_split_filter, flag_single_within_user_split] if flag) <= 1

        # Build filtered data_repo_dict
        data_repo_dict = {}
        for ds_key in ds_keys:
            data_repo_dict[ds_key] = self.data_repo_np_dict[pred_target][ds_key]
            if flag_overlap_filter:
                idx = np.where([
                    p in set(self.overlapping_pids_dict[dataset.prediction_target][overlap_ds_key_train][ds_key])
                    for p in data_repo_dict[ds_key].pids
                ])[0]
                data_repo_dict[ds_key] = data_repo_dict[ds_key][idx]
            if flag_split_filter:
                if split_group == "test":
                    split_pids = np.array(
                        self.split_5fold_pids_dict[dataset.prediction_target][ds_key][str(split_fold)]
                    )
                elif split_group == "train":
                    split_pids = [
                        self.split_5fold_pids_dict[dataset.prediction_target][ds_key][str(i)]
                        for i in range(1, 6) if i != int(split_fold)
                    ]
                    split_pids = np.concatenate(split_pids)
                idx = np.where([p in set(split_pids) for p in data_repo_dict[ds_key].pids])[0]
                data_repo_dict[ds_key] = data_repo_dict[ds_key][idx]
            if flag_single_within_user_split:
                df_data_idx = pd.DataFrame(
                    self.data_repo_np_dict[pred_target][ds_key].pids, columns=["pid"]
                ).groupby("pid").apply(lambda x: np.array(x.index))
                df_data_idx = pd.DataFrame(df_data_idx, columns=["idx"])
                df_data_idx["length"] = df_data_idx["idx"].apply(len)
                df_data_idx["length_test"] = df_data_idx["length"].apply(
                    lambda x: int(np.ceil(x * (1 - within_split_ratio)))
                )
                df_data_idx["idx_train"] = df_data_idx.apply(
                    lambda row: row["idx"][:-row["length_test"]], axis=1
                )
                df_data_idx["idx_test"] = df_data_idx.apply(
                    lambda row: row["idx"][-row["length_test"]:], axis=1
                )
                if within_split_group == "train":
                    data_idx = np.concatenate(df_data_idx["idx_train"].values)
                elif within_split_group == "test":
                    data_idx = np.concatenate(df_data_idx["idx_test"].values)
                data_repo_dict[ds_key] = data_repo_dict[ds_key][data_idx]

        if flag_train:
            # ── Train/val split (same logic as dl_erm.py) ──
            # Build person dict for splitting
            all_pids = {}
            for ds_key in ds_keys:
                pids = data_repo_dict[ds_key].pids
                for i, p in enumerate(pids):
                    if p not in all_pids:
                        all_pids[p] = {"ds_key": ds_key, "indices": []}
                    all_pids[p]["indices"].append(i)

            # Per-person: randomly take one sample for validation, rest for training
            per_ds_train_idx = {ds_key: [] for ds_key in ds_keys}
            per_ds_val_idx = {ds_key: [] for ds_key in ds_keys}

            for pid, info in all_pids.items():
                ds_key = info["ds_key"]
                indices = info["indices"]
                if len(indices) == 1:
                    per_ds_val_idx[ds_key].append(indices[0])
                    per_ds_train_idx[ds_key] += indices
                else:
                    val_pos = np.random.randint(0, len(indices))
                    per_ds_val_idx[ds_key].append(indices[val_pos])
                    per_ds_train_idx[ds_key] += indices[:val_pos] + indices[val_pos + 1:]

            # Concatenate across datasets
            train_X = np.concatenate([data_repo_dict[k].X[per_ds_train_idx[k]] for k in ds_keys])
            train_y = np.concatenate([data_repo_dict[k].y[per_ds_train_idx[k]] for k in ds_keys])
            val_X = np.concatenate([data_repo_dict[k].X[per_ds_val_idx[k]] for k in ds_keys])
            val_y = np.concatenate([data_repo_dict[k].y[per_ds_val_idx[k]] for k in ds_keys])
            whole_X = np.concatenate([data_repo_dict[k].X for k in ds_keys])
            whole_y = np.concatenate([data_repo_dict[k].y for k in ds_keys])

            # Optional test set (when training on N-1 datasets, the Nth is test)
            test_X, test_y = None, None
            if len(ds_keys) == len(ds_key_all) - 1:
                ds_key_rest = [i for i in ds_key_all if i not in ds_keys][0]
                rest_repo = self.data_repo_np_dict[pred_target][ds_key_rest]
                if flag_overlap_filter:
                    idx = np.where([
                        p in set(self.overlapping_pids_dict[dataset.prediction_target][overlap_ds_key_train][ds_key_rest])
                        for p in rest_repo.pids
                    ])[0]
                    rest_repo = rest_repo[idx]
                test_X = rest_repo.X
                test_y = rest_repo.y

            # ── Autoencoder imputation (IMP-1) ──
            if self.config.get("imputation") == "autoencoder":
                imp_params = self.config.get("imputation_params", {})
                nan_masks = load_nan_masks(
                    list(ds_key_all), pred_target=pred_target,
                    flag_more_feat_types=True)
                if nan_masks is not None:
                    # Split NaN masks with same indices as data
                    # Note: nan_masks are full-feature (before feature selection),
                    # so select the same feature indices used by the model
                    from data_loader.data_loader_dl import dl_feat_preparation
                    from utils.common_settings import global_config
                    _fp = dl_feat_preparation(
                        config_name="dl_feat_prep",
                        flag_more_feat_types=global_config["all"]["flag_more_feat_types"])
                    si = _fp.selected_feature_idx

                    train_mask = np.concatenate([nan_masks[k][per_ds_train_idx[k]][:, :, si] for k in ds_keys])
                    val_mask = np.concatenate([nan_masks[k][per_ds_val_idx[k]][:, :, si] for k in ds_keys])

                    # Train AE on training split
                    T, F_dim = train_X.shape[1], train_X.shape[2]
                    imputer = AutoencoderImputer(
                        input_dim=T * F_dim,
                        bottleneck_dim=imp_params.get("bottleneck_dim", 20))
                    imputer.fit(train_X, train_mask,
                                epochs=imp_params.get("epochs", 10),
                                lr=imp_params.get("lr", 1e-3),
                                verbose=self.config["training_params"].get("verbose", 0))

                    # Apply trained AE to all splits
                    train_X = imputer.transform(train_X, train_mask)
                    val_X = imputer.transform(val_X, val_mask)
                    whole_mask = np.concatenate([nan_masks[k][:, :, si] for k in ds_keys])
                    whole_X = imputer.transform(whole_X, whole_mask)

                    if test_X is not None and ds_key_rest in nan_masks:
                        test_mask = nan_masks[ds_key_rest][:, :, si]
                        if flag_overlap_filter:
                            test_mask = test_mask[idx]
                        test_X = imputer.transform(test_X, test_mask)

            training_data = TrainingData(
                train_X=train_X, train_y=train_y,
                val_X=val_X, val_y=val_y,
                whole_X=whole_X, whole_y=whole_y,
                test_X=test_X, test_y=test_y,
            )

            # Build the eval repo for later use by the harness
            whole_y_labels = np.argmax(whole_y, axis=1) if whole_y.ndim > 1 else whole_y
            whole_pids = np.concatenate([data_repo_dict[k].pids for k in ds_keys])
            eval_repo = DataRepo_torch(X=whole_X, y=whole_y_labels, pids=whole_pids)

            self.data_repo = DataRepo_torch(X=training_data, y=None, pids=None, eval_repo=eval_repo)
        else:
            # ── Test mode: just return all data as numpy ──
            X = np.concatenate([data_repo_dict[k].X for k in ds_keys])
            y_raw = np.concatenate([data_repo_dict[k].y for k in ds_keys])
            y = np.argmax(y_raw, axis=1) if y_raw.ndim > 1 else y_raw
            pids = np.concatenate([data_repo_dict[k].pids for k in ds_keys])
            self.data_repo = DataRepo_torch(X=X, y=y, pids=pids)

        return self.data_repo

    def prep_model(self, data_train=None, criteria="balanced_acc"):
        """Define a PyTorch classifier. Must be overridden by subclasses."""
        self.config["model_params"].update({
            "input_shape": self.input_shape,
        })
        raise NotImplementedError(
            "DepressionDetectionAlgorithm_DL_torch.prep_model() must be overridden"
        )
