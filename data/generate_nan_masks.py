"""One-time script to generate NaN mask pickle files from raw data.

For each dataset, re-reads the raw DataFrames, extracts the 28-day windows,
and saves a boolean mask (True = originally NaN) before median imputation.

Run once: python data/generate_nan_masks.py
"""

import os
import sys
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from tqdm import tqdm

from utils.common_settings import global_config
from data_loader import data_loader_ml
from data_loader.data_loader_dl import dl_feat_preparation
from utils import path_definitions


def generate_nan_masks():
    ds_keys = global_config["all"]["ds_keys"]
    pred_targets = global_config["all"]["prediction_tasks"]
    flag_more_feat_types = global_config["all"]["flag_more_feat_types"]

    feat_prep = dl_feat_preparation(
        config_name="dl_feat_prep",
        flag_more_feat_types=flag_more_feat_types,
        verbose=1,
    )

    for pred_target in pred_targets:
        for ds_key in ds_keys:
            print(f"Generating NaN mask for {pred_target} -- {ds_key}")
            institution, phase = ds_key.split("_")
            phase = int(phase)

            # Load raw dataset
            dataset = data_loader_ml.data_loader_single(
                pred_target, institution, phase,
                flag_more_feat_types=flag_more_feat_types,
            )

            # Extract 28-day windows (same as prep_data_repo)
            from copy import deepcopy
            df_datapoints = deepcopy(dataset.datapoints)
            df_datapoints_X = df_datapoints["X_raw"].apply(
                lambda df: df[feat_prep.feature_list].iloc[-28:]
            )

            # Compute NaN masks before imputation
            nan_masks = []
            for df in tqdm(df_datapoints_X.values, desc=f"  {ds_key}"):
                nan_masks.append(df.isna().values)

            nan_mask_arr = np.array(nan_masks)
            print(f"  Shape: {nan_mask_arr.shape}, "
                  f"NaN rate: {100 * nan_mask_arr.mean():.1f}%")

            # Save alongside the numpy data files
            if flag_more_feat_types:
                out_dir = os.path.join(path_definitions.DATA_PATH, "np_max_feature_types")
            else:
                out_dir = os.path.join(path_definitions.DATA_PATH, "np")

            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{pred_target}--{ds_key}--nan_mask.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(nan_mask_arr, f)
            print(f"  Saved: {out_path}")


if __name__ == "__main__":
    generate_nan_masks()
