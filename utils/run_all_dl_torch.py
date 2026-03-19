"""Run all dl_torch model configs across all evaluation tasks.

Usage:
    python run_all.py                  # run all configs x all tasks, 5 parallel
    python run_all.py --max_parallel 3 # run 3 at a time
    python run_all.py --dry_run        # print commands without executing
"""

import argparse
import subprocess
import time
from itertools import product

CONFIGS = [
    "dl_torch_erm_transformer",
    "dl_torch_mae_transformer",
    "dl_torch_mae_cnn",
    "dl_torch_modality_token",
    "dl_torch_modality_token_reorder",
    "dl_torch_modality_token_reorder_aug",
    "dl_torch_modality_token_reorder_imp",
    "dl_torch_modality_token_reorder_mae",
    "dl_torch_reorder_cnn",
    "dl_torch_reorder_cnn_aug",
    "dl_torch_reorder_cnn_imp",
    "dl_torch_reorder_deep",
    "dl_torch_reorder_transformer",
]

TASKS = [
    "allbutone",
    "single_within_user",
    "crosscovid",
    "two_overlap",
]


def run_all(max_parallel=5, dry_run=False):
    jobs = list(product(CONFIGS, TASKS))
    total = len(jobs)
    print(f"Total runs: {total} ({len(CONFIGS)} configs x {len(TASKS)} tasks)")
    print(f"Max parallel: {max_parallel}")
    print()

    if dry_run:
        for config, task in jobs:
            print(f"python evaluation/model_train_eval.py "
                  f"--config_name={config} --pred_target=dep_weekly "
                  f"--eval_task={task} --verbose 0")
        return

    active = []
    completed = 0
    failed = []

    for config, task in jobs:
        # Wait if at max parallel capacity
        while len(active) >= max_parallel:
            active, newly_done, newly_failed = _check_active(active)
            completed += newly_done
            failed.extend(newly_failed)
            if len(active) >= max_parallel:
                time.sleep(5)

        cmd = [
            "python", "evaluation/model_train_eval.py",
            f"--config_name={config}",
            "--pred_target=dep_weekly",
            f"--eval_task={task}",
            "--verbose", "0",
        ]
        log_file = f"logs/{config}__{task}.log"
        print(f"[{completed + len(active) + 1}/{total}] Starting {config} / {task}")

        import os
        os.makedirs("logs", exist_ok=True)
        log_fh = open(log_file, "w")
        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
        active.append((proc, log_fh, config, task))

    # Wait for remaining
    while active:
        active, newly_done, newly_failed = _check_active(active)
        completed += newly_done
        failed.extend(newly_failed)
        if active:
            time.sleep(5)

    print(f"\nDone. {completed} completed, {len(failed)} failed.")
    if failed:
        print("Failed runs:")
        for config, task, rc in failed:
            print(f"  {config} / {task} (exit code {rc})")
            print(f"  Log: logs/{config}__{task}.log")


def _check_active(active):
    still_active = []
    newly_done = 0
    newly_failed = []
    for proc, log_fh, config, task in active:
        rc = proc.poll()
        if rc is None:
            still_active.append((proc, log_fh, config, task))
        else:
            log_fh.close()
            if rc == 0:
                print(f"  Completed: {config} / {task}")
                newly_done += 1
            else:
                print(f"  FAILED: {config} / {task} (exit code {rc})")
                newly_failed.append((config, task, rc))
                newly_done += 1
    return still_active, newly_done, newly_failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_parallel", type=int, default=5)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    run_all(max_parallel=args.max_parallel, dry_run=args.dry_run)
