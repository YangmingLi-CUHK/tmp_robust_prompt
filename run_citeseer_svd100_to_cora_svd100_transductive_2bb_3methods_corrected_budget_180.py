#!/usr/bin/env python3
"""Run the fixed Peak/Stable Citeseer-SVD100 backbones on corrected-budget Cora attacks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import socket
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME = "citeseer_svd100_to_cora_svd100_transductive_2bb_3methods_corrected_budget_180"
PTB_RATES = ("0.00", "0.05", "0.10", "0.15", "0.20", "0.25")
DOWNSTREAM_SEEDS = (1, 2, 3, 4, 5)

CANONICAL_FEATURE_SHA256 = "cba12dbb6b543cf81e601fb29eebc8d2897c35d2455ae6b51658dc95c94228e5"
CANONICAL_LABEL_SHA256 = "1f2fde4fd4b4aca1a4ca053376fb00f5ebeb8fa3e04e8b2a9c0bfd273ca1c83b"
CANONICAL_INDEX_SHA256 = {
    "train": "1d2230968368cac607798c04c24d6b634ca2c0e92f3149cc48efb1b0d562dec8",
    "val": "00838368d5334cfef5493e7b33c635f57efd307c4dbedd602c178c76683db299",
    "test": "37fc182a19c25253f522562d4ecd6a533676928f43530ffe781c67d2c342186f",
}
SOURCE_SVD_RECEIPT = {
    "dataset": "Citeseer_Nettack_LCC",
    "original_shape": [2110, 3703],
    "original_x_sha256": "b3ae8a20c3ad485d49b3357b5a7537963e388dceee1005032310f90b877b8c63",
    "preprocessing": "raw_bow_to_l1_to_independent_svd100",
    "reduced_shape": [2110, 100],
    "reduced_x_sha256": "c5acb8034c33d907e4aec92fda71254e59b556367d9c96a92964f75044bbbb3e",
    "svd_backend": "torch_geometric.SVDFeatureReduction(torch.linalg.svd)",
    "torch_geometric_version": "2.7.0",
    "torch_version": "2.8.0+cu129",
    "version": 1,
}

BACKBONES = (
    {
        "id": "peak_bb",
        "label": "Peak BB",
        "pretrain_seed": 1,
        "filename": (
            "Citeseer.GraphCL.GCN.256_hidden_dim.preprocess_svd_100."
            "aug1_dropN.aug2_dropN.lr_0.001.ratio_0.1.seed_1.pth"
        ),
        "sha256": "25bfafe8bcfd4495df6a49542b7e64c2b1897468bcfdcb6cf71cddc7c53a84d4",
        "source_svd_sha256": SOURCE_SVD_RECEIPT["reduced_x_sha256"],
        "clean_val_accuracy": "0.535849",
        "clean_test_accuracy": "0.531561",
        "selection": (
            "highest single-checkpoint clean validation accuracy among all 135 runs; "
            "dropN/dropN,ratio=0.1,lr=0.001,pretrain_seed=1"
        ),
        "selection_group_val_mean": "",
        "selection_group_val_std": "",
    },
    {
        "id": "stable_bb",
        "label": "Stable BB",
        "pretrain_seed": 1,
        "filename": (
            "Citeseer.GraphCL.GCN.256_hidden_dim.preprocess_svd_100."
            "aug1_maskN.aug2_dropN.lr_0.001.ratio_0.2.seed_1.pth"
        ),
        "sha256": "a0e2ad626e53016ab73da1db02480da203a13b87e3a30e32a6d9f1214c08a51b",
        "source_svd_sha256": SOURCE_SVD_RECEIPT["reduced_x_sha256"],
        "clean_val_accuracy": "0.516981",
        "clean_test_accuracy": "0.559801",
        "selection": (
            "highest clean validation checkpoint within the rank-1 complete-5-seed "
            "validation-mean group; maskN/dropN,ratio=0.2,lr=0.001,pretrain_seed=1"
        ),
        "selection_group_val_mean": "0.464151",
        "selection_group_val_std": "0.044522",
    },
)

EXPECTED_CHECKPOINT_SHA256 = {row["id"]: row["sha256"] for row in BACKBONES}
EXPECTED_CLEAN_REPLAY = {
    row["id"]: {
        "val_accuracy": row["clean_val_accuracy"],
        "test_accuracy": row["clean_test_accuracy"],
    }
    for row in BACKBONES
}

EXPECTED_ATTACKS = {
    "0.00": {"added": 0, "deleted": 0, "edges": 5278, "runtime_edges": 13264,
             "sha256": "57b4528c357b3b8ff5ed44ecca47f3de42b84f36c4814961e048c48e67bd65ce"},
    "0.05": {"added": 257, "deleted": 6, "edges": 5529, "runtime_edges": 13766,
             "sha256": "660e5ad3b7182007c2ba351e0160c20981bb30345246050b8497b43f111edb80"},
    "0.10": {"added": 488, "deleted": 39, "edges": 5727, "runtime_edges": 14162,
             "sha256": "412dd306682d73f6a0adf918fac029ce661c50f1c20935f8fb5bdc92d715f74a"},
    "0.15": {"added": 728, "deleted": 63, "edges": 5943, "runtime_edges": 14594,
             "sha256": "e785143e7127598e14465536337a4a55b6f8a6f5888b0cba083e4e9e3cf54f68"},
    "0.20": {"added": 976, "deleted": 79, "edges": 6175, "runtime_edges": 15058,
             "sha256": "cb71b9db4fc517f15c3f2631dd9dc686589cb82e35b30ba307fb525bf440ff31"},
    "0.25": {"added": 1212, "deleted": 107, "edges": 6383, "runtime_edges": 15474,
             "sha256": "ac4192398d9be424fb15d80df2a0090115f49a5f78fee24fe4870a3b55ed2824"},
}

METHODS = (
    {
        "id": "gpromptshield_original_matched",
        "prompt_type": "RobustPrompt-T",
        "prompt_variant": "original",
        "filter_mode": "original",
        "effective_edge_path": "feature_cosine_tau_tune;filter_mode_argument_ignored",
        "prompt_space": "Cora_SVD100_input",
        "description": (
            "Original implementation with the 2026-07-30 Citeseer-SVD1433 transfer-script "
            "matched settings"
        ),
    },
    {
        "id": "gpromptshield_ours",
        "prompt_type": "RobustPrompt-T",
        "prompt_variant": "ours",
        "filter_mode": "original",
        "effective_edge_path": (
            "two_pass_embedding_cosine_tau_tune;configured_filter_diagnostic_only"
        ),
        "prompt_space": "Cora_SVD100_input",
        "description": (
            "Repository modified implementation with the same 2026-07-30 matched settings"
        ),
    },
    {
        "id": "gppt_repo_frozen_graphcl_nofilter",
        "prompt_type": "GPPT",
        "prompt_variant": "not_applicable",
        "filter_mode": "none",
        "effective_edge_path": "no_filter",
        "prompt_space": "frozen_GCN_embedding256",
        "description": "Repository GPPT adaptation on frozen GraphCL, without the added cosine filter",
    },
)

FORMAL_RUN_COUNT = len(BACKBONES) * len(METHODS) * len(PTB_RATES) * len(DOWNSTREAM_SEEDS)
FORMAL_GROUP_COUNT = len(BACKBONES) * len(METHODS) * len(PTB_RATES)

RESULT_FIELDS = [
    "run_key", "status", "backbone_id", "backbone_label", "backbone_selection",
    "method", "ptb_rate", "pretrain_seed", "downstream_seed",
    "source_dataset", "target_dataset", "feature_alignment", "target_preprocessing",
    "encoder_dimensions", "prompt_space", "prompt_variant", "filter_mode_argument",
    "effective_edge_path", "epochs",
    "shot", "split", "train_n", "val_n", "test_n", "attack_added", "attack_deleted",
    "attack_modifications", "attack_realized_rate", "attack_edges", "runtime_edges",
    "attack_graph_sha256",
    "feature_file_sha256", "label_file_sha256", "train_idx_sha256", "val_idx_sha256",
    "test_idx_sha256",
    "target_svd_sha256", "target_cache_file_sha256", "target_original_x_sha256",
    "source_svd_sha256",
    "checkpoint_sha256", "checkpoint", "test_accuracy", "test_macro_f1", "code_sha256",
    "config_sha256", "log_path", "log_sha256", "recorded_utc", "error",
]

SUMMARY_FIELDS = [
    "backbone_id", "backbone_label", "method", "ptb_rate", "selection_status",
    "n_seeds", "seeds_ok", "seeds_missing",
    "test_accuracy_mean", "test_accuracy_std", "test_macro_f1_mean", "test_macro_f1_std",
]

FINAL_RESULT_RE = re.compile(
    r"^FINAL_RESULT \| test_accuracy=(?P<accuracy>[0-9.eE+-]+) "
    r"\| macro_f1=(?P<macro_f1>[0-9.eE+-]+)$"
)


class RunFailure(RuntimeError):
    def __init__(self, message: str, log_path: Path):
        super().__init__(message)
        self.log_path = log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    asset_dir = "experiment_assets/citeseer_svd100_transductive"
    parser.add_argument("--checkpoint-dir", default=asset_dir)
    parser.add_argument(
        "--target-cache",
        default="data/preprocessed/cora_clean_full_l1_svd_100.pt",
    )
    parser.add_argument(
        "--output-dir",
        default=f"logs/{EXPERIMENT_NAME}",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print all 180 hard-coded run keys without loading artifacts.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate all server artifacts and write manifests, but do not start training.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RESULT_FIELDS:
            raise RuntimeError(f"Unexpected result CSV schema in {path}: {reader.fieldnames}")
        rows = list(reader)
    keys = [row["run_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate run_key values in {path}.")
    return rows


def selected_checkpoint_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    checkpoint_dir = Path(args.checkpoint_dir)
    return [dict(row, path=checkpoint_dir / str(row["filename"])) for row in BACKBONES]


def build_plan(selected: list[dict[str, object]]) -> list[dict[str, object]]:
    plan = []
    for checkpoint in selected:
        for method in METHODS:
            for ptb_rate in PTB_RATES:
                for downstream_seed in DOWNSTREAM_SEEDS:
                    plan.append(
                        {
                            "run_key": (
                                f"backbone={checkpoint['id']}|method={method['id']}|"
                                f"ptb={ptb_rate}|downstream_seed={downstream_seed}"
                            ),
                            "method": method,
                            "ptb_rate": ptb_rate,
                            "pretrain_seed": checkpoint["pretrain_seed"],
                            "downstream_seed": downstream_seed,
                            "checkpoint": checkpoint,
                        }
                    )
    if (
        len(plan) != FORMAL_RUN_COUNT
        or len({row["run_key"] for row in plan}) != FORMAL_RUN_COUNT
    ):
        raise RuntimeError(
            f"The fixed two-backbone plan must contain exactly {FORMAL_RUN_COUNT} unique runs."
        )
    return plan


def validate_checkpoint(path: Path, expected_hash: str) -> None:
    import torch

    from prompt_graph.model import GCN

    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch for {path.name}: expected {expected_hash}, got {actual_hash}."
        )
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"Checkpoint is not a non-empty state_dict: {path}")
    if any(not bool(torch.isfinite(value).all()) for value in state_dict.values()):
        raise RuntimeError(f"Checkpoint contains non-finite values: {path}")
    GCN(input_dim=100, hid_dim=256, num_layer=2).load_state_dict(state_dict, strict=True)


def code_bundle_sha256() -> str:
    paths = [
        Path(__file__).resolve(),
        PROJECT_ROOT / "run_citeseer_svd100_to_cora_svd100_transductive_2bb_3methods_corrected_budget_180.sh",
        PROJECT_ROOT / "MyTask.py",
        PROJECT_ROOT / "eval_citeseer_svd100_to_cora_svd100.py",
        PROJECT_ROOT / "eval_pretrain.py",
        PROJECT_ROOT / "project_paths.py",
        PROJECT_ROOT / "prompt_graph/utils/downstream_svd_cache.py",
        PROJECT_ROOT / "prompt_graph/utils/__init__.py",
        PROJECT_ROOT / "prompt_graph/utils/seed.py",
        PROJECT_ROOT / "prompt_graph/utils/constraint.py",
        PROJECT_ROOT / "prompt_graph/utils/loss.py",
        PROJECT_ROOT / "prompt_graph/utils/center_embedding.py",
        PROJECT_ROOT / "prompt_graph/utils/robustpt.py",
        PROJECT_ROOT / "prompt_graph/utils/edge_anomaly_metrics.py",
        PROJECT_ROOT / "prompt_graph/data/load4data.py",
        PROJECT_ROOT / "prompt_graph/data/__init__.py",
        PROJECT_ROOT / "prompt_graph/tasker/__init__.py",
        PROJECT_ROOT / "prompt_graph/tasker/task.py",
        PROJECT_ROOT / "prompt_graph/tasker/node_task.py",
        PROJECT_ROOT / "prompt_graph/prompt/__init__.py",
        PROJECT_ROOT / "prompt_graph/prompt/RobustPrompt_T.py",
        PROJECT_ROOT / "prompt_graph/prompt/RobustPrompt_T_original.py",
        PROJECT_ROOT / "prompt_graph/prompt/GPPTPrompt.py",
        PROJECT_ROOT / "prompt_graph/filters/__init__.py",
        PROJECT_ROOT / "prompt_graph/filters/filter_factory.py",
        PROJECT_ROOT / "prompt_graph/filters/neighbor_similarity_filter.py",
        PROJECT_ROOT / "prompt_graph/filters/nsp_filter.py",
        PROJECT_ROOT / "prompt_graph/filters/focusedcleaner_lp_filter.py",
        PROJECT_ROOT / "prompt_graph/evaluation/GPPTEva.py",
        PROJECT_ROOT / "prompt_graph/evaluation/RobustPromptTranductiveEva.py",
        PROJECT_ROOT / "prompt_graph/evaluation/__init__.py",
        PROJECT_ROOT / "prompt_graph/model/GCN.py",
        PROJECT_ROOT / "prompt_graph/model/__init__.py",
        PROJECT_ROOT / "prompt_graph/utils/get_args.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def git_tracked_status() -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.splitlines()


def replay_selected_clean_receipts(
    selected: list[dict[str, object]],
    clean_svd_data: object,
    device: object,
) -> list[dict[str, object]]:
    from eval_pretrain import evaluate_checkpoint

    replay_rows = []
    for row in selected:
        val_accuracy, test_accuracy = evaluate_checkpoint(
            str(Path(row["path"]).resolve()), clean_svd_data, device
        )
        observed_val = f"{float(val_accuracy):.6f}"
        observed_test = f"{float(test_accuracy):.6f}"
        expected_val = EXPECTED_CLEAN_REPLAY[str(row["id"])]["val_accuracy"]
        expected_test = EXPECTED_CLEAN_REPLAY[str(row["id"])]["test_accuracy"]
        if observed_val != expected_val or observed_test != expected_test:
            raise RuntimeError(
                "The active Cora SVD100 cache does not replay the frozen clean receipt for "
                f"backbone {row['id']}: expected val/test="
                f"{expected_val}/{expected_test}, observed={observed_val}/{observed_test}."
            )
        replay_rows.append(
            {
                "backbone_id": row["id"],
                "pretrain_seed": row["pretrain_seed"],
                "expected_val_accuracy": expected_val,
                "observed_val_accuracy": observed_val,
                "expected_test_accuracy": expected_test,
                "observed_test_accuracy": observed_test,
                "status": "exact_6dp_match",
            }
        )
    print("Target SVD clean replay verified | checkpoints=2 | metrics=4/4 exact_6dp_match")
    return replay_rows


def full_preflight(
    args: argparse.Namespace,
    selected: list[dict[str, object]],
) -> dict[str, object]:
    import numpy
    import scipy
    import sklearn
    import torch
    import torch_geometric

    from eval_citeseer_svd100_to_cora_svd100 import load_cora_svd100, tensor_sha256
    from prompt_graph.data import load4cora_downstream_clean, load4node_attack_specified_raw
    from prompt_graph.utils.downstream_svd_cache import apply_cora_svd100_cache

    child_python = subprocess.run(
        [args.python_bin, "-c", "import os,sys; print(os.path.realpath(sys.executable))"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    controller_python = str(Path(sys.executable).resolve())
    if str(Path(child_python).resolve()) != controller_python:
        raise RuntimeError(
            "Controller and child interpreter differ: "
            f"controller={controller_python}, child={child_python}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start the formal GPU experiment.")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise RuntimeError(
            f"Invalid logical CUDA device {args.device}; visible count={torch.cuda.device_count()}."
        )
    torch.cuda.set_device(args.device)
    device_properties = torch.cuda.get_device_properties(args.device)
    runtime_environment = {
        "python_executable": controller_python,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "logical_device": args.device,
        "device_name": device_properties.name,
        "device_total_memory": device_properties.total_memory,
    }

    source_receipt = SOURCE_SVD_RECEIPT
    source_hashes = {str(row["source_svd_sha256"]) for row in selected}
    if source_hashes != {source_receipt.get("reduced_x_sha256")} or "" in source_hashes:
        raise RuntimeError("Selected checkpoint rows and source SVD receipt disagree.")
    if source_receipt.get("preprocessing") != "raw_bow_to_l1_to_independent_svd100":
        raise RuntimeError("Unexpected Citeseer SVD100 preprocessing receipt.")

    for row in selected:
        validate_checkpoint(Path(row["path"]), str(row["sha256"]))

    attack_root = PROJECT_ROOT / "data_attack_fewshot/Cora/shot_5/1/Meta_Self/raw"
    shared_file_hashes = {
        "feature_file_sha256": file_sha256(attack_root / "Cora_features.npz"),
        "label_file_sha256": file_sha256(attack_root / "Cora_labels.npy"),
    }
    expected_shared_hashes = {
        "feature_file_sha256": CANONICAL_FEATURE_SHA256,
        "label_file_sha256": CANONICAL_LABEL_SHA256,
    }
    if shared_file_hashes != expected_shared_hashes:
        raise RuntimeError(
            f"Canonical feature/label SHA256 mismatch: {shared_file_hashes}."
        )
    attack_manifest = []
    data_by_rate = {}
    for ptb_rate in PTB_RATES:
        graph_path = attack_root / f"Meta_Self_Cora_{ptb_rate}.pt"
        actual_hash = file_sha256(graph_path)
        expected = EXPECTED_ATTACKS[ptb_rate]
        if actual_hash != expected["sha256"]:
            raise RuntimeError(
                f"Canonical raw SHA256 mismatch for ptb={ptb_rate}: "
                f"expected {expected['sha256']}, got {actual_hash}."
            )
        index_hashes = {
            split_name: file_sha256(
                attack_root / f"Meta_Self_Cora_{ptb_rate}_idx_{split_name}.npy"
            )
            for split_name in ("train", "val", "test")
        }
        if index_hashes != CANONICAL_INDEX_SHA256:
            raise RuntimeError(
                f"Canonical split SHA256 mismatch for ptb={ptb_rate}: {index_hashes}."
            )
        data, _ = load4node_attack_specified_raw(
            "data_attack_fewshot", "Cora", f"Meta_Self-{ptb_rate}", shot_num=5, run_split=1
        )
        if int(data.num_edges) != expected["runtime_edges"]:
            raise RuntimeError(f"Runtime edge mismatch for ptb={ptb_rate}.")
        data_by_rate[ptb_rate] = data
        attack_manifest.append(
            {
                "ptb_rate": ptb_rate,
                "raw_path": str(graph_path.resolve()),
                "raw_sha256": actual_hash,
                "clean_edges": 5278,
                "added": expected["added"],
                "deleted": expected["deleted"],
                "modifications": expected["added"] + expected["deleted"],
                "realized_rate": f"{(expected['added'] + expected['deleted']) / 5278:.6f}",
                "attack_edges": expected["edges"],
                "runtime_edges": expected["runtime_edges"],
                "feature_file_sha256": shared_file_hashes["feature_file_sha256"],
                "label_file_sha256": shared_file_hashes["label_file_sha256"],
                "train_idx_sha256": index_hashes["train"],
                "val_idx_sha256": index_hashes["val"],
                "test_idx_sha256": index_hashes["test"],
            }
        )

    target_cache = Path(args.target_cache)
    _, target_cache_dataset, target_cache_prepare_action = load_cora_svd100(target_cache)
    if int(target_cache_dataset.num_classes) != 7:
        raise RuntimeError(
            f"Expected 7 Cora classes while preparing SVD100, got {target_cache_dataset.num_classes}."
        )
    print(
        "Target SVD100 cache prepared | "
        f"action={target_cache_prepare_action} | path={target_cache.resolve()}"
    )
    target_cache_file_hash = file_sha256(target_cache)
    target_receipt = apply_cora_svd100_cache(
        data_by_rate["0.00"], target_cache, "Cora", 100
    )
    if not bool(torch.isfinite(data_by_rate["0.00"].x).all()):
        raise RuntimeError("Validated Cora SVD100 contains non-finite values.")
    expected_cache_runtime = {
        "torch_version": str(torch.__version__),
        "torch_geometric_version": str(torch_geometric.__version__),
    }
    cache_runtime_mismatches = {
        field: (target_receipt.get(field), expected)
        for field, expected in expected_cache_runtime.items()
        if str(target_receipt.get(field)) != expected
    }
    if cache_runtime_mismatches:
        raise RuntimeError(
            "Cora SVD100 cache runtime receipt differs from the clean evaluator runtime; "
            f"refusing a replay that could recompute the cache: {cache_runtime_mismatches}."
        )

    clean_raw_data, _ = load4cora_downstream_clean("Cora", shot_num=5, run_split=1)
    clean_original_hash = tensor_sha256(clean_raw_data.x.detach().cpu().contiguous())
    if clean_original_hash != target_receipt["original_x_sha256"]:
        raise RuntimeError(
            "Clean evaluator Cora features no longer match the SVD100 cache fit input: "
            f"expected {target_receipt['original_x_sha256']}, got {clean_original_hash}."
        )
    clean_replay_data, clean_replay_dataset, cache_action = load_cora_svd100(target_cache)
    if cache_action != "loaded" or int(clean_replay_dataset.num_classes) != 7:
        raise RuntimeError(
            "Clean evaluator did not strictly reuse the existing Cora SVD100 cache: "
            f"cache_action={cache_action}, classes={clean_replay_dataset.num_classes}."
        )
    if file_sha256(target_cache) != target_cache_file_hash:
        raise RuntimeError("Cora SVD100 cache file changed during clean replay preflight.")
    target_clean_replay = replay_selected_clean_receipts(
        selected,
        clean_replay_data,
        torch.device(f"cuda:{args.device}"),
    )
    if file_sha256(target_cache) != target_cache_file_hash:
        raise RuntimeError("Cora SVD100 cache file changed during clean metric replay.")

    return {
        "source_receipt": source_receipt,
        "target_receipt": target_receipt,
        "target_cache_file_sha256": target_cache_file_hash,
        "target_clean_replay": target_clean_replay,
        "attack_manifest": attack_manifest,
        "shared_file_hashes": shared_file_hashes,
        "runtime_environment": runtime_environment,
        "code_sha256": code_bundle_sha256(),
        "git_commit": git_commit(),
        "git_tracked_status": git_tracked_status(),
    }


def config_sha256(context: dict[str, object]) -> str:
    config = {
        "experiment": EXPERIMENT_NAME,
        "code_sha256": context["code_sha256"],
        "matrix": "2_fixed_backbones*3_methods*6_ptb*5_downstream_seeds",
        "methods": METHODS,
        "ptb_rates": PTB_RATES,
        "downstream_seeds": DOWNSTREAM_SEEDS,
        "backbones": BACKBONES,
        "pretrain": "Citeseer_LCC_L1_independent_SVD100_GraphCL",
        "encoder": "GCN_100_256_256_final_epoch_frozen",
        "target": "Cora_full_2708_L1_fixed_independent_SVD100",
        "target_svd_sha256": context["target_receipt"]["reduced_x_sha256"],
        "target_cache_file_sha256": context["target_cache_file_sha256"],
        "target_clean_replay": context["target_clean_replay"],
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "attack_sha256": {rate: EXPECTED_ATTACKS[rate]["sha256"] for rate in PTB_RATES},
        "prompt_epochs": 200,
        "robust_prompt": {
            "prompt_lr": 0.01,
            "pt_threshold": 0.5,
            "weight_mse": 0.1,
            "weight_kl": 0.3,
            "weight_constraint": 0.2,
            "temperature": 1.0,
            "sim_threshold": 0.2,
            "degree_threshold": 1,
            "out_detect_threshold": 0.4,
            "p_plus": True,
            "attention": False,
            "cosine_constraint": True,
        },
        "gppt": "repo_adaptation_frozen_graphcl_no_added_filter",
        "endpoint": "train_loss_early_stop_final_state_no_validation_selection",
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_or_verify(path: Path, content: str) -> None:
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise RuntimeError(
                f"Existing provenance file differs from this run: {path}. Use a new output directory."
            )
        return
    atomic_write_text(path, content)


def write_manifests(
    args: argparse.Namespace,
    selected: list[dict[str, object]],
    plan: list[dict[str, object]],
    context: dict[str, object],
    config_hash: str,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_receipt = context["target_receipt"]
    source_receipt = context["source_receipt"]
    manifest_rows = [
        ("experiment", EXPERIMENT_NAME),
        ("git_commit", context["git_commit"]),
        ("git_tracked_status_json", json.dumps(context["git_tracked_status"], ensure_ascii=False)),
        ("code_sha256", context["code_sha256"]),
        ("config_sha256", config_hash),
        ("runtime_environment_json", json.dumps(context["runtime_environment"], sort_keys=True)),
        ("peak_bb_selection", BACKBONES[0]["selection"]),
        ("stable_bb_selection", BACKBONES[1]["selection"]),
        (
            "matrix",
            "2 fixed backbones * 3 methods * 6 corrected-budget ptb levels * "
            "5 downstream seeds = 180 runs",
        ),
        (
            "seed_policy",
            "each fixed pretrain checkpoint is reused across downstream seeds 1..5; "
            "five downstream replicates per backbone-method-ptb group",
        ),
        ("source_pipeline", "Citeseer raw3703->L1->independent SVD100->GraphCL GCN(100->256->256)"),
        ("source_svd_sha256", source_receipt["reduced_x_sha256"]),
        (
            "target_pipeline",
            "Cora raw1433->L1->once-per-clone independent SVD100->fixed cache->"
            "frozen GCN(100->256->256)",
        ),
        ("target_svd_sha256", target_receipt["reduced_x_sha256"]),
        ("target_cache_policy", "compute_if_missing_then_strict_reuse"),
        ("target_cache_file_sha256", context["target_cache_file_sha256"]),
        (
            "target_cache_anchor",
            "original clean evaluator replayed on 2 selected checkpoints; "
            "4/4 frozen val/test metrics require exact 6-decimal match",
        ),
        ("feature_alignment", "independent_svd_shape_only; no shared basis or semantic alignment"),
        ("attack_data", "canonical specified Meta_Self raw; direct raw loader bypasses PyG processed caches"),
        ("feature_file_sha256", context["shared_file_hashes"]["feature_file_sha256"]),
        ("label_file_sha256", context["shared_file_hashes"]["label_file_sha256"]),
        ("train_idx_sha256", CANONICAL_INDEX_SHA256["train"]),
        ("val_idx_sha256", CANONICAL_INDEX_SHA256["val"]),
        ("test_idx_sha256", CANONICAL_INDEX_SHA256["test"]),
        ("pollution_rates", "/".join(PTB_RATES)),
        ("gpromptshield_original", METHODS[0]["description"]),
        ("gpromptshield_ours", METHODS[1]["description"]),
        ("gppt", METHODS[2]["description"]),
        (
            "rprompt_matched_settings",
            "from 2026-07-30 run_citeseer_svd1433_to_cora_rprompt_original.sh: "
            "prompt_lr=0.01,pt_threshold=0.5,weight_mse=0.1,weight_kl=0.3,"
            "weight_constraint=0.2,temperature=1.0,sim=0.2,degree=1,ood=0.4,"
            "p_plus=true,attention=false,cosine_constraint=true",
        ),
        ("threshold_policy", "fixed before SVD100 pollution scan; no polluted validation/test retuning"),
        (
            "backbone_policy",
            "Peak BB and Stable BB are fixed before pollution runs; each checkpoint is "
            "loaded strictly, requires_grad=false, and eval mode is verified in each child log",
        ),
        ("endpoint", "train-loss early stop; final state evaluated; no downstream validation selection"),
        (
            "reporting",
            "for each backbone-method-ptb group use all five downstream seeds; "
            "simple mean and population std; no seed removed",
        ),
        ("test_policy", "test is report-only; no test-oracle method or hyperparameter selection"),
    ]
    manifest_content = "field\tvalue\n" + "".join(f"{key}\t{value}\n" for key, value in manifest_rows)
    write_or_verify(output_dir / "manifest.tsv", manifest_content)

    checkpoint_fields = [
        "backbone_id", "backbone_label", "selection", "selection_group_val_mean",
        "selection_group_val_std", "pretrain_seed", "checkpoint", "checkpoint_sha256",
        "source_svd_sha256", "clean_val_accuracy", "clean_test_accuracy",
    ]
    checkpoint_lines = ["\t".join(checkpoint_fields)]
    for row in selected:
        checkpoint_lines.append(
            "\t".join(
                str(
                    row[
                        {
                            "backbone_id": "id",
                            "backbone_label": "label",
                            "pretrain_seed": "pretrain_seed",
                            "checkpoint": "path",
                            "checkpoint_sha256": "sha256",
                            "source_svd_sha256": "source_svd_sha256",
                            "clean_val_accuracy": "clean_val_accuracy",
                            "clean_test_accuracy": "clean_test_accuracy",
                            "selection": "selection",
                            "selection_group_val_mean": "selection_group_val_mean",
                            "selection_group_val_std": "selection_group_val_std",
                        }[field]
                    ]
                )
                for field in checkpoint_fields
            )
        )
    write_or_verify(output_dir / "selected_checkpoints.tsv", "\n".join(checkpoint_lines) + "\n")

    replay_fields = [
        "backbone_id", "pretrain_seed", "expected_val_accuracy", "observed_val_accuracy",
        "expected_test_accuracy", "observed_test_accuracy", "status",
    ]
    replay_lines = ["\t".join(replay_fields)]
    for row in context["target_clean_replay"]:
        replay_lines.append("\t".join(str(row[field]) for field in replay_fields))
    write_or_verify(
        output_dir / "target_svd_clean_replay.tsv", "\n".join(replay_lines) + "\n"
    )

    attack_fields = [
        "ptb_rate", "raw_path", "raw_sha256", "clean_edges", "added", "deleted",
        "modifications", "realized_rate", "attack_edges", "runtime_edges", "feature_file_sha256",
        "label_file_sha256", "train_idx_sha256", "val_idx_sha256", "test_idx_sha256",
    ]
    attack_lines = ["\t".join(attack_fields)]
    for row in context["attack_manifest"]:
        attack_lines.append("\t".join(str(row[field]) for field in attack_fields))
    write_or_verify(output_dir / "attack_graph_manifest.tsv", "\n".join(attack_lines) + "\n")

    plan_fields = [
        "run_key", "backbone_id", "method", "ptb_rate", "pretrain_seed",
        "downstream_seed", "checkpoint", "checkpoint_sha256",
    ]
    plan_lines = ["\t".join(plan_fields)]
    for run in plan:
        plan_lines.append(
            "\t".join(
                [
                    str(run["run_key"]),
                    str(run["checkpoint"]["id"]),
                    str(run["method"]["id"]),
                    str(run["ptb_rate"]),
                    str(run["pretrain_seed"]),
                    str(run["downstream_seed"]),
                    str(Path(run["checkpoint"]["path"]).resolve()),
                    str(run["checkpoint"]["sha256"]),
                ]
            )
        )
    write_or_verify(output_dir / "plan.tsv", "\n".join(plan_lines) + "\n")


def summary_rows(results: list[dict[str, str]]) -> list[dict[str, object]]:
    summaries = []
    for backbone in BACKBONES:
        for method in METHODS:
            for ptb_rate in PTB_RATES:
                rows = [
                    row for row in results
                    if row["status"] == "ok"
                    and row["backbone_id"] == backbone["id"]
                    and row["method"] == method["id"]
                    and row["ptb_rate"] == ptb_rate
                ]
                seeds_ok = sorted(int(row["downstream_seed"]) for row in rows)
                if len(seeds_ok) != len(set(seeds_ok)):
                    raise RuntimeError(
                        f"Duplicate downstream seed for {backbone['id']}/{method['id']} "
                        f"ptb={ptb_rate}."
                    )
                missing = sorted(set(DOWNSTREAM_SEEDS) - set(seeds_ok))
                accuracies = [float(row["test_accuracy"]) for row in rows]
                f1_values = [float(row["test_macro_f1"]) for row in rows]
                complete = seeds_ok == list(DOWNSTREAM_SEEDS)
                summaries.append(
                    {
                        "backbone_id": backbone["id"],
                        "backbone_label": backbone["label"],
                        "method": method["id"],
                        "ptb_rate": ptb_rate,
                        "selection_status": "complete_5_of_5" if complete else "incomplete",
                        "n_seeds": len(rows),
                        "seeds_ok": ";".join(map(str, seeds_ok)),
                        "seeds_missing": ";".join(map(str, missing)),
                        "test_accuracy_mean": f"{statistics.mean(accuracies):.6f}" if rows else "",
                        "test_accuracy_std": f"{statistics.pstdev(accuracies):.6f}" if rows else "",
                        "test_macro_f1_mean": f"{statistics.mean(f1_values):.6f}" if rows else "",
                        "test_macro_f1_std": f"{statistics.pstdev(f1_values):.6f}" if rows else "",
                    }
                )
    return summaries


def sort_results(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    backbone_order = {backbone["id"]: index for index, backbone in enumerate(BACKBONES)}
    method_order = {method["id"]: index for index, method in enumerate(METHODS)}
    ptb_order = {rate: index for index, rate in enumerate(PTB_RATES)}
    return sorted(
        rows,
        key=lambda row: (
            backbone_order.get(row["backbone_id"], 999),
            method_order.get(row["method"], 999),
            ptb_order.get(row["ptb_rate"], 999),
            int(row["downstream_seed"]),
        ),
    )


def upsert_result(path: Path, row: dict[str, object]) -> list[dict[str, str]]:
    existing = read_results(path)
    normalized = {field: str(row.get(field, "")) for field in RESULT_FIELDS}
    replaced = False
    for index, old_row in enumerate(existing):
        if old_row["run_key"] == normalized["run_key"]:
            existing[index] = normalized
            replaced = True
            break
    if not replaced:
        existing.append(normalized)
    existing = sort_results(existing)
    atomic_write_csv(path, RESULT_FIELDS, existing)
    return existing


def build_command(args: argparse.Namespace, run: dict[str, object]) -> list[str]:
    method = run["method"]
    checkpoint = run["checkpoint"]
    command = [
        args.python_bin,
        "MyTask.py",
        "--pre_train_model_path", str(Path(checkpoint["path"]).resolve()),
        "--pretrain_dataset_name", "Citeseer",
        "--task", "NodeTask",
        "--dataset_name", "Cora",
        "--preprocess_method", "svd",
        "--svd_out_dim", "100",
        "--downstream_svd_cache", str(Path(args.target_cache).resolve()),
        "--gnn_type", "GCN",
        "--prompt_type", str(method["prompt_type"]),
        "--shot_num", "5",
        "--run_split", "1",
        "--hid_dim", "256",
        "--num_layer", "2",
        "--epochs", "200",
        "--seed", str(run["downstream_seed"]),
        "--device", str(args.device),
        "--filter_mode", str(method["filter_mode"]),
        "--attack_downstream",
        "--specified",
        "--strict_attack_raw",
        "--attack_method", f"Meta_Self-{run['ptb_rate']}",
    ]
    if method["prompt_type"] == "RobustPrompt-T":
        command.extend(
            [
                "--prompt_variant", str(method["prompt_variant"]),
                "--prompt_lr", "0.01",
                "--pt_threshold", "0.5",
                "--weight_mse", "0.1",
                "--weight_kl", "0.3",
                "--weight_constraint", "0.2",
                "--temperature", "1.0",
                "--pt_sim_threshold", "0.2",
                "--pt_degree_threshold", "1",
                "--pt_out_detect_threshold", "0.4",
                "--p_plus",
                "--no_attention",
                "--cosine_constraint",
            ]
        )
    return command


def run_one(
    args: argparse.Namespace,
    run: dict[str, object],
    context: dict[str, object],
    config_hash: str,
) -> dict[str, object]:
    method = run["method"]
    ptb_rate = str(run["ptb_rate"])
    downstream_seed = int(run["downstream_seed"])
    backbone_id = str(run["checkpoint"]["id"])
    expected_attack = EXPECTED_ATTACKS[ptb_rate]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = (
        Path(args.output_dir)
        / "run_logs"
        / backbone_id
        / str(method["id"])
        / f"ptb_{ptb_rate}"
        / f"downstream_seed_{downstream_seed}.attempt_{timestamp}_{os.getpid()}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(args, run)
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(downstream_seed)
    environment["PYTHONUNBUFFERED"] = "1"
    final_results = []
    strict_attack_seen = False
    cache_seen = False
    frozen_backbone_seen = False

    try:
        with log_path.open("w", encoding="utf-8", newline="") as log_handle:
            log_handle.write(f"RUN_KEY\t{run['run_key']}\n")
            log_handle.write(f"COMMAND_JSON\t{json.dumps(command)}\n")
            log_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
                match = FINAL_RESULT_RE.fullmatch(line.rstrip("\r\n"))
                if match:
                    final_results.append(match.groupdict())
                if (
                    "Strict attack raw verified" in line
                    and f"attack=Meta_Self-{ptb_rate}" in line
                    and f"added={expected_attack['added']}" in line
                    and f"deleted={expected_attack['deleted']}" in line
                    and f"runtime_edges={expected_attack['runtime_edges']}" in line
                    and f"raw_sha256={expected_attack['sha256']}" in line
                ):
                    strict_attack_seen = True
                if (
                    "Downstream feature cache verified" in line
                    and f"reduced_sha256={context['target_receipt']['reduced_x_sha256']}" in line
                ):
                    cache_seen = True
                if (
                    "Frozen backbone verified" in line
                    and f"prompt_type={method['prompt_type']}" in line
                    and "trainable_parameters=0" in line
                    and "mode=eval" in line
                ):
                    frozen_backbone_seen = True
            return_code = process.wait()
    except Exception as error:
        raise RunFailure(
            f"Failed while executing MyTask; see {log_path}: {error}", log_path
        ) from error

    if return_code != 0:
        raise RunFailure(f"MyTask exited with code {return_code}; see {log_path}", log_path)
    if len(final_results) != 1:
        raise RunFailure(
            f"Expected one FINAL_RESULT line, found {len(final_results)} in {log_path}",
            log_path,
        )
    if not strict_attack_seen or not cache_seen or not frozen_backbone_seen:
        raise RunFailure(
            "Run log lacks required provenance receipts: "
            f"strict={strict_attack_seen}, cache={cache_seen}, "
            f"frozen_backbone={frozen_backbone_seen}; see {log_path}",
            log_path,
        )
    accuracy = float(final_results[0]["accuracy"])
    macro_f1 = float(final_results[0]["macro_f1"])
    if not math.isfinite(accuracy) or not math.isfinite(macro_f1):
        raise RunFailure(f"Non-finite metrics in {log_path}", log_path)
    if not 0.0 <= accuracy <= 1.0 or not 0.0 <= macro_f1 <= 1.0:
        raise RunFailure(f"Out-of-range metrics in {log_path}", log_path)
    return {
        "test_accuracy": f"{accuracy:.10f}",
        "test_macro_f1": f"{macro_f1:.10f}",
        "log_path": str(log_path.resolve()),
        "log_sha256": file_sha256(log_path),
    }


def base_result_row(
    args: argparse.Namespace,
    run: dict[str, object],
    context: dict[str, object],
    config_hash: str,
) -> dict[str, object]:
    method = run["method"]
    checkpoint = run["checkpoint"]
    ptb_rate = str(run["ptb_rate"])
    attack = EXPECTED_ATTACKS[ptb_rate]
    target_receipt = context["target_receipt"]
    return {
        "run_key": run["run_key"],
        "status": "failed",
        "backbone_id": checkpoint["id"],
        "backbone_label": checkpoint["label"],
        "backbone_selection": checkpoint["selection"],
        "method": method["id"],
        "ptb_rate": ptb_rate,
        "pretrain_seed": run["pretrain_seed"],
        "downstream_seed": run["downstream_seed"],
        "source_dataset": "Citeseer_Nettack_LCC",
        "target_dataset": "Cora_full_2708_corrected_Meta_Self",
        "feature_alignment": "independent_svd_shape_only",
        "target_preprocessing": "raw1433->L1->fixed_independent_SVD100",
        "encoder_dimensions": "100->256->256",
        "prompt_space": method["prompt_space"],
        "prompt_variant": method["prompt_variant"],
        "filter_mode_argument": method["filter_mode"],
        "effective_edge_path": method["effective_edge_path"],
        "epochs": 200,
        "shot": 5,
        "split": 1,
        "train_n": 35,
        "val_n": 265,
        "test_n": 2408,
        "attack_added": attack["added"],
        "attack_deleted": attack["deleted"],
        "attack_modifications": attack["added"] + attack["deleted"],
        "attack_realized_rate": f"{(attack['added'] + attack['deleted']) / 5278:.6f}",
        "attack_edges": attack["edges"],
        "runtime_edges": attack["runtime_edges"],
        "attack_graph_sha256": attack["sha256"],
        "feature_file_sha256": CANONICAL_FEATURE_SHA256,
        "label_file_sha256": CANONICAL_LABEL_SHA256,
        "train_idx_sha256": CANONICAL_INDEX_SHA256["train"],
        "val_idx_sha256": CANONICAL_INDEX_SHA256["val"],
        "test_idx_sha256": CANONICAL_INDEX_SHA256["test"],
        "target_svd_sha256": target_receipt["reduced_x_sha256"],
        "target_cache_file_sha256": context["target_cache_file_sha256"],
        "target_original_x_sha256": target_receipt["original_x_sha256"],
        "source_svd_sha256": checkpoint["source_svd_sha256"],
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint": str(Path(checkpoint["path"]).resolve()),
        "test_accuracy": "",
        "test_macro_f1": "",
        "code_sha256": context["code_sha256"],
        "config_sha256": config_hash,
        "log_path": "",
        "log_sha256": "",
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "error": "",
    }


def safe_to_skip(
    existing: dict[str, str],
    expected: dict[str, object],
) -> bool:
    if existing.get("status") != "ok":
        return False
    guarded_fields = [
        "backbone_id", "backbone_label", "backbone_selection", "method", "ptb_rate",
        "pretrain_seed", "downstream_seed", "attack_graph_sha256",
        "feature_file_sha256", "label_file_sha256", "train_idx_sha256", "val_idx_sha256",
        "test_idx_sha256", "target_svd_sha256", "target_original_x_sha256",
        "target_cache_file_sha256", "source_svd_sha256", "checkpoint_sha256",
        "code_sha256", "config_sha256",
        "filter_mode_argument", "effective_edge_path",
    ]
    mismatches = [
        field for field in guarded_fields
        if existing.get(field, "") != str(expected.get(field, ""))
    ]
    if mismatches:
        raise RuntimeError(
            f"Recorded successful run {existing['run_key']} has changed provenance fields: {mismatches}. "
            "Use a new output directory."
        )
    log_path = Path(existing.get("log_path", ""))
    if not log_path.is_file() or log_path.stat().st_size == 0:
        raise RuntimeError(f"Recorded successful run has no non-empty log receipt: {log_path}")
    recorded_log_hash = existing.get("log_sha256", "")
    actual_log_hash = file_sha256(log_path)
    if len(recorded_log_hash) != 64 or actual_log_hash != recorded_log_hash:
        raise RuntimeError(
            f"Successful log SHA256 mismatch for {existing['run_key']}: "
            f"recorded={recorded_log_hash}, actual={actual_log_hash}."
        )

    method = next(item for item in METHODS if item["id"] == existing["method"])
    attack = EXPECTED_ATTACKS[existing["ptb_rate"]]
    run_key_count = 0
    final_results = []
    strict_attack_seen = False
    cache_seen = False
    frozen_backbone_seen = False
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.rstrip("\r\n")
            if stripped == f"RUN_KEY\t{existing['run_key']}":
                run_key_count += 1
            match = FINAL_RESULT_RE.fullmatch(stripped)
            if match:
                final_results.append(match.groupdict())
            if (
                "Strict attack raw verified" in line
                and f"attack=Meta_Self-{existing['ptb_rate']}" in line
                and f"added={attack['added']}" in line
                and f"deleted={attack['deleted']}" in line
                and f"runtime_edges={attack['runtime_edges']}" in line
                and f"raw_sha256={attack['sha256']}" in line
            ):
                strict_attack_seen = True
            if (
                "Downstream feature cache verified" in line
                and f"reduced_sha256={existing['target_svd_sha256']}" in line
            ):
                cache_seen = True
            if (
                "Frozen backbone verified" in line
                and f"prompt_type={method['prompt_type']}" in line
                and "trainable_parameters=0" in line
                and "mode=eval" in line
            ):
                frozen_backbone_seen = True

    if run_key_count != 1 or len(final_results) != 1:
        raise RuntimeError(
            f"Successful log structure mismatch for {existing['run_key']}: "
            f"run_key_count={run_key_count}, final_result_count={len(final_results)}."
        )
    parsed_accuracy = f"{float(final_results[0]['accuracy']):.10f}"
    parsed_macro_f1 = f"{float(final_results[0]['macro_f1']):.10f}"
    if (
        parsed_accuracy != existing.get("test_accuracy")
        or parsed_macro_f1 != existing.get("test_macro_f1")
    ):
        raise RuntimeError(
            f"Successful log metrics disagree with CSV for {existing['run_key']}."
        )
    if not strict_attack_seen or not cache_seen or not frozen_backbone_seen:
        raise RuntimeError(
            f"Successful log provenance receipt mismatch for {existing['run_key']}: "
            f"strict={strict_attack_seen}, cache={cache_seen}, "
            f"frozen_backbone={frozen_backbone_seen}."
        )
    return True


def run_experiment(
    args: argparse.Namespace,
    selected: list[dict[str, object]],
    plan: list[dict[str, object]],
) -> int:
    context = full_preflight(args, selected)
    config_hash = config_sha256(context)
    write_manifests(args, selected, plan, context, config_hash)
    results_path = Path(args.output_dir) / "per_seed_results_incremental.csv"
    summary_path = Path(args.output_dir) / "summary_incremental.csv"
    existing_results = read_results(results_path)
    atomic_write_csv(summary_path, SUMMARY_FIELDS, summary_rows(existing_results))

    print(
        "Preflight passed | corrected canonical raw verified | fixed Cora SVD100 verified | "
        "clean replay=4/4 exact | 2 selected checkpoints verified | plan=180"
    )
    if args.preflight_only:
        print(f"Preflight-only complete: {Path(args.output_dir).resolve()}")
        return 0

    existing_by_key = {row["run_key"]: row for row in existing_results}
    for index, run in enumerate(plan, start=1):
        row = base_result_row(args, run, context, config_hash)
        old_row = existing_by_key.get(str(run["run_key"]))
        if old_row is not None and safe_to_skip(old_row, row):
            print(f"[{index}/{FORMAL_RUN_COUNT}] verified skip: {run['run_key']}")
            continue

        print(f"[{index}/{FORMAL_RUN_COUNT}] start: {run['run_key']}")
        try:
            metrics = run_one(args, run, context, config_hash)
            row.update(metrics)
            row["status"] = "ok"
            row["recorded_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            results = upsert_result(results_path, row)
            atomic_write_csv(summary_path, SUMMARY_FIELDS, summary_rows(results))
            existing_by_key[str(run["run_key"])] = {
                field: str(row.get(field, "")) for field in RESULT_FIELDS
            }
            print(
                f"[{index}/{FORMAL_RUN_COUNT}] recorded: {run['run_key']} | "
                f"test={row['test_accuracy']} | macro_f1={row['test_macro_f1']}"
            )
        except Exception as error:
            if isinstance(error, RunFailure):
                row["log_path"] = str(error.log_path.resolve())
                if error.log_path.is_file():
                    row["log_sha256"] = file_sha256(error.log_path)
            row["error"] = " ".join(str(error).splitlines())
            row["recorded_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            results = upsert_result(results_path, row)
            atomic_write_csv(summary_path, SUMMARY_FIELDS, summary_rows(results))
            print(f"FAILED: {run['run_key']} | {row['error']}", file=sys.stderr)
            print("Stop-after-first-failure is intentional; fix the cause and rerun the same command.", file=sys.stderr)
            return 1

    results = read_results(results_path)
    successful = [row for row in results if row["status"] == "ok"]
    summaries = summary_rows(results)
    complete_groups = [row for row in summaries if row["selection_status"] == "complete_5_of_5"]
    if len(successful) != FORMAL_RUN_COUNT or len(complete_groups) != FORMAL_GROUP_COUNT:
        print(
            f"Incomplete experiment: successful={len(successful)}/{FORMAL_RUN_COUNT}, "
            f"complete_groups={len(complete_groups)}/{FORMAL_GROUP_COUNT}. "
            "Rerun the same command.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Completed: {FORMAL_RUN_COUNT}/{FORMAL_RUN_COUNT} runs and "
        f"{FORMAL_GROUP_COUNT}/{FORMAL_GROUP_COUNT} backbone-by-method-by-ptb groups | "
        f"{Path(args.output_dir).resolve()}"
    )
    return 0


def acquire_output_lock(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".controller.lock"
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        sort_keys=True,
    )
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        existing = lock_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            f"Output directory is already locked: {lock_path} ({existing}). "
            "If no controller is running, inspect and remove only this stale lock file."
        ) from error
    try:
        os.write(descriptor, payload.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return lock_path


def release_output_lock(lock_path: Path) -> None:
    if lock_path.is_file():
        lock_path.unlink()


def main() -> int:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    selected = selected_checkpoint_rows(args)
    plan = build_plan(selected)

    if args.plan_only:
        print(
            "Formal plan: 2 fixed backbones * 3 methods * 6 ptb levels * "
            "5 downstream seeds = 180 runs"
        )
        for run in plan:
            print(run["run_key"])
        return 0

    lock_path = acquire_output_lock(Path(args.output_dir))
    try:
        return run_experiment(args, selected, plan)
    finally:
        release_output_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
