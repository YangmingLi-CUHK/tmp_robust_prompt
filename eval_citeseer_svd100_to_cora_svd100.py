#!/usr/bin/env python3
"""Evaluate one Citeseer-SVD100 GraphCL checkpoint on clean Cora-SVD100.

This is a deliberately narrow evaluator for the 135-run transfer experiment. It
keeps the existing linear-probe protocol, but prevents the old evaluator from
silently using Cora's 1433-dimensional features or selecting by test accuracy.
Each invocation atomically updates a per-seed result table and its 5-seed group
summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


EXPERIMENT_NAME = "citeseer_svd100_to_cora_svd100_graphcl_linear_probe"
FEATURE_ALIGNMENT = "independent_svd_shape_only"
EXPECTED_SEEDS = {1, 2, 3, 4, 5}
AUGMENTATIONS = {"dropN", "permE", "maskN"}
AUGMENTATION_ORDER = {name: index for index, name in enumerate(("dropN", "permE", "maskN"))}

CHECKPOINT_PATTERN = re.compile(
    r"^Citeseer\.GraphCL\.GCN\."
    r"(?P<hid_dim>\d+)_hidden_dim\."
    r"preprocess_svd_(?P<svd_dim>\d+)\."
    r"aug1_(?P<aug1>dropN|permE|maskN)\."
    r"aug2_(?P<aug2>dropN|permE|maskN)\."
    r"lr_(?P<lr>\d+(?:\.\d+)?)\."
    r"ratio_(?P<aug_ratio>\d+(?:\.\d+)?)\."
    r"seed_(?P<seed>\d+)\.pth$"
)

RESULT_FIELDS = [
    "run_key",
    "status",
    "aug1",
    "aug2",
    "aug_ratio",
    "pretrain_lr",
    "pretrain_seed",
    "source_dataset",
    "target_dataset",
    "source_original_dim",
    "target_original_dim",
    "svd_dim",
    "feature_alignment",
    "source_svd_sha256",
    "encoder_dimensions",
    "target_nodes",
    "target_edges",
    "shot",
    "split",
    "train_n",
    "val_n",
    "test_n",
    "val_accuracy",
    "test_accuracy",
    "checkpoint_sha256",
    "checkpoint",
    "log_path",
    "recorded_utc",
    "error",
]

SUMMARY_FIELDS = [
    "rank_by_val",
    "selection_status",
    "aug1",
    "aug2",
    "aug_ratio",
    "pretrain_lr",
    "n_seeds",
    "seeds_ok",
    "seeds_missing",
    "val_mean",
    "val_std",
    "test_mean",
    "test_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen Citeseer-SVD100 GraphCL encoder + logistic-regression "
            "evaluation on independently fitted Cora-SVD100 features."
        )
    )
    parser.add_argument("--checkpoint", help="One Citeseer SVD100 checkpoint.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument(
        "--cache_path",
        default="data/preprocessed/cora_clean_full_l1_svd_100.pt",
        help="Cora SVD100 feature cache; content fingerprint is validated before reuse.",
    )
    parser.add_argument(
        "--results_csv",
        default=(
            "logs/citeseer_svd100_to_cora_svd100_graphcl_135/"
            "per_seed_results_incremental.csv"
        ),
    )
    parser.add_argument(
        "--summary_csv",
        default=(
            "logs/citeseer_svd100_to_cora_svd100_graphcl_135/"
            "group_summary_incremental.csv"
        ),
    )
    parser.add_argument("--log_path", default="")
    parser.add_argument(
        "--source_cache_path",
        default="data/deeprobust/citeseer_nettack_lcc_l1_svd_100.pt",
    )
    parser.add_argument(
        "--source_cache_receipt",
        default=(
            "logs/citeseer_svd100_to_cora_svd100_graphcl_135/"
            "citeseer_svd100_cache_receipt.json"
        ),
    )
    parser.add_argument(
        "--prepare_source_cache_only",
        action="store_true",
        help="Prepare Citeseer SVD100 in a separate seeded process, then exit.",
    )
    parser.add_argument(
        "--validate_checkpoint_only",
        action="store_true",
        help="Strictly load the 100->256->256 GCN checkpoint, then exit without data evaluation.",
    )
    parser.add_argument(
        "--skip_recorded",
        action="store_true",
        help="Return successfully if this exact run already has status=ok.",
    )
    args = parser.parse_args()

    if (args.shot, args.split) != (5, 1):
        parser.error("This dedicated experiment fixes the downstream split at 5-shot/split-1.")
    if not args.prepare_source_cache_only and not args.checkpoint:
        parser.error("--checkpoint is required unless --prepare_source_cache_only is used.")
    return args


def parse_checkpoint(path: Path) -> dict[str, object]:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(
            "Checkpoint name does not match the dedicated Citeseer-SVD100 GraphCL format: "
            f"{path.name}"
        )

    info: dict[str, object] = match.groupdict()
    info["hid_dim"] = int(str(info["hid_dim"]))
    info["svd_dim"] = int(str(info["svd_dim"]))
    info["seed"] = int(str(info["seed"]))

    if info["hid_dim"] != 256:
        raise ValueError(f"Expected hid_dim=256, got {info['hid_dim']}.")
    if info["svd_dim"] != 100:
        raise ValueError(f"Expected SVD dimension 100, got {info['svd_dim']}.")
    if info["seed"] not in EXPECTED_SEEDS:
        raise ValueError(f"Expected pretraining seed in {sorted(EXPECTED_SEEDS)}, got {info['seed']}.")
    if info["aug1"] not in AUGMENTATIONS or info["aug2"] not in AUGMENTATIONS:
        raise ValueError("Unexpected GraphCL augmentation.")
    if Decimal(str(info["lr"])) != Decimal("0.001"):
        raise ValueError(f"Expected pretraining lr=0.001, got {info['lr']}.")
    if Decimal(str(info["aug_ratio"])) not in {
        Decimal("0.1"),
        Decimal("0.2"),
        Decimal("0.3"),
    }:
        raise ValueError(f"Unexpected augmentation ratio: {info['aug_ratio']}.")

    info["run_key"] = (
        f"a1={info['aug1']}|a2={info['aug2']}|"
        f"r={info['aug_ratio']}|s={info['seed']}"
    )
    return info


def read_csv(path: Path, expected_fields: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fields:
            raise RuntimeError(
                f"Unexpected CSV schema in {path}. Expected {expected_fields}, "
                f"got {reader.fieldnames}."
            )
        rows = list(reader)

    keys = [row["run_key"] for row in rows] if expected_fields == RESULT_FIELDS else []
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate run_key values found in {path}.")
    return rows


def atomic_write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def result_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        int(str(row["pretrain_seed"])),
        Decimal(str(row["aug_ratio"])),
        AUGMENTATION_ORDER[str(row["aug1"])],
        AUGMENTATION_ORDER[str(row["aug2"])],
    )


def successful_result(results_path: Path, run_key: str) -> tuple[dict[str, str] | None, list[dict[str, str]]]:
    rows = read_csv(results_path, RESULT_FIELDS)
    matching = [row for row in rows if row["run_key"] == run_key and row["status"] == "ok"]
    if len(matching) > 1:
        raise RuntimeError(f"Duplicate successful result for run_key {run_key}.")
    return (matching[0] if matching else None), rows


def upsert_result(results_path: Path, new_row: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = list(read_csv(results_path, RESULT_FIELDS))
    matching = [row for row in rows if row["run_key"] == new_row["run_key"]]
    if len(matching) > 1:
        raise RuntimeError(f"Duplicate run_key {new_row['run_key']} in {results_path}.")

    if matching:
        old_row = matching[0]
        if old_row["status"] == "ok" and new_row["status"] != "ok":
            raise RuntimeError(
                f"Refusing to replace successful result {new_row['run_key']} with a failed result."
            )
        rows = [row for row in rows if row["run_key"] != new_row["run_key"]]
    rows.append(new_row)
    rows.sort(key=result_sort_key)
    atomic_write_csv(results_path, RESULT_FIELDS, rows)
    return rows


def _mean_std(values: list[float]) -> tuple[str, str]:
    if not values:
        return "", ""
    return f"{statistics.fmean(values):.6f}", f"{statistics.pstdev(values):.6f}"


def build_group_summary(result_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    group_keys = {
        (
            str(row["aug1"]),
            str(row["aug2"]),
            str(row["aug_ratio"]),
            str(row["pretrain_lr"]),
        )
        for row in result_rows
    }
    summaries: list[dict[str, object]] = []

    for aug1, aug2, aug_ratio, pretrain_lr in group_keys:
        successful = [
            row
            for row in result_rows
            if row["status"] == "ok"
            and (
                str(row["aug1"]),
                str(row["aug2"]),
                str(row["aug_ratio"]),
                str(row["pretrain_lr"]),
            )
            == (aug1, aug2, aug_ratio, pretrain_lr)
        ]
        seeds = {int(str(row["pretrain_seed"])) for row in successful}
        if len(seeds) != len(successful):
            raise RuntimeError(
                f"Duplicate successful seeds in group {(aug1, aug2, aug_ratio, pretrain_lr)}."
            )

        val_values = [float(str(row["val_accuracy"])) for row in successful]
        test_values = [float(str(row["test_accuracy"])) for row in successful]
        if not all(math.isfinite(value) for value in val_values + test_values):
            raise RuntimeError("Non-finite accuracy found in successful result rows.")

        val_mean, val_std = _mean_std(val_values)
        test_mean, test_std = _mean_std(test_values)
        complete = seeds == EXPECTED_SEEDS
        summaries.append(
            {
                "rank_by_val": "",
                "selection_status": "complete_5_of_5" if complete else "partial_not_selectable",
                "aug1": aug1,
                "aug2": aug2,
                "aug_ratio": aug_ratio,
                "pretrain_lr": pretrain_lr,
                "n_seeds": len(seeds),
                "seeds_ok": ";".join(str(seed) for seed in sorted(seeds)),
                "seeds_missing": ";".join(str(seed) for seed in sorted(EXPECTED_SEEDS - seeds)),
                "val_mean": val_mean,
                "val_std": val_std,
                "test_mean": test_mean,
                "test_std": test_std,
            }
        )

    complete_rows = [row for row in summaries if row["selection_status"] == "complete_5_of_5"]
    partial_rows = [row for row in summaries if row["selection_status"] != "complete_5_of_5"]
    complete_rows.sort(
        key=lambda row: (
            -float(str(row["val_mean"])),
            AUGMENTATION_ORDER[str(row["aug1"])],
            AUGMENTATION_ORDER[str(row["aug2"])],
            Decimal(str(row["aug_ratio"])),
        )
    )
    partial_rows.sort(
        key=lambda row: (
            -int(str(row["n_seeds"])),
            AUGMENTATION_ORDER[str(row["aug1"])],
            AUGMENTATION_ORDER[str(row["aug2"])],
            Decimal(str(row["aug_ratio"])),
        )
    )
    for rank, row in enumerate(complete_rows, start=1):
        row["rank_by_val"] = rank
    return complete_rows + partial_rows


def update_group_summary(summary_path: Path, result_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries = build_group_summary(result_rows)
    atomic_write_csv(summary_path, SUMMARY_FIELDS, summaries)
    return summaries


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_architecture(path: Path) -> str:
    import torch

    from prompt_graph.model import GCN

    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Checkpoint is missing or empty: {path}")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError("Checkpoint is not a non-empty GNN state_dict.")
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"Checkpoint value {key!r} is not a tensor.")
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"Checkpoint tensor {key!r} contains non-finite values.")

    gnn = GCN(input_dim=100, hid_dim=256, num_layer=2)
    gnn.load_state_dict(state_dict, strict=True)
    return file_sha256(path)


def tensor_sha256(tensor: object) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def source_svd_hash_from_receipt(receipt_path: Path) -> str:
    receipt = _load_json(receipt_path)
    if receipt is None:
        raise RuntimeError(f"Missing or unreadable source SVD receipt: {receipt_path}")
    if receipt.get("preprocessing") != "raw_bow_to_l1_to_independent_svd100":
        raise RuntimeError(f"Unexpected source SVD receipt preprocessing in {receipt_path}.")
    reduced_hash = receipt.get("reduced_x_sha256")
    if not isinstance(reduced_hash, str) or len(reduced_hash) != 64:
        raise RuntimeError(f"Invalid source SVD hash in {receipt_path}.")
    return reduced_hash


def prepare_citeseer_svd100_cache(
    cache_path: Path,
    receipt_path: Path,
    results_path: Path,
) -> dict[str, object]:
    import torch
    import torch_geometric
    from torch_geometric.transforms import SVDFeatureReduction

    from prompt_graph.data.load4data import load4citeseer_pretrain

    data, original_dim, _ = load4citeseer_pretrain()
    original_x = data.x.detach().cpu().contiguous()
    if tuple(original_x.shape) != (2110, 3703) or original_dim != 3703:
        raise RuntimeError(
            f"Expected Citeseer LCC L1 features (2110, 3703), got {tuple(original_x.shape)}."
        )
    if not bool(torch.isfinite(original_x).all()):
        raise RuntimeError("Citeseer L1-normalized features contain non-finite values.")

    base_receipt: dict[str, object] = {
        "version": 1,
        "dataset": "Citeseer_Nettack_LCC",
        "preprocessing": "raw_bow_to_l1_to_independent_svd100",
        "original_shape": [2110, 3703],
        "original_x_sha256": tensor_sha256(original_x),
        "reduced_shape": [2110, 100],
        "svd_backend": "torch_geometric.SVDFeatureReduction(torch.linalg.svd)",
        "torch_version": str(torch.__version__),
        "torch_geometric_version": str(getattr(torch_geometric, "__version__", "unknown")),
    }
    old_receipt = _load_json(receipt_path)
    base_matches = old_receipt is not None and all(
        old_receipt.get(key) == value for key, value in base_receipt.items()
    )

    cached_x = None
    cached_hash = None
    if cache_path.is_file():
        try:
            candidate = torch.load(cache_path, map_location="cpu", weights_only=True)
            if (
                isinstance(candidate, torch.Tensor)
                and tuple(candidate.shape) == (2110, 100)
                and bool(torch.isfinite(candidate).all())
            ):
                cached_x = candidate.detach().cpu()
                cached_hash = tensor_sha256(cached_x)
        except Exception:
            pass

    if (
        base_matches
        and cached_x is not None
        and cached_hash == old_receipt.get("reduced_x_sha256")
    ):
        print(
            "Source SVD cache verified: "
            f"{cache_path} | sha256={cached_hash} | provenance fixed for all 135 runs"
        )
        return old_receipt

    reduced_data = SVDFeatureReduction(out_channels=100)(data.clone())
    reduced_x = reduced_data.x.detach().cpu()
    if tuple(reduced_x.shape) != (2110, 100):
        raise RuntimeError(f"Citeseer SVD produced unexpected shape {tuple(reduced_x.shape)}.")
    if not bool(torch.isfinite(reduced_x).all()):
        raise RuntimeError("Citeseer SVD100 features contain non-finite values.")
    reduced_hash = tensor_sha256(reduced_x)

    successful_rows = [
        row for row in read_csv(results_path, RESULT_FIELDS) if row["status"] == "ok"
    ]
    if successful_rows:
        expected_hash = old_receipt.get("reduced_x_sha256") if base_matches else None
        if expected_hash != reduced_hash:
            raise RuntimeError(
                "Source SVD provenance changed after successful results were recorded. "
                "Use a new OUTPUT_DIR; refusing to mix SVD bases."
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
    try:
        torch.save(reduced_x, temporary_cache)
        os.replace(temporary_cache, cache_path)
    finally:
        if temporary_cache.exists():
            temporary_cache.unlink()

    new_receipt = {
        **base_receipt,
        "reduced_x_sha256": reduced_hash,
        "cache_path": str(cache_path.resolve()),
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_json(receipt_path, new_receipt)
    action = "restored" if successful_rows else "prepared"
    print(
        f"Source SVD cache {action}: {cache_path} | sha256={reduced_hash} | "
        "provenance fixed for all 135 runs"
    )
    return new_receipt


def load_cora_svd100(cache_path: Path) -> tuple[object, object, str]:
    import torch
    import torch_geometric
    from torch_geometric.transforms import SVDFeatureReduction

    from prompt_graph.data import load4cora_downstream_clean

    data, dataset = load4cora_downstream_clean("Cora", shot_num=5, run_split=1)
    original_x = data.x.detach().cpu().contiguous()
    expected_original_shape = (2708, 1433)
    if tuple(original_x.shape) != expected_original_shape:
        raise RuntimeError(
            f"Expected full clean Cora features {expected_original_shape}, got {tuple(original_x.shape)}."
        )
    if not torch.isfinite(original_x).all():
        raise RuntimeError("Cora L1-normalized features contain non-finite values.")

    original_hash = tensor_sha256(original_x)
    expected_cache_metadata = {
        "version": 1,
        "dataset": "Cora",
        "data_scope": "full_clean_ptb_0.00",
        "preprocessing": "raw_bow_to_l1_to_independent_svd100",
        "original_shape": list(expected_original_shape),
        "original_x_sha256": original_hash,
        "svd_dim": 100,
        "svd_backend": "torch_geometric.SVDFeatureReduction(torch.linalg.svd)",
        "torch_version": str(torch.__version__),
        "torch_geometric_version": str(getattr(torch_geometric, "__version__", "unknown")),
    }

    reduced_x = None
    cache_action = "computed"
    if cache_path.is_file():
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            metadata_matches = isinstance(payload, dict) and all(
                payload.get(key) == value for key, value in expected_cache_metadata.items()
            )
            candidate = payload.get("x") if isinstance(payload, dict) else None
            candidate_hash = tensor_sha256(candidate) if isinstance(candidate, torch.Tensor) else None
            valid_candidate = (
                isinstance(candidate, torch.Tensor)
                and tuple(candidate.shape) == (2708, 100)
                and bool(torch.isfinite(candidate).all())
                and payload.get("reduced_x_sha256") == candidate_hash
            )
            if metadata_matches and valid_candidate:
                reduced_x = candidate.detach().cpu()
                cache_action = "loaded"
            else:
                cache_action = "recomputed_invalid_cache"
        except Exception:
            cache_action = "recomputed_unreadable_cache"

    if reduced_x is None:
        reduced_data = SVDFeatureReduction(out_channels=100)(data.clone())
        reduced_x = reduced_data.x.detach().cpu()
        if tuple(reduced_x.shape) != (2708, 100):
            raise RuntimeError(f"Cora SVD produced unexpected shape {tuple(reduced_x.shape)}.")
        if not torch.isfinite(reduced_x).all():
            raise RuntimeError("Cora SVD100 features contain non-finite values.")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_cache = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
        try:
            torch.save(
                {
                    **expected_cache_metadata,
                    "reduced_x_sha256": tensor_sha256(reduced_x),
                    "x": reduced_x,
                },
                temporary_cache,
            )
            os.replace(temporary_cache, cache_path)
        finally:
            if temporary_cache.exists():
                temporary_cache.unlink()

    data.x = reduced_x
    train_n = int(data.train_mask.sum().item())
    val_n = int(data.val_mask.sum().item())
    test_n = int(data.test_mask.sum().item())
    if (train_n, val_n, test_n) != (35, 265, 2408):
        raise RuntimeError(
            "Expected Cora 5-shot/split-1 masks 35/265/2408, got "
            f"{train_n}/{val_n}/{test_n}."
        )
    if train_n + val_n + test_n != data.num_nodes:
        raise RuntimeError("Cora train/val/test masks do not cover exactly all nodes.")
    if bool((data.train_mask & data.val_mask).any()) or bool((data.train_mask & data.test_mask).any()) or bool(
        (data.val_mask & data.test_mask).any()
    ):
        raise RuntimeError("Cora train/val/test masks overlap.")
    return data, dataset, cache_action


def make_base_row(info: dict[str, object], args: argparse.Namespace, checkpoint: Path) -> dict[str, object]:
    return {
        "run_key": info["run_key"],
        "status": "failed",
        "aug1": info["aug1"],
        "aug2": info["aug2"],
        "aug_ratio": info["aug_ratio"],
        "pretrain_lr": info["lr"],
        "pretrain_seed": info["seed"],
        "source_dataset": "Citeseer_Nettack_LCC",
        "target_dataset": "Cora_full_clean",
        "source_original_dim": 3703,
        "target_original_dim": 1433,
        "svd_dim": 100,
        "feature_alignment": FEATURE_ALIGNMENT,
        "source_svd_sha256": "",
        "encoder_dimensions": "100->256->256",
        "target_nodes": "",
        "target_edges": "",
        "shot": args.shot,
        "split": args.split,
        "train_n": "",
        "val_n": "",
        "test_n": "",
        "val_accuracy": "",
        "test_accuracy": "",
        "checkpoint_sha256": "",
        "checkpoint": str(checkpoint.resolve()),
        "log_path": str(Path(args.log_path).resolve()) if args.log_path else "",
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "error": "",
    }


def main() -> int:
    args = parse_args()
    results_path = Path(args.results_csv)
    summary_path = Path(args.summary_csv)

    if args.prepare_source_cache_only:
        try:
            prepare_citeseer_svd100_cache(
                Path(args.source_cache_path),
                Path(args.source_cache_receipt),
                results_path,
            )
            return 0
        except Exception as error:
            print(f"ERROR: source SVD100 preflight failed: {error}", file=sys.stderr)
            return 1

    checkpoint = Path(args.checkpoint)

    try:
        info = parse_checkpoint(checkpoint)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if args.validate_checkpoint_only:
        try:
            checkpoint_hash = validate_checkpoint_architecture(checkpoint)
            print(f"Checkpoint valid: {checkpoint.name} | sha256={checkpoint_hash}")
            return 0
        except Exception as error:
            print(f"INVALID CHECKPOINT: {error}", file=sys.stderr)
            return 1

    if args.skip_recorded:
        try:
            recorded, result_rows = successful_result(results_path, str(info["run_key"]))
            if recorded is not None:
                if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                    raise FileNotFoundError(
                        f"Recorded checkpoint is missing or empty: {checkpoint}"
                    )
                actual_hash = file_sha256(checkpoint)
                expected_hash = recorded["checkpoint_sha256"]
                if not expected_hash or actual_hash != expected_hash:
                    raise RuntimeError(
                        "Recorded checkpoint SHA256 mismatch: "
                        f"expected {expected_hash or '<missing>'}, got {actual_hash}."
                    )
                current_source_hash = source_svd_hash_from_receipt(
                    Path(args.source_cache_receipt)
                )
                if recorded["source_svd_sha256"] != current_source_hash:
                    raise RuntimeError(
                        "Recorded source SVD hash does not match the active source cache receipt."
                    )
                summaries = update_group_summary(summary_path, result_rows)
                complete_groups = sum(
                    summary["selection_status"] == "complete_5_of_5" for summary in summaries
                )
                print(
                    f"Already recorded and checkpoint verified: {info['run_key']} | "
                    f"summary rebuilt | complete_groups={complete_groups}/27"
                )
                return 0
        except Exception as error:
            print(f"ERROR: cannot safely skip recorded run: {error}", file=sys.stderr)
            return 1

    row = make_base_row(info, args, checkpoint)
    try:
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(f"Checkpoint is missing or empty: {checkpoint}")

        row["source_svd_sha256"] = source_svd_hash_from_receipt(
            Path(args.source_cache_receipt)
        )

        data, dataset, cache_action = load_cora_svd100(Path(args.cache_path))
        if int(dataset.num_classes) != 7:
            raise RuntimeError(f"Expected 7 Cora classes, got {dataset.num_classes}.")

        import torch

        from eval_pretrain import evaluate_checkpoint

        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        val_accuracy, test_accuracy = evaluate_checkpoint(str(checkpoint), data, device)
        if not math.isfinite(val_accuracy) or not math.isfinite(test_accuracy):
            raise RuntimeError("Evaluation returned non-finite accuracy.")

        row.update(
            {
                "status": "ok",
                "target_nodes": int(data.num_nodes),
                "target_edges": int(data.num_edges),
                "train_n": int(data.train_mask.sum().item()),
                "val_n": int(data.val_mask.sum().item()),
                "test_n": int(data.test_mask.sum().item()),
                "val_accuracy": f"{val_accuracy:.6f}",
                "test_accuracy": f"{test_accuracy:.6f}",
                "checkpoint_sha256": file_sha256(checkpoint),
                "error": "",
            }
        )
        result_rows = upsert_result(results_path, row)
        summaries = update_group_summary(summary_path, result_rows)

        complete_groups = [
            summary for summary in summaries if summary["selection_status"] == "complete_5_of_5"
        ]
        print(
            f"Result recorded: {info['run_key']} | cache={cache_action} | device={device} | "
            f"val={val_accuracy:.4f} | test={test_accuracy:.4f}"
        )
        print(
            f"Progress: {sum(result['status'] == 'ok' for result in result_rows)}/135 successful runs; "
            f"{len(complete_groups)}/27 complete 5-seed groups."
        )
        if complete_groups:
            best = complete_groups[0]
            print(
                "Current best complete group by validation mean: "
                f"aug1={best['aug1']} aug2={best['aug2']} ratio={best['aug_ratio']} | "
                f"val={best['val_mean']}+/-{best['val_std']} | "
                f"test={best['test_mean']}+/-{best['test_std']}"
            )
        return 0
    except Exception as error:
        row["error"] = " ".join(str(error).splitlines())
        row["recorded_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            result_rows = upsert_result(results_path, row)
            update_group_summary(summary_path, result_rows)
        except Exception as recording_error:
            print(f"ERROR while recording failure: {recording_error}", file=sys.stderr)
        print(f"ERROR: {row['error']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
