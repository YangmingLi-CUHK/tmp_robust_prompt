"""Strict reuse of fixed downstream feature caches for transfer experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def apply_cora_svd100_cache(
    data: Data,
    cache_path: str | Path,
    dataset_name: str,
    svd_out_dim: int,
) -> dict[str, object]:
    """Replace Cora features with the exact SVD100 basis used for clean selection.

    Metattack changes graph structure only. Every pollution level must therefore
    reuse one validated Cora feature tensor instead of fitting another SVD basis.
    """

    if dataset_name != "Cora" or svd_out_dim != 100:
        raise ValueError(
            "The fixed downstream cache path is dedicated to Cora SVD100; "
            f"got dataset={dataset_name!r}, svd_out_dim={svd_out_dim}."
        )

    path = Path(cache_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            "Missing the Cora SVD100 cache selected with the clean experiment: "
            f"{path}. Refusing to refit SVD in a downstream prompt run."
        )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Cora SVD100 cache is not a metadata payload: {path}")

    expected_metadata = {
        "version": 1,
        "dataset": "Cora",
        "data_scope": "full_clean_ptb_0.00",
        "preprocessing": "raw_bow_to_l1_to_independent_svd100",
        "original_shape": [2708, 1433],
        "svd_dim": 100,
        "svd_backend": "torch_geometric.SVDFeatureReduction(torch.linalg.svd)",
    }
    mismatches = {
        key: (payload.get(key), expected)
        for key, expected in expected_metadata.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Unexpected Cora SVD100 cache metadata: {mismatches}")

    raw_x = data.x.detach().cpu().contiguous()
    if tuple(raw_x.shape) != (2708, 1433) or not bool(torch.isfinite(raw_x).all()):
        raise RuntimeError(
            "Expected finite Cora raw/L1 features with shape (2708, 1433), got "
            f"{tuple(raw_x.shape)}."
        )

    expected_original_hash = payload.get("original_x_sha256")
    if not isinstance(expected_original_hash, str) or len(expected_original_hash) != 64:
        raise RuntimeError("Cora SVD100 cache has no valid original_x_sha256 receipt.")

    raw_hash = tensor_sha256(raw_x)
    normalization = "already_l1"
    if raw_hash != expected_original_hash:
        normalized_x = NormalizeFeatures()(Data(x=raw_x.clone())).x.contiguous()
        normalized_hash = tensor_sha256(normalized_x)
        if normalized_hash != expected_original_hash:
            raise RuntimeError(
                "The active Cora feature matrix does not match the feature order/content "
                "used to fit the selected SVD100 cache: "
                f"expected {expected_original_hash}, raw {raw_hash}, "
                f"L1-normalized {normalized_hash}."
            )
        normalization = "raw_to_l1"

    reduced_x = payload.get("x")
    if not isinstance(reduced_x, torch.Tensor):
        raise RuntimeError("Cora SVD100 cache payload has no tensor field 'x'.")
    reduced_x = reduced_x.detach().cpu().contiguous()
    reduced_hash = tensor_sha256(reduced_x)
    expected_reduced_hash = payload.get("reduced_x_sha256")
    if (
        tuple(reduced_x.shape) != (2708, 100)
        or not bool(torch.isfinite(reduced_x).all())
        or reduced_hash != expected_reduced_hash
    ):
        raise RuntimeError(
            "Invalid Cora SVD100 tensor or SHA256 receipt: "
            f"shape={tuple(reduced_x.shape)}, expected_hash={expected_reduced_hash}, "
            f"actual_hash={reduced_hash}."
        )

    row_norms = reduced_x.norm(p=2, dim=1)
    min_row_norm = float(row_norms.min().item())
    if min_row_norm <= 0.0:
        raise RuntimeError(
            "Cora SVD100 contains a zero-norm row, which is unsafe for the original "
            "GPromptShield cosine normalization."
        )

    mask_counts = (
        int(data.train_mask.sum().item()),
        int(data.val_mask.sum().item()),
        int(data.test_mask.sum().item()),
    )
    if mask_counts != (35, 265, 2408):
        raise RuntimeError(
            "Expected Cora 5-shot/split-1 masks 35/265/2408, got "
            f"{mask_counts[0]}/{mask_counts[1]}/{mask_counts[2]}."
        )

    data.x = reduced_x.to(device=data.x.device)
    receipt = {
        "cache_path": str(path.resolve()),
        "normalization": normalization,
        "original_x_sha256": expected_original_hash,
        "reduced_x_sha256": reduced_hash,
        "min_row_l2_norm": min_row_norm,
        "torch_version": payload.get("torch_version", "unknown"),
        "torch_geometric_version": payload.get("torch_geometric_version", "unknown"),
    }
    print(
        "Downstream feature cache verified | "
        "pipeline=Cora_raw1433->L1->fixed_independent_SVD100 | "
        f"normalization={normalization} | reduced_sha256={reduced_hash} | "
        f"min_row_l2={min_row_norm:.8g}"
    )
    return receipt
