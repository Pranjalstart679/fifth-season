from __future__ import annotations

try:
    from .fuse_datasets import run_full_fusion_pipeline
except ImportError:
    from fuse_datasets import run_full_fusion_pipeline


if __name__ == "__main__":
    fused, train_dataset, test_dataset = run_full_fusion_pipeline()
    print(
        f"Fusion complete: fused={fused.sizes.get('time', 0)}, "
        f"train={train_dataset.sizes.get('time', 0)}, test={test_dataset.sizes.get('time', 0)}"
    )
