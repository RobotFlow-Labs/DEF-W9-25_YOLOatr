"""Tests for YOLOatr dataset and augmentation pipeline."""

import numpy as np
import torch

from anima_yoloatr.dataset import YOLODataset, collate_fn


class TestYOLODataset:
    """Test dataset loading and augmentation."""

    def test_synthetic_sample(self):
        """Test that dataset returns synthetic data when no real data exists."""
        ds = YOLODataset(data_root="/nonexistent/path", img_size=640, augment=False)
        sample = ds[0]
        assert "image" in sample
        assert "labels" in sample
        assert sample["image"].shape[0] == 3  # channels
        assert sample["labels"].ndim == 2
        assert sample["labels"].shape[1] == 5  # cls, cx, cy, w, h

    def test_synthetic_batch(self):
        """Test collation of synthetic samples."""
        ds = YOLODataset(data_root="/nonexistent/path", img_size=640, augment=False)
        batch = [ds[0] for _ in range(4)]
        collated = collate_fn(batch)
        assert "images" in collated
        assert "labels" in collated
        assert collated["images"].shape == (4, 3, 640, 640)
        # Labels: [N, 6] with batch_idx prepended
        assert collated["labels"].shape[1] == 6

    def test_hyp_defaults(self):
        """Test that default augmentation hyperparameters match paper CAP."""
        ds = YOLODataset(data_root="/nonexistent/path", img_size=640)
        assert ds.hyp["mosaic"] == 0.1  # Low mosaic
        assert ds.hyp["shear"] == 0.0   # No shear
        assert ds.hyp["mixup"] == 0.4   # High mixup
        assert ds.hyp["copy_paste"] == 0.5  # High copy-paste
        assert ds.hyp["hsv_h"] == 0.015
        assert ds.hyp["hsv_s"] == 0.7
        assert ds.hyp["hsv_v"] == 0.4
        assert ds.hyp["degrees"] == 3.0
        assert ds.hyp["flipud"] == 0.1
        assert ds.hyp["fliplr"] == 0.5

    def test_letterbox(self):
        """Test letterbox resize preserves aspect ratio."""
        ds = YOLODataset(data_root="/nonexistent/path", img_size=640, augment=False)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        labels = np.array([[0, 0.5, 0.5, 0.1, 0.1]], dtype=np.float32)
        img_out, labels_out = ds._letterbox(img, labels, 640)
        assert img_out.shape == (640, 640, 3)
        assert labels_out.shape == (1, 5)


class TestCollate:
    """Test custom collation function."""

    def test_empty_labels(self):
        """Test collation with no labels."""
        batch = [
            {
                "image": torch.randn(3, 640, 640),
                "labels": torch.zeros((0, 5)),
                "img_path": "",
            }
            for _ in range(2)
        ]
        collated = collate_fn(batch)
        assert collated["images"].shape == (2, 3, 640, 640)
        assert collated["labels"].shape == (0, 6)

    def test_mixed_labels(self):
        """Test collation with varying label counts."""
        batch = [
            {
                "image": torch.randn(3, 640, 640),
                "labels": torch.tensor([[0, 0.5, 0.5, 0.1, 0.1]]),
                "img_path": "",
            },
            {
                "image": torch.randn(3, 640, 640),
                "labels": torch.tensor([
                    [1, 0.3, 0.3, 0.2, 0.2],
                    [2, 0.7, 0.7, 0.15, 0.15],
                ]),
                "img_path": "",
            },
        ]
        collated = collate_fn(batch)
        assert collated["images"].shape == (2, 3, 640, 640)
        assert collated["labels"].shape == (3, 6)
        # Check batch indices
        assert collated["labels"][0, 0] == 0.0  # first image
        assert collated["labels"][1, 0] == 1.0  # second image
        assert collated["labels"][2, 0] == 1.0  # second image
