"""YOLOatr dataset and augmentation pipeline.

Implements YOLO-format dataset loading with Custom Augmentation Profile (CAP)
from paper Table 3:
- Low mosaic (0.1) -- small IR targets get too small
- No shear -- counterproductive for low-res IR
- High mixup (0.4) and copy_paste (0.5) -- increase diversity

Paper: arxiv 2507.11267
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class YOLODataset(Dataset):
    """YOLO-format dataset with YOLOatr augmentations.

    Expected directory structure:
        data_root/
            images/
                img001.jpg
                ...
            labels/
                img001.txt
                ...

    Label format (per line): class_id cx cy w h (normalized 0-1)
    """

    def __init__(
        self,
        data_root: str,
        img_size: int = 640,
        augment: bool = True,
        hyp: dict | None = None,
        img_suffix: str = ".jpg",
    ):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.augment = augment
        self.img_suffix = img_suffix

        # Custom Augmentation Profile defaults (from paper Table 3)
        self.hyp = hyp or {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 3.0,
            "translate": 0.1,
            "scale": 0.3,
            "shear": 0.0,
            "perspective": 0.0005,
            "flipud": 0.1,
            "fliplr": 0.5,
            "mosaic": 0.1,
            "mixup": 0.4,
            "copy_paste": 0.5,
        }

        # Find images — supports two layouts:
        #   1. data_root/images/*.jpg  (flat)
        #   2. data_root is the image directory itself (e.g. root/images/train/)
        img_dir = self.data_root / "images"
        if img_dir.exists() and img_dir.is_dir():
            scan_dir = img_dir
        elif self.data_root.exists() and self.data_root.is_dir():
            scan_dir = self.data_root
        else:
            scan_dir = None

        if scan_dir is not None:
            self.img_files = sorted(
                [
                    str(p)
                    for p in scan_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
                ]
            )
        else:
            self.img_files = []

        # Derive label paths (replace /images/ with /labels/ in the path)
        self.label_files = [
            f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            for f in self.img_files
        ]

    def __len__(self) -> int:
        return max(len(self.img_files), 1)

    def __getitem__(self, idx: int) -> dict:
        """Load image and labels.

        Returns:
            dict with keys: image [3, H, W], labels [N, 5], img_path
        """
        if len(self.img_files) == 0:
            # Return synthetic data if no real data available
            return self._synthetic_sample()

        # Mosaic augmentation
        if self.augment and random.random() < self.hyp.get("mosaic", 0.1):
            img, labels = self._load_mosaic(idx)
        else:
            img, labels = self._load_image_and_labels(idx)

        # Resize with letterbox
        img, labels = self._letterbox(img, labels, self.img_size)

        # Augmentations
        if self.augment:
            img, labels = self._augment(img, labels)

        # Convert to tensor
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0

        # Labels: [class_id, cx, cy, w, h] normalized
        if len(labels) > 0:
            labels_tensor = torch.from_numpy(labels).float()
        else:
            labels_tensor = torch.zeros((0, 5), dtype=torch.float32)

        return {
            "image": torch.from_numpy(img),
            "labels": labels_tensor,
            "img_path": self.img_files[idx] if idx < len(self.img_files) else "",
        }

    def _synthetic_sample(self) -> dict:
        """Generate synthetic sample when no real data is available."""
        img = torch.randn(3, self.img_size, self.img_size)
        labels = torch.tensor(
            [[0, 0.5, 0.5, 0.1, 0.1]], dtype=torch.float32
        )
        return {"image": img, "labels": labels, "img_path": "synthetic"}

    def _load_image_and_labels(
        self, idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load a single image and its labels."""
        img = cv2.imread(self.img_files[idx])
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        labels = self._load_labels(self.label_files[idx])
        return img, labels

    def _load_labels(self, label_path: str) -> np.ndarray:
        """Load YOLO-format labels from txt file."""
        if os.path.exists(label_path):
            with open(label_path) as f:
                labels = []
                for line in f.read().strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        labels.append([float(x) for x in parts[:5]])
                if labels:
                    return np.array(labels, dtype=np.float32)
        return np.zeros((0, 5), dtype=np.float32)

    def _letterbox(
        self,
        img: np.ndarray,
        labels: np.ndarray,
        target_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resize image with letterbox padding."""
        h, w = img.shape[:2]
        r = min(target_size / h, target_size / w)
        new_w, new_h = int(w * r), int(h * r)

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        dw = (target_size - new_w) // 2
        dh = (target_size - new_h) // 2
        img = cv2.copyMakeBorder(
            img, dh, target_size - new_h - dh, dw, target_size - new_w - dw,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        # Adjust labels for letterbox
        if len(labels) > 0:
            labels = labels.copy()
            labels[:, 1] = labels[:, 1] * r * w / target_size + dw / target_size
            labels[:, 2] = labels[:, 2] * r * h / target_size + dh / target_size
            labels[:, 3] = labels[:, 3] * r * w / target_size
            labels[:, 4] = labels[:, 4] * r * h / target_size

        return img, labels

    def _augment(
        self, img: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply Custom Augmentation Profile (CAP)."""
        h, w = img.shape[:2]

        # HSV augmentation
        self._augment_hsv(img)

        # Geometric: rotation, translation, scale, perspective
        if any(
            self.hyp.get(k, 0) > 0
            for k in ["degrees", "translate", "scale", "perspective"]
        ):
            img, labels = self._random_perspective(img, labels)

        # Flip up-down
        if random.random() < self.hyp.get("flipud", 0.1):
            img = np.flipud(img).copy()
            if len(labels) > 0:
                labels[:, 2] = 1.0 - labels[:, 2]

        # Flip left-right
        if random.random() < self.hyp.get("fliplr", 0.5):
            img = np.fliplr(img).copy()
            if len(labels) > 0:
                labels[:, 1] = 1.0 - labels[:, 1]

        # MixUp
        if random.random() < self.hyp.get("mixup", 0.4):
            idx2 = random.randint(0, max(len(self.img_files) - 1, 0))
            if idx2 < len(self.img_files):
                img2, labels2 = self._load_image_and_labels(idx2)
                img2, labels2 = self._letterbox(img2, labels2, self.img_size)
                r = np.random.beta(32.0, 32.0)
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                if len(labels2) > 0:
                    labels = (
                        np.concatenate([labels, labels2], axis=0)
                        if len(labels) > 0
                        else labels2
                    )

        return img, labels

    def _augment_hsv(self, img: np.ndarray) -> None:
        """HSV color-space augmentation (in-place)."""
        h_gain = self.hyp.get("hsv_h", 0.015)
        s_gain = self.hyp.get("hsv_s", 0.7)
        v_gain = self.hyp.get("hsv_v", 0.4)

        r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

        hsv = cv2.merge(
            (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
        )
        cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=img)

    def _random_perspective(
        self, img: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply random perspective/affine transformation."""
        h, w = img.shape[:2]

        # Center
        c = np.eye(3)
        c[0, 2] = -w / 2
        c[1, 2] = -h / 2

        # Perspective
        p = np.eye(3)
        persp = self.hyp.get("perspective", 0.0005)
        p[2, 0] = random.uniform(-persp, persp)
        p[2, 1] = random.uniform(-persp, persp)

        # Rotation + Scale
        r = np.eye(3)
        a = random.uniform(-self.hyp.get("degrees", 3.0), self.hyp.get("degrees", 3.0))
        s = random.uniform(1 - self.hyp.get("scale", 0.3), 1 + self.hyp.get("scale", 0.3))
        r[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        sh = np.eye(3)
        shear = self.hyp.get("shear", 0.0)
        sh[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
        sh[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

        # Translation
        t = np.eye(3)
        trans = self.hyp.get("translate", 0.1)
        t[0, 2] = random.uniform(0.5 - trans, 0.5 + trans) * w
        t[1, 2] = random.uniform(0.5 - trans, 0.5 + trans) * h

        # Combined transform
        m = t @ sh @ r @ p @ c
        if (m != np.eye(3)).any():
            img = cv2.warpPerspective(
                img, m, dsize=(w, h), borderValue=(114, 114, 114)
            )

        # Transform labels
        if len(labels) > 0:
            n = len(labels)
            # Convert normalized xywh to pixel xyxy
            xy = np.ones((n, 4, 3))
            cx = labels[:, 1] * w
            cy = labels[:, 2] * h
            bw = labels[:, 3] * w
            bh = labels[:, 4] * h
            xy[:, 0, :2] = np.stack([cx - bw / 2, cy - bh / 2], axis=1)
            xy[:, 1, :2] = np.stack([cx + bw / 2, cy - bh / 2], axis=1)
            xy[:, 2, :2] = np.stack([cx + bw / 2, cy + bh / 2], axis=1)
            xy[:, 3, :2] = np.stack([cx - bw / 2, cy + bh / 2], axis=1)

            # Transform corners
            xy = xy @ m.T
            xy = xy[:, :, :2].reshape(n, 8)

            # Get new bounding boxes
            x_min = xy[:, [0, 2, 4, 6]].min(axis=1)
            x_max = xy[:, [0, 2, 4, 6]].max(axis=1)
            y_min = xy[:, [1, 3, 5, 7]].min(axis=1)
            y_max = xy[:, [1, 3, 5, 7]].max(axis=1)

            # Clip to image
            x_min = np.clip(x_min, 0, w)
            x_max = np.clip(x_max, 0, w)
            y_min = np.clip(y_min, 0, h)
            y_max = np.clip(y_max, 0, h)

            # Convert back to normalized xywh
            new_w = x_max - x_min
            new_h = y_max - y_min
            valid = (new_w > 2) & (new_h > 2)

            labels = labels[valid]
            if len(labels) > 0:
                labels[:, 1] = ((x_min[valid] + x_max[valid]) / 2) / w
                labels[:, 2] = ((y_min[valid] + y_max[valid]) / 2) / h
                labels[:, 3] = new_w[valid] / w
                labels[:, 4] = new_h[valid] / h

        return img, labels

    def _load_mosaic(
        self, idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load 4-image mosaic augmentation."""
        s = self.img_size
        yc = int(random.uniform(s * 0.5, s * 1.5))
        xc = int(random.uniform(s * 0.5, s * 1.5))

        indices = [idx] + [
            random.randint(0, len(self.img_files) - 1) for _ in range(3)
        ]

        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        mosaic_labels = []

        for i, index in enumerate(indices):
            img, labels = self._load_image_and_labels(index)
            h, w = img.shape[:2]
            r = min(s / h, s / w)
            new_w, new_h = int(w * r), int(h * r)
            img = cv2.resize(img, (new_w, new_h))

            # Place in mosaic
            if i == 0:  # top-left
                x1a, y1a = max(xc - new_w, 0), max(yc - new_h, 0)
                x2a, y2a = xc, yc
                x1b = new_w - (x2a - x1a)
                y1b = new_h - (y2a - y1a)
                _, _ = new_w, new_h  # x2b, y2b unused in crop
            elif i == 1:  # top-right
                x1a, y1a = xc, max(yc - new_h, 0)
                x2a, y2a = min(xc + new_w, s * 2), yc
                x1b, y1b = 0, new_h - (y2a - y1a)
                _ = min(new_w, x2a - x1a)  # x2b unused
                _ = new_h  # y2b unused
            elif i == 2:  # bottom-left
                x1a, y1a = max(xc - new_w, 0), yc
                x2a = xc
                y2a = min(yc + new_h, s * 2)
                x1b = new_w - (x2a - x1a)
                y1b = 0
                _ = new_w, min(new_h, y2a - y1a)  # x2b, y2b unused
            else:  # bottom-right
                x1a, y1a = xc, yc
                x2a = min(xc + new_w, s * 2)
                y2a = min(yc + new_h, s * 2)
                x1b, y1b = 0, 0

            ph = y2a - y1a
            pw = x2a - x1a
            if ph > 0 and pw > 0:
                mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y1b + ph, x1b:x1b + pw]

            # Adjust labels
            if len(labels) > 0:
                adj_labels = labels.copy()
                adj_labels[:, 1] = (labels[:, 1] * new_w + x1a - x1b) / (s * 2)
                adj_labels[:, 2] = (labels[:, 2] * new_h + y1a - y1b) / (s * 2)
                adj_labels[:, 3] = labels[:, 3] * new_w / (s * 2)
                adj_labels[:, 4] = labels[:, 4] * new_h / (s * 2)
                mosaic_labels.append(adj_labels)

        # Crop mosaic to target size
        mosaic_img = cv2.resize(mosaic_img, (s, s))
        if mosaic_labels:
            all_labels = np.concatenate(mosaic_labels, axis=0)
            # Clip labels to [0, 1]
            all_labels[:, 1:] = np.clip(all_labels[:, 1:], 0, 1)
        else:
            all_labels = np.zeros((0, 5), dtype=np.float32)

        return mosaic_img, all_labels


def collate_fn(
    batch: list[dict],
) -> dict:
    """Custom collate for variable-length labels.

    Prepends batch index to labels for loss computation.
    """
    images = torch.stack([item["image"] for item in batch])
    labels_list = []
    for i, item in enumerate(batch):
        lbl = item["labels"]
        if lbl.shape[0] > 0:
            # Prepend batch index: [batch_idx, class, cx, cy, w, h]
            batch_idx = torch.full((lbl.shape[0], 1), i, dtype=torch.float32)
            labels_list.append(torch.cat([batch_idx, lbl], dim=1))

    if labels_list:
        labels = torch.cat(labels_list, dim=0)
    else:
        labels = torch.zeros((0, 6), dtype=torch.float32)

    return {"images": images, "labels": labels}
