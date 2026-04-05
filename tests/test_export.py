"""Tests for YOLOatr export pipeline."""

import os
import tempfile

from anima_yoloatr.export import export_onnx, export_safetensors
from anima_yoloatr.model import build_model


class TestExportSafetensors:
    """Test safetensors export."""

    def test_export_and_reload(self):
        model = build_model(num_classes=4)
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            export_safetensors(model, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

            from safetensors.torch import load_file

            state = load_file(path)
            assert len(state) > 0
        finally:
            os.unlink(path)


class TestExportOnnx:
    """Test ONNX export."""

    def test_export_creates_file(self):
        model = build_model(num_classes=4)
        model.eval()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            export_onnx(model, path, input_size=640, opset=17)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

            import onnx

            onnx_model = onnx.load(path)
            onnx.checker.check_model(onnx_model)
        finally:
            os.unlink(path)

    def test_onnx_runtime_inference(self):
        model = build_model(num_classes=4)
        model.eval()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        try:
            export_onnx(model, path, input_size=640, opset=17)

            try:
                import onnxruntime as ort
            except ImportError:
                return  # skip if no onnxruntime

            import numpy as np

            sess = ort.InferenceSession(path)
            dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
            outputs = sess.run(None, {"images": dummy})
            assert len(outputs) == 1
            # Total anchors: 3*(160*160 + 80*80 + 40*40 + 20*20)
            total = 3 * (160 * 160 + 80 * 80 + 40 * 40 + 20 * 20)
            assert outputs[0].shape == (1, total, 9)
        finally:
            os.unlink(path)
