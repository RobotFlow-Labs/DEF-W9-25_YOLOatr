"""YOLOatr model export pipeline.

Export formats:
1. safetensors — portable weight format
2. ONNX — cross-platform inference (opset 17)
3. TensorRT FP32 — NVIDIA optimized inference
4. TensorRT FP16 — NVIDIA half-precision inference

Paper: arxiv 2507.11267
"""

from __future__ import annotations

import argparse
import os
import subprocess

import torch

from anima_yoloatr.model import build_model
from anima_yoloatr.utils import load_config

PROJECT = "project_yoloatr"
ARTIFACTS = "/mnt/artifacts-datai"
DEFAULT_EXPORT_DIR = f"{ARTIFACTS}/exports/{PROJECT}"


def export_safetensors(
    model: torch.nn.Module,
    output_path: str,
) -> str:
    """Export model weights to safetensors format."""
    from safetensors.torch import save_file

    state_dict = model.state_dict()
    save_file(state_dict, output_path)
    print(f"[EXPORT] safetensors: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")
    return output_path


def export_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_size: int = 640,
    opset: int = 17,
    num_classes: int = 4,
) -> str:
    """Export model to ONNX format."""
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

    # Wrap model for ONNX export (decode mode for inference)
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x, decode=True)

    wrapper = OnnxWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["images"],
        output_names=["detections"],
    )
    print(f"[EXPORT] ONNX: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")
    return output_path


def validate_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    input_size: int = 640,
    atol: float = 1e-3,
) -> bool:
    """Validate ONNX model output matches PyTorch."""
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnxruntime not installed, skipping ONNX validation")
        return True

    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, input_size, input_size, device=device)

    with torch.no_grad():
        pt_out = model(dummy, decode=True).cpu().numpy()

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {"images": dummy.cpu().numpy()})[0]

    max_diff = np.abs(pt_out - ort_out).max()
    ok = max_diff < atol
    print(f"[VALIDATE] ONNX max diff: {max_diff:.6f} {'PASS' if ok else 'FAIL'}")
    return ok


def export_trt(
    onnx_path: str,
    output_path: str,
    fp16: bool = False,
    input_size: int = 640,
) -> str:
    """Export ONNX to TensorRT engine.

    Uses trtexec CLI tool. Falls back to shared toolkit if available.
    """
    precision = "fp16" if fp16 else "fp32"
    print(f"[EXPORT] Building TRT {precision} engine...")

    # Try trtexec first
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        f"--minShapes=images:1x3x{input_size}x{input_size}",
        f"--optShapes=images:1x3x{input_size}x{input_size}",
        f"--maxShapes=images:8x3x{input_size}x{input_size}",
    ]
    if fp16:
        cmd.append("--fp16")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(
                f"[EXPORT] TRT {precision}: {output_path} "
                f"({os.path.getsize(output_path) / 1e6:.1f} MB)"
            )
            return output_path
        print(f"[WARN] trtexec failed: {result.stderr[:200]}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[WARN] trtexec not available: {e}")

    # Fallback to shared toolkit (exports both fp16 and fp32 at once)
    shared_trt = "/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py"
    if os.path.exists(shared_trt):
        output_dir = os.path.dirname(output_path)
        cmd = [
            "python", shared_trt,
            "--onnx", onnx_path,
            "--output-dir", output_dir,
        ]
        if not fp16:
            cmd.append("--no-fp16")
        if fp16:
            cmd.append("--no-fp32")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                print(f"[EXPORT] TRT {precision} (shared toolkit): {output_dir}")
                # Check if the expected output file exists
                if os.path.exists(output_path):
                    return output_path
                # Try common naming patterns
                for candidate in [
                    output_path,
                    os.path.join(output_dir, f"yoloatr_{precision}.trt"),
                    os.path.join(output_dir, f"model_{precision}.engine"),
                ]:
                    if os.path.exists(candidate):
                        return candidate
                print(f"[WARN] TRT built but output file not found at {output_path}")
                return output_dir
            print(f"[WARN] Shared TRT toolkit failed: {result.stderr[:200]}")
        except Exception as e:
            print(f"[WARN] Shared TRT toolkit error: {e}")

    print(f"[SKIP] TRT {precision} export failed — trtexec and shared toolkit unavailable")
    return ""


def export_all(
    checkpoint_path: str,
    export_dir: str = DEFAULT_EXPORT_DIR,
    num_classes: int = 4,
    input_size: int = 640,
    onnx_opset: int = 17,
) -> dict[str, str]:
    """Run full export pipeline: pth -> safetensors -> ONNX -> TRT fp32 + fp16.

    Args:
        checkpoint_path: Path to trained checkpoint (.pth)
        export_dir: Directory for exported models
        num_classes: Number of detection classes
        input_size: Input image size

    Returns:
        Dict mapping format name to output path
    """
    os.makedirs(export_dir, exist_ok=True)
    results = {}

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=num_classes)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    print(f"[EXPORT] Loaded checkpoint: {checkpoint_path}")
    print(f"[EXPORT] Output dir: {export_dir}")

    # 1. safetensors
    st_path = os.path.join(export_dir, "yoloatr.safetensors")
    results["safetensors"] = export_safetensors(model, st_path)

    # 2. ONNX
    onnx_path = os.path.join(export_dir, "yoloatr.onnx")
    results["onnx"] = export_onnx(model, onnx_path, input_size, onnx_opset, num_classes)

    # Validate ONNX
    validate_onnx(model, onnx_path, input_size)

    # 3. TRT FP32
    trt_fp32_path = os.path.join(export_dir, "yoloatr_fp32.engine")
    results["trt_fp32"] = export_trt(onnx_path, trt_fp32_path, fp16=False, input_size=input_size)

    # 4. TRT FP16
    trt_fp16_path = os.path.join(export_dir, "yoloatr_fp16.engine")
    results["trt_fp16"] = export_trt(onnx_path, trt_fp16_path, fp16=True, input_size=input_size)

    print("\n[EXPORT] Summary:")
    for fmt, path in results.items():
        status = "OK" if path and os.path.exists(path) else "SKIP"
        print(f"  [{status}] {fmt}: {path}")

    return results


def main() -> None:
    """CLI entry point for export."""
    parser = argparse.ArgumentParser(description="Export YOLOatr model")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/paper.toml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Export output directory",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    export_cfg = config.get("export", {})

    export_dir = args.output_dir or export_cfg.get("output_dir", DEFAULT_EXPORT_DIR)

    export_all(
        checkpoint_path=args.checkpoint,
        export_dir=export_dir,
        num_classes=model_cfg.get("num_classes", 4),
        input_size=model_cfg.get("input_size", 640),
        onnx_opset=export_cfg.get("onnx_opset", 17),
    )


if __name__ == "__main__":
    main()
