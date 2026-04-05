"""YOLOatr serving node.

FastAPI server with /health, /ready, /info, /predict endpoints.
Also supports ROS2 Detection2DArray publishing when rclpy is available.

Usage:
    python -m anima_yoloatr.serve
    # or via Docker
    docker compose -f docker-compose.serve.yml --profile serve up -d
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

MODULE_NAME = "project_yoloatr"
MODULE_VERSION = "0.1.0"


class YOLOatrServe:
    """YOLOatr inference server."""

    def __init__(
        self,
        weight_path: str | None = None,
        device: str = "auto",
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.model = None
        self.device = self._resolve_device(device)
        self.weight_path = weight_path
        self.start_time = time.time()
        self.ready = False

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def setup_inference(self) -> None:
        """Load model weights and prepare for inference."""
        from anima_yoloatr.model import build_model

        self.model = build_model(num_classes=4)
        self.model = self.model.to(self.device)

        if self.weight_path and Path(self.weight_path).exists():
            ckpt = torch.load(self.weight_path, map_location=self.device)
            if "model" in ckpt:
                self.model.load_state_dict(ckpt["model"])
            else:
                self.model.load_state_dict(ckpt)

        self.model.eval()
        self.ready = True

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> list[dict]:
        """Run inference on a single image.

        Args:
            image: BGR image (H, W, 3) uint8

        Returns:
            List of detections: [{x1, y1, x2, y2, confidence, class_id, class_name}]
        """
        import cv2

        from anima_yoloatr.evaluate import non_max_suppression

        if self.model is None:
            return []

        class_names = ["T72_Tank", "BTR70", "SUV", "Pickup"]

        # Preprocess
        h0, w0 = image.shape[:2]
        img = cv2.resize(image, (640, 640))
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        # Inference
        predictions = self.model(tensor, decode=True)

        # NMS
        detections = non_max_suppression(
            predictions,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.nms_threshold,
        )[0]

        # Scale back to original image
        results = []
        sx, sy = w0 / 640, h0 / 640
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det.tolist()
            results.append({
                "x1": x1 * sx,
                "y1": y1 * sy,
                "x2": x2 * sx,
                "y2": y2 * sy,
                "confidence": conf,
                "class_id": int(cls_id),
                "class_name": class_names[int(cls_id)]
                if int(cls_id) < len(class_names) else "unknown",
            })

        return results

    def get_health(self) -> dict:
        """Health check response."""
        gpu_vram = 0
        if torch.cuda.is_available():
            gpu_vram = torch.cuda.memory_allocated() / 1e6

        return {
            "status": "healthy" if self.ready else "starting",
            "module": MODULE_NAME,
            "uptime_s": time.time() - self.start_time,
            "gpu_vram_mb": gpu_vram,
        }

    def get_ready(self) -> dict:
        """Readiness check response."""
        return {
            "ready": self.ready,
            "module": MODULE_NAME,
            "version": MODULE_VERSION,
            "weights_loaded": self.model is not None,
        }

    def get_info(self) -> dict:
        """Module info response."""
        return {
            "module": MODULE_NAME,
            "version": MODULE_VERSION,
            "description": "YOLOatr: Automatic Target Detection in Thermal IR",
            "num_classes": 4,
            "classes": ["T72_Tank", "BTR70", "SUV", "Pickup"],
            "input_size": 640,
            "device": str(self.device),
        }


def create_app():  # -> FastAPI (imported lazily)
    """Create FastAPI application."""
    from fastapi import FastAPI, UploadFile
    from fastapi.responses import JSONResponse

    app = FastAPI(title="YOLOatr Inference API", version=MODULE_VERSION)

    weight_dir = os.environ.get("ANIMA_WEIGHT_DIR", "/data/weights")
    weight_path = os.path.join(weight_dir, "best.pth")
    device = os.environ.get("ANIMA_DEVICE", "auto")

    server = YOLOatrServe(weight_path=weight_path, device=device)

    @app.on_event("startup")
    async def startup():
        server.setup_inference()

    @app.get("/health")
    async def health():
        return server.get_health()

    @app.get("/ready")
    async def ready():
        result = server.get_ready()
        status_code = 200 if result["ready"] else 503
        return JSONResponse(content=result, status_code=status_code)

    @app.get("/info")
    async def info():
        return server.get_info()

    @app.post("/predict")
    async def predict(file: UploadFile):
        import cv2

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return JSONResponse(
                content={"error": "Invalid image"}, status_code=400
            )
        detections = server.predict(image)
        return {"detections": detections, "count": len(detections)}

    return app


def main() -> None:
    """Run the serving application."""
    import uvicorn

    port = int(os.environ.get("ANIMA_SERVE_PORT", 8080))
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
