# PRD-07: Integration

> Status: TODO
> Module: anima_yoloatr

## Objective
Docker serving infrastructure, ROS2 node, HuggingFace model push, and ANIMA
ecosystem integration.

## Docker Serving
- Dockerfile.serve: 3-layer build (anima-serve:jazzy base)
- docker-compose.serve.yml: profiles (serve, ros2, api, test)
- FastAPI endpoints: /health, /ready, /info, /predict
- Weight download from HF at runtime

## ROS2 Node
- AnimaNode subclass for YOLOatr inference
- Input: sensor_msgs/Image (thermal IR)
- Output: vision_msgs/Detection2DArray
- Topic: /yoloatr/detections

## HuggingFace Push
- Repo: ilessio-aiflowlab/project_yoloatr-checkpoint
- Contents: safetensors, ONNX, TRT engines, config, metrics

## Deliverables
- [ ] Dockerfile.serve
- [ ] docker-compose.serve.yml
- [ ] .env.serve
- [ ] src/anima_yoloatr/serve.py -- AnimaNode subclass
- [ ] HF model card (README.md for HF repo)
- [ ] Push script

## Acceptance Criteria
- Docker image builds successfully
- /health endpoint returns 200
- /predict accepts image and returns detections
- ROS2 node publishes Detection2DArray
- HF repo accessible and downloadable
