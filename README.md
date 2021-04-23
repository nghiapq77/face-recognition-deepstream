# Face Recognition DeepStream
This is a face recognition app built on DeepStream reference app.  
[RetinaFace](https://github.com/biubug6/Face-Detector-1MB-with-landmark) and [ArcFace](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) is used for detection and recognition respectively.

## Prerequisites
You must have the following development packages installed
- GStreamer-1.0
- GStreamer-1.0 Base Plugins
- GStreamer-1.0 gstrtspserver
- X11 client-side library
- Glib json library - json-glib-1.0

You can install those packages via **apt**:
```bash
sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgstrtspserver-1.0-dev libx11-dev libjson-glib-dev
```

## Requirements
- DeepStream 5.0
- CUDA 10.2

## Installation
```bash
git clone https://github.com/nghiapq77/face-recognition-deepstream.git
cd face-recognition-deepstream
mkdir build && cd build && ln -s ../Makefile
make -j$(nproc)
```

## Pretrained
Please refer to [this repo](https://github.com/nghiapq77/face-recognition-cpp-tensorrt) for pretrained models and serialized TensorRT engine.

## Usage
```bash
./deepstream-app -c <config-file>
```

## Develop
See [DEVELOP.md](DEVELOP.md) for my modified code to original app.

## References
- [Face-Detector-1MB-with-landmark](https://github.com/biubug6/Face-Detector-1MB-with-landmark)
- [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)
