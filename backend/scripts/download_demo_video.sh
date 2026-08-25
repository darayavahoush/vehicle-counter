#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
mkdir -p demo_videos
curl -L -o demo_videos/car-detection.mp4 \
  "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
echo "Saved to backend/demo_videos/car-detection.mp4"
