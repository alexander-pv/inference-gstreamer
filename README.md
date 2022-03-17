
Samples of GStreamer pipelines based on python bindings and NVIDIA DeepStream modules:

To run examples:

* Start RTSP server, for example: [link](https://github.com/aler9/rtsp-simple-server)
* Prepare streaming, via ffmpeg:

```bash
$ bash stream_video.sh <video_address>
```
* Run Docker container with DeepStream

```bash
$ 
```

Examples:

1. `gstreamer_read_rtsp.py` - Jetson, dGPU, RTSP reading example with reconnection.
