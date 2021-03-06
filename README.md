
Samples of Python and C GStreamer pipelines with NVIDIA DeepStream modules:

To run examples:

* Start RTSP server, for example: [link](https://github.com/aler9/rtsp-simple-server).
* Prepare streaming, for instance, with ffmpeg:

```bash
$ bash stream_video.sh <video_address>
```
* For Jetson platform clone the repository. (!Streaming on Jetson is quite computational intensive. It is better to stream via another computer.)
* For dGPU use Docker.

```bash
$ docker build -f ./dockerfiles/deepstream-5.0-20.07-devel.Dockerfile -t gst_deepstream5.0_cuda10.2:dgpu .
$ xhost +local:docker
$ docker run --name ds_test --gpus=all --runtime nvidia  -e DISPLAY=$DISPLAY \
         --rm -i -t --net=host \
         -v $PWD:/home/ubuntu/inference-gstreamer \
         -v /tmp/.X11-unix/:/tmp/.X11-unix "gst_deepstream5.0_cuda10.2:dgpu" bash

```

Examples:

| Example                          | Description                                        |   Support    |
|----------------------------------|----------------------------------------------------|:------------:|
| `gst_read_rtsp.py`               | RTSP stream reading with reconnection              | Jetson, dGPU |
| `gst_read_multiple_rtsp.py`      | Multiple RTSP streams reading with reconnection    | Jetson, dGPU |
| `gst_primary_detector.py`        | RTSP object detection with deepstream PrimaryModel | Jetson, dGPU |
| `gst_trafficcam_model.py`        | TLT-pretrained TrafficCamNet model                 | Jetson, dGPU |
| `gst_dashcam_model.py`           | TLT-pretrained DashCamNet model                    | Jetson, dGPU |
| `gst_multiple_rtsp_inference.py` | RTSP streams + DashCam + PeopleNet + Tracker       | Jetson, dGPU |
|                                  |                                                    |              |

Arguments parser:
```
  -h, --help               show this help message and exit
  -ip ip                   str, rtsp ip address, default='127.0.0.1'
  -port port               int, rtsp port, default=8554
  -name name               str, rtsp address name or names, default='stream'
  -codec codec             str, video codec, default='h264
  -debug_level debug_level str, GStreamer debug level, default=0
  -d                       bool, display pipeline output
  -v                       bool, more verbosity
```


#### References

* [NVIDIA-AI-IOT deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)
* [DeepStream FAQ](https://docs.nvidia.com/metropolis/deepstream/5.0DP/dev-guide/index.html#page/DeepStream%20Plugins%20Development%20Guide/deepstream_plugin_faq.html)
* [DeepStream structures](https://docs.nvidia.com/metropolis/deepstream/5.0/python-api/index.html)
* [GStreamer C tutorial](https://gstreamer.freedesktop.org/documentation/tutorials/index.html?gi-language=c)
* [PyGObject documentation](http://lazka.github.io/pgi-docs/)
* [Python GStreamer Tutorial](https://brettviren.github.io/pygst-tutorial-org/pygst-tutorial.html)
* [RidgeRun Embedded Linux Developer Connection](https://developer.ridgerun.com/wiki/index.php?title=Main_Page)