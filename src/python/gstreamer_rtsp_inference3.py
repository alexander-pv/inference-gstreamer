import os
import sys

import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtsp', '1.0')
from gi.repository import GObject, Gst, GstRtsp

# Path for pyds library
sys.path.append(os.path.join(os.getcwd(), 'models', 'deep_stream'))
sys.path.append(os.path.join('/', 'opt', 'nvidia', 'deepstream', 'deepstream', 'lib'))
import pyds

from common import nvutils
from common import gstreamer_wrappers as gsw

COLORS = [[128, 128, 64], [0, 0, 128], [0, 128, 128], [128, 0, 0],
          [128, 0, 128], [128, 128, 0], [0, 128, 0], [0, 0, 64],
          [0, 0, 192], [0, 128, 64], [0, 128, 192], [128, 0, 64],
          [128, 0, 192], [128, 128, 128]]

if __name__ == '__main__':

    """
    Inference with TLT-pretrained PeopleSegNetV2 model
    https://ngc.nvidia.com/catalog/models/nvidia:tlt_peoplesegnet/files
    
    Issue:
        nvsegvisual gstnvsegvisual.cpp:614:gst_nvseg_visual_transform_internal:<nvsegvisual> SEG METADATA NOT FOUND
    """

    DISPLAY = True
    IP = '10.42.0.1'
    PORT = 8554
    CODEC = "h264"
    RTSP_REC_SRC = f'rtsp://{IP}:{PORT}/stream'
    PGIE_CONFIG = r'configs/pgie_people_segnetv2.txt'
    PGIE_BATCH = 1
    MUXER_BATCH = 1
    WIDTH, HEIGHT = 1920, 1080

    # GStreamer initialization
    Gst.debug_set_active(True)
    # Debug levels: 0-7
    Gst.debug_set_default_threshold(2)
    GObject.threads_init()
    Gst.init(None)

    # Gstreamer pipeline init
    pipeline = Gst.Pipeline.new('rtsp_client')

    # rtppsrc block
    rtspsrc = Gst.ElementFactory.make('rtspsrc', 'video-source')
    rtspsrc.set_property('location', RTSP_REC_SRC)
    rtspsrc.set_property('protocols', GstRtsp.RTSPLowerTrans.TCP)
    rtspsrc.set_property('retry', 25000)

    # Extract H26x video from RTSP: rtph26xdepay
    rtph26xdepay = Gst.ElementFactory.make(f"rtp{CODEC}depay", f"rtp{CODEC}depay")
    # Parse H26x video: h26xparser
    h26xparser = Gst.ElementFactory.make(f"{CODEC}parse", f"{CODEC}-parser")

    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    decoder.set_property("enable-max-performance", 1)
    decoder.set_property("enable-frame-type-reporting", 1)

    # Init streammuxer
    streammux = Gst.ElementFactory.make("nvstreammux", "stream_muxer")
    streammux.set_property('live-source', 1)
    streammux.set_property('width', WIDTH)
    streammux.set_property('height', HEIGHT)
    streammux.set_property('batch-size', MUXER_BATCH)
    streammux.set_property('batched-push-timeout', 10)

    # Primary GPU inference engine
    pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference")
    pgie.set_property('config-file-path', PGIE_CONFIG)
    pgie.set_property("batch-size", PGIE_BATCH)

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvidia_convertor")

    # Add capsfilter with RGBA for pgie_buffer_probe
    capsfilter = Gst.ElementFactory.make("capsfilter", "filter")
    capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=(string)RGBA"))

    if DISPLAY:
        # Segmentation vizualization
        nvsegvisual = Gst.ElementFactory.make("nvsegvisual", "nvsegvisual")
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        sink.set_property('sync', 1)
        nvsegvisual.set_property('batch-size', PGIE_BATCH)
        nvsegvisual.set_property('width', WIDTH)
        nvsegvisual.set_property('height', HEIGHT)
    else:
        sink = Gst.ElementFactory.make("fakesink", 'fake_sink')

    pipeline.add(rtspsrc)
    pipeline.add(rtph26xdepay)
    pipeline.add(h26xparser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    if DISPLAY:
        pipeline.add(capsfilter)
        pipeline.add(nvsegvisual)
        pipeline.add(transform)
    pipeline.add(sink)

    # Create sink pad for streammux with a specified id and connect with decoder srcpad
    streammux_sinkpad = streammux.get_request_pad('sink_%u' % 1)
    decoder_srcpad = decoder.get_static_pad("src")
    decoder_srcpad.link(streammux_sinkpad)

    # Link all elements in pipeline except source which has a dynamic source pad'''
    print("Linking elements in the Pipeline \n")
    rtph26xdepay.link(h26xparser)
    h26xparser.link(decoder)
    decoder.link(streammux)
    streammux.link(pgie)
    pgie.link(nvvidconv)
    if DISPLAY:
        nvvidconv.link(capsfilter)
        capsfilter.link(nvsegvisual)
        # pgie.link(nvsegvisual)
        nvsegvisual.link(transform)
        transform.link(sink)
    else:
        nvvidconv.link(sink)
        streammux.lnk(sink)

    # Probe for inference description
    # pgie_src_pad = pgie.get_static_pad("src")
    # pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, seg_src_pad_buffer_probe, 0)

    # Init RTSPHandler
    rtsp_handler = gsw.RTSPHandler(pipeline=pipeline,
                                   loop=GObject.MainLoop(),
                                   source=rtspsrc,
                                   connect_plugin=rtph26xdepay,
                                   **{'rtsp': {'addr': IP, 'port': PORT}}
                                   )

    try:
        rtsp_handler.loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)
    del pipeline
