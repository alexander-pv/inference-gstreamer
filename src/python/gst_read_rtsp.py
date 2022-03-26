import logging
import sys

import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtsp', '1.0')
from gi.repository import GObject, Gst

from common import nvutils
from common import gstreamer_wrappers as gsw
from common.utils import parse_arguments


def main():
    args = parse_arguments()
    rtsp_source = f'rtsp://{args.ip}:{args.port}/{args.name}'

    Gst.debug_set_active(bool(args.debug_level))
    Gst.debug_set_default_threshold(args.debug_level)
    GObject.threads_init()
    Gst.init(None)

    pipeline = Gst.Pipeline.new('rtsp_client')
    rtsp_bin = gsw.RTSPBin(builder_id=0, location=rtsp_source, compression=args.codec)

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvidia_convertor")

    capsfilter = Gst.ElementFactory.make("capsfilter", "filter")
    capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=(string)RGBA"))
    pipeline.add(capsfilter)

    if args.d:
        if nvutils.is_aarch64():
            transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
            pipeline.add(transform)
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        sink.set_property('sync', 1)
    else:
        sink = Gst.ElementFactory.make("fakesink", 'fake_sink')

    pipeline.add(rtsp_bin.rtspsrc)
    if rtsp_bin.is_aarch64:
        pipeline.add(rtsp_bin.rtph26xdepay)
        pipeline.add(rtsp_bin.h26xparser)
    pipeline.add(rtsp_bin.decoder)
    pipeline.add(nvvidconv)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    if nvutils.is_aarch64():
        rtsp_bin.rtph26xdepay.link(rtsp_bin.h26xparser)
        rtsp_bin.h26xparser.link(rtsp_bin.decoder)
        rtsp_bin.decoder.link(nvvidconv)
    else:
        nvidconv_sinkpad = nvvidconv.get_static_pad("sink")
        decodebin_handler = gsw.DecodeBinHandler(sink_pad=nvidconv_sinkpad)
        rtsp_bin.decoder.connect("pad-added", decodebin_handler.decodebin_pad_added)
        rtsp_bin.decoder.connect("pad-removed", decodebin_handler.decodebin_pad_removed)

    nvvidconv.link(capsfilter)
    if args.d:
        if nvutils.is_aarch64():
            capsfilter.link(transform)
            transform.link(sink)
        else:
            capsfilter.link(sink)
    else:
        capsfilter.link(sink)

    rtsp_handler = gsw.RTSPHandler(pipeline=pipeline,
                                   loop=GObject.MainLoop(),
                                   basic_blocks={0: rtsp_bin},
                                   **{'rtsp': {'addr': args.ip, 'port': args.port}}
                                   )
    try:
        rtsp_handler.loop.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(e)

    pipeline.set_state(Gst.State.NULL)
    del pipeline


if __name__ == '__main__':
    sys.exit(main())
