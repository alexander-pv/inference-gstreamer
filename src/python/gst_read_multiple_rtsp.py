import logging
import sys

import gi
import numpy as np

gi.require_version('Gst', '1.0')
gi.require_version('GstRtsp', '1.0')

from gi.repository import GObject, Gst
from common import nvutils
from common import gstreamer_wrappers as gsw
from common import utils


def main():
    args = utils.parse_arguments()
    if args.v:
        utils.set_logging()

    rtsp_sources = [f'rtsp://{args.ip}:{args.port}/{name}' for name in args.name]

    stream_multiplier = 1
    width, height = 1920, 1080
    batched_push_timeout = 10
    attach_sys_ts = 1
    n_sources = len(rtsp_sources) * stream_multiplier
    tiler_rows = int(n_sources ** 0.5)
    tiler_columns = int(np.math.ceil((1.0 * n_sources) / tiler_rows))

    Gst.debug_set_active(bool(args.debug_level))
    Gst.debug_set_default_threshold(args.debug_level)
    GObject.threads_init()
    Gst.init(None)

    pipeline = Gst.Pipeline.new('rtsp_client')

    streammux = Gst.ElementFactory.make("nvstreammux", "stream_muxer")
    streammux.set_property('live-source', 1)

    streammux.set_property('width', width)
    streammux.set_property('height', height)
    streammux.set_property('batch-size', stream_multiplier)
    streammux.set_property('batched-push-timeout', batched_push_timeout)
    streammux.set_property('attach-sys-ts', attach_sys_ts)
    pipeline.add(streammux)

    rtsp_blocks = {}
    for i in range(n_sources):

        rtsp_bin = gsw.RTSPBin(builder_id=i, location=rtsp_sources[i], compression=args.codec)
        rtsp_blocks.update({i: rtsp_bin})

        if nvutils.is_aarch64():
            pipeline.add(rtsp_bin.rtspsrc)
            pipeline.add(rtsp_bin.depayer)
            pipeline.add(rtsp_bin.parser)
            pipeline.add(rtsp_bin.decoder)

            rtsp_bin.depayer.link(rtsp_bin.parser)
            rtsp_bin.parser.link(rtsp_bin.decoder)

            decoder_srcpad = rtsp_bin.decoder.get_static_pad("src")
            streammux_handler = gsw.StreamMuxHandler(next_element=streammux, scr_pad=decoder_srcpad, index=i)
            rtsp_bin.rtspsrc.connect("pad-added", streammux_handler.on_pad_added)
            rtsp_bin.rtspsrc.connect("pad-removed", streammux_handler.on_pad_removed)
        else:
            pipeline.add(rtsp_bin.rtspsrc)
            pipeline.add(rtsp_bin.decoder)
            streammux_handler = gsw.StreamMuxHandler(next_element=streammux, index=i)
            rtsp_bin.decoder.connect("pad-added", streammux_handler.on_pad_added)
            rtsp_bin.decoder.connect("pad-removed", streammux_handler.on_pad_removed)

    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", width)
    tiler.set_property("height", height)
    pipeline.add(tiler)

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvidia_convertor")

    capsfilter = Gst.ElementFactory.make("capsfilter", "filter")
    capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=(string)NV12"))
    pipeline.add(capsfilter)

    if args.d:
        if nvutils.is_aarch64():
            transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
            pipeline.add(transform)
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        sink.set_property('sync', 1)
    else:
        sink = Gst.ElementFactory.make("fakesink", 'fake_sink')

    pipeline.add(nvvidconv)
    pipeline.add(sink)

    logging.info("Linking elements in the Pipeline \n")
    streammux.link(tiler)
    tiler.link(nvvidconv)
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
                                   basic_blocks=rtsp_blocks,
                                   **{'rtsp': {'addr': args.ip, 'port': args.port}}
                                   )
    try:
        rtsp_handler.loop.run()
    except KeyboardInterrupt as e:
        logging.error(e)
    except Exception as e:
        logging.error(e)

    pipeline.set_state(Gst.State.NULL)
    del pipeline


if __name__ == '__main__':
    sys.exit(main())
