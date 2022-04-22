import logging
import os
import sys

import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtsp', '1.0')
from gi.repository import GObject, Gst

# Path for pyds library if python bindings are not installed
sys.path.append(os.path.join('/', 'opt', 'nvidia', 'deepstream', 'deepstream', 'lib'))
import pyds

from common import nvutils
from common import gstreamer_wrappers as gsw
from common import utils

PGIE_CLASSES = ["Vehicle", "TwoWheeler", "Person", "RoadSign"]
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

stream_fps = {}


def pgie_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        obj_counter = {
            PGIE_CLASS_ID_VEHICLE: 0,
            PGIE_CLASS_ID_PERSON: 0,
            PGIE_CLASS_ID_BICYCLE: 0,
            PGIE_CLASS_ID_ROADSIGN: 0
        }
        l_obj_info = []
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            # Getting Image data using nvbufsurface
            # the input should be address of buffer and batch_id
            # n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            rect_params = obj_meta.rect_params
            top = int(rect_params.top)
            left = int(rect_params.left)
            width = int(rect_params.width)
            height = int(rect_params.height)
            obj_name = PGIE_CLASSES[obj_meta.class_id]
            l_obj_info.append(
                (
                    frame_meta.batch_id,
                    frame_meta.pad_index,
                    obj_meta.confidence,
                    (top, left, width, height),
                    obj_name
                )
            )
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        print(len(l_obj_info))
        print(*l_obj_info, sep='\n')
        print("\nFrame Number=", frame_number,
              "\nNumber of Objects=", num_rects,
              "\nVehicle_count=", obj_counter[PGIE_CLASS_ID_VEHICLE],
              "\nPerson_count=", obj_counter[PGIE_CLASS_ID_PERSON]
              )

        stream_fps['stream0'].get_fps()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def main():
    pgie_config = os.path.join('..', 'configs', 'pgie_primary_detector.txt')
    pgie_batch = 1
    batched_push_timeout = 10
    attach_sys_ts = 1
    width, height = 1920, 1080

    args = utils.parse_arguments()
    if args.v:
        utils.set_logging()

    rtsp_source = f'rtsp://{args.ip}:{args.port}/{args.name}'

    Gst.debug_set_active(bool(args.debug_level))
    Gst.debug_set_default_threshold(args.debug_level)
    GObject.threads_init()
    Gst.init(None)

    stream_fps.update({'stream0': nvutils.GetFPS(stream_id=0,
                                                 seconds=5,
                                                 save_log=False,
                                                 model_name=__file__,
                                                 rtsp_fps=30,
                                                 )
                       }
                      )

    pipeline = Gst.Pipeline.new('rtsp_client')
    rtsp_bin = gsw.RTSPBin(builder_id=0, location=rtsp_source, compression=args.codec)

    streammux = Gst.ElementFactory.make("nvstreammux", "stream_muxer")
    streammux.set_property('live-source', 1)
    streammux.set_property('width', width)
    streammux.set_property('height', height)
    streammux.set_property('batch-size', pgie_batch)
    streammux.set_property('batched-push-timeout', batched_push_timeout)
    streammux.set_property('attach-sys-ts', attach_sys_ts)

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', pgie_config)
    pgie.set_property("batch-size", pgie_batch)

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvidia_convertor")

    capsfilter = Gst.ElementFactory.make("capsfilter", "filter")
    capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=(string)RGBA"))

    if args.d:
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if nvutils.is_aarch64():
            transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        sink.set_property('sync', 1)
    else:
        sink = Gst.ElementFactory.make("fakesink", 'fake_sink')

    pipeline.add(rtsp_bin.rtspsrc)
    if nvutils.is_aarch64():
        pipeline.add(rtsp_bin.depayer)
        pipeline.add(rtsp_bin.parser)
    pipeline.add(rtsp_bin.decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)

    if args.d:
        pipeline.add(capsfilter)
        pipeline.add(nvosd)
        if nvutils.is_aarch64():
            pipeline.add(transform)
    pipeline.add(sink)

    if nvutils.is_aarch64():
        streammux_sinkpad = streammux.get_request_pad('sink_%u' % 0)
        decoder_srcpad = rtsp_bin.decoder.get_static_pad("src")
        decoder_srcpad.link(streammux_sinkpad)
    else:
        streammux_handler = gsw.StreamMuxHandler(next_element=streammux, index=0)
        rtsp_bin.decoder.connect("pad-added", streammux_handler.on_pad_added)
        rtsp_bin.decoder.connect("pad-removed", streammux_handler.on_pad_removed)

    logging.info("Linking elements in the Pipeline")
    if nvutils.is_aarch64():
        rtsp_bin.depayer.link(rtsp_bin.parser)
        rtsp_bin.parser.link(rtsp_bin.decoder)

    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(capsfilter)

    if args.d:
        capsfilter.link(nvosd)
        if nvutils.is_aarch64():
            nvosd.link(transform)
            transform.link(sink)
        else:
            nvosd.link(sink)
    else:
        capsfilter.link(sink)

    pgie_src_pad = pgie.get_static_pad("src")
    pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_buffer_probe, 0)

    rtsp_handler = gsw.RTSPHandler(pipeline=pipeline,
                                   loop=GObject.MainLoop(),
                                   basic_blocks={0: rtsp_bin},
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
