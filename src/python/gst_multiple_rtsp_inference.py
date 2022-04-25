import os
import sys
import gi
import logging
import numpy as np

gi.require_version('Gst', '1.0')
gi.require_version('GstRtsp', '1.0')
from gi.repository import GObject, Gst, GstRtsp
from common import nvutils
from common import gstreamer_wrappers as gsw
from common import utils

# Path for pyds library
sys.path.append(os.path.join(os.getcwd(), 'models', 'deep_stream'))
sys.path.append(os.path.join('/', 'opt', 'nvidia', 'deepstream', 'deepstream', 'lib'))
import pyds

MODELS_CLASSES = {'pgie': {'names': ["Person", "Bag", "Face"],
                           'counter': {i: 0 for i in range(3)}
                           },

                  'sgie0': {'names': ["Car", "TwoWheeler", "Person", "RoadSign"],
                            'counter': {i: 0 for i in range(4)}
                            },
                  }

stream_fps = {}


def det_buffer_probe(model_name):
    def _buffer_probe(pad, info, u_data):
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

            obj_counter = MODELS_CLASSES[model_name]['counter']
            for k in obj_counter.keys():
                obj_counter[k] = 0

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
                obj_name = MODELS_CLASSES[model_name]['names'][obj_meta.class_id]
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

            # print(len(l_obj_info))
            classes_info = [
                f"""\t{MODELS_CLASSES[model_name]['names'][cls_id]}: {obj_counter[cls_id]}"""
                for cls_id in obj_counter.keys()
            ]
            classes_info = '\n'.join(classes_info)

            print("\nFrame Number=", frame_number,
                  f"\nModel: {model_name}\n{classes_info}",
                  "\nNumber of Objects=", num_rects,
                  )
            print(*l_obj_info, sep='\n')

            if model_name == 'pgie':
                stream_fps["stream{0}".format(frame_meta.pad_index)].get_fps()

            # Get frame rate through this probe
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    return _buffer_probe


def main():
    args = utils.parse_arguments()
    if args.v:
        utils.set_logging()

    rtsp_sources = [f'rtsp://{args.ip}:{args.port}/{name}' for name in args.name]
    print(rtsp_sources)

    stream_multiplier = 1
    width, height = 1920, 1080
    batched_push_timeout = 10
    attach_sys_ts = 1
    n_sources = len(rtsp_sources) * stream_multiplier
    tiler_rows = int(n_sources ** 0.5)
    tiler_columns = int(np.math.ceil((1.0 * n_sources) / tiler_rows))

    # Models configs
    config_folder = os.path.join('..', 'configs')

    models_config = {1: os.path.join(config_folder, 'pgie_peoplenet.txt'),
                     2: os.path.join(config_folder, 'pgie_dashcamnet_no_person.txt'),
                     }
    n_sources = len(rtsp_sources)
    models_batch = {1: n_sources,
                    2: 1,
                    }

    # Tracker
    tracker_algs = ('libnvds_nvdcf', 'libnvds_mot_iou', 'libnvds_mot_klt')
    tracker = tracker_algs[0]
    tracker_config = os.path.join(config_folder, 'tracker_config.yml')
    tracker_wh = 1024
    tracker_batch_process = 1
    tracker_display_id = 1
    tracker_path = os.path.join('/', 'opt', 'nvidia', 'deepstream', 'deepstream', 'lib', f'{tracker}.so')

    Gst.debug_set_active(bool(args.debug_level))
    Gst.debug_set_default_threshold(args.debug_level)
    GObject.threads_init()
    Gst.init(None)

    # Init GStreamer pipeline
    pipeline = Gst.Pipeline.new('rtsp_client')

    # Init streammuxer
    streammux = Gst.ElementFactory.make("nvstreammux", "stream_muxer")
    streammux.set_property('live-source', 1)

    streammux.set_property('width', width)
    streammux.set_property('height', height)
    streammux.set_property('batch-size', n_sources)
    streammux.set_property('batched-push-timeout', batched_push_timeout)
    streammux.set_property('attach-sys-ts', attach_sys_ts)
    pipeline.add(streammux)

    # Init multiple RTSP sources
    rtsp_blocks = {}
    for i in range(n_sources):

        rtsp_bin = gsw.RTSPBin(builder_id=i, location=rtsp_sources[i], compression=args.codec)
        rtsp_blocks.update({i: rtsp_bin})

        stream_fps.update({f'stream{i}': nvutils.GetFPS(stream_id=i,
                                                        seconds=5,
                                                        save_log=False,
                                                        model_name=__file__,
                                                        rtsp_fps=30,
                                                        )
                           }
                          )

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

    # Primary GPU inference engine - pgie
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', models_config.get(1))
    pgie.set_property("batch-size", models_batch.get(1))

    # Secondary GPU inference engine - sgie0
    sgie0 = Gst.ElementFactory.make("nvinfer", "secondary-gie0")
    sgie0.set_property('config-file-path', models_config.get(2))
    sgie0.set_property("batch-size", models_batch.get(2))

    # Object tracking
    nvtracker = Gst.ElementFactory.make("nvtracker", "nvtracker0")
    nvtracker.set_property('tracker-width', tracker_wh)
    nvtracker.set_property('tracker-height', tracker_wh)
    nvtracker.set_property('ll-lib-file', tracker_path)
    nvtracker.set_property('ll-config-file', tracker_config)
    nvtracker.set_property('enable-batch-process', tracker_batch_process)
    nvtracker.set_property('display-tracking-id', tracker_display_id)

    # Combine streams to one array
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", width)
    tiler.set_property("height", height)

    # nvvidconv convertor
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvidia_convertor")

    # Add capsfilter
    capsfilter = Gst.ElementFactory.make("capsfilter", "filter")
    capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=(string)RGBA"))

    pipeline.add(pgie)
    pipeline.add(sgie0)
    pipeline.add(nvtracker)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(capsfilter)

    if args.d:
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        pipeline.add(nvosd)
        if nvutils.is_aarch64():
            transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
            pipeline.add(transform)
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        sink.set_property('sync', 1)
    else:
        sink = Gst.ElementFactory.make("fakesink", 'fake_sink')
    pipeline.add(sink)

    logging.info("Linking elements in the Pipeline")

    streammux.link(pgie)
    pgie.link(sgie0)
    sgie0.link(nvtracker)
    nvtracker.link(tiler)
    tiler.link(nvvidconv)
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

    # Probe for inference description
    pgie_src_pad = pgie.get_static_pad("src")
    pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, det_buffer_probe('pgie'), 0)

    # Init RTSPHandler
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
