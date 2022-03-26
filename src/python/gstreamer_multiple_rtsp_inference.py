import os
import sys
import gi
import numpy as np

gi.require_version('Gst', '1.0')
gi.require_version('GstRtsp', '1.0')
from gi.repository import GObject, Gst, GstRtsp
from common import nvutils
from common import gstreamer_wrappers as gsw

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


if __name__ == '__main__':

    """
    Note:
    Режим отображения видеопотоков DISPLAY:

    Плагин streammuxer для режима отображения DISPLAY требует nvmultistreamtiler, 
    что работает с плагином отображения с nveglglessink. nveglglessink требует для работы плагин nvegltransform.
    Блоки nveglglessink, nvegltransform вызывают Segmentation Fault при потере всех потоков и попытке перепокдлючения
    через функционал GStreamer.

    Вариант обходного решения:
    `check_tcp`-метод, блокирующий операции GStreamer до момента, когда TCP-соединение не восстановится. 

    Когда TCP-соединение становится доступным, GstBus доставляет 
    сообщение GstMessage: EOS, общий GStreamer-пайплайн восстанавливается.

    Тем самым не нужно пересоздавать GStreamer-пайплайн и перечитывать заново TensorRT-модели.
    Работает только в случае общих перебоев в сети. Случай перебоев с конкретными камерами обработка не затрагивает.
    """

    DISPLAY = True
    IP = '127.0.0.1'
    PORT = 8554
    CODEC = 'h264'
    RTSP_SRCS = [f'rtsp://{IP}:{PORT}/stream',
                 f'rtsp://{IP}:{PORT}/stream2']
    WIDTH, HEIGHT = 1920, 1080
    CONFIG_FOLDER = 'configs'
    MODELS_CONFIG = {1: os.path.join(CONFIG_FOLDER, 'pgie_peoplenet.txt'),
                     2: os.path.join(CONFIG_FOLDER, 'pgie_dashcamnet_no_person.txt'),
                     3: os.path.join(CONFIG_FOLDER, 'sgie_lpdnet.txt'),
                     }
    stream_multiplier = 1
    n_sources = len(RTSP_SRCS) * stream_multiplier
    MODELS_BATCH = {1: n_sources,
                    2: 1,
                    3: 1,
                    }
    BATCHED_PUSH_TIMEOUT = 10
    ATTACH_SYS_TS = 1
    TRACKER_ALG = 'libnvds_nvdcf'  # Options: 'libnvds_nvdcf', 'libnvds_mot_iou', 'libnvds_mot_klt'
    TRACKER_CONFIG = os.path.join(CONFIG_FOLDER, 'tracker_config.yml')
    TRACKER_WH = 1024
    TRACKER_PATH = os.path.join('/', 'opt', 'nvidia', 'deepstream', 'deepstream', 'lib', f'{TRACKER_ALG}.so')

    tiler_rows = int(n_sources ** 0.5)
    tiler_columns = int(np.math.ceil((1.0 * n_sources) / tiler_rows))

    # GStreamer initialization
    # Gst.debug_set_active(True)
    # Debug levels: 0-7
    # Gst.debug_set_default_threshold(2)
    GObject.threads_init()
    Gst.init(None)

    # Init GStreamer pipeline
    pipeline = Gst.Pipeline.new('rtsp_client')

    # Init streammuxer
    streammux = Gst.ElementFactory.make("nvstreammux", "stream_muxer")
    streammux.set_property('live-source', 1)

    streammux.set_property('width', WIDTH)
    streammux.set_property('height', HEIGHT)
    streammux.set_property('batch-size', n_sources)
    streammux.set_property('batched-push-timeout', BATCHED_PUSH_TIMEOUT)
    streammux.set_property('attach-sys-ts', ATTACH_SYS_TS)
    pipeline.add(streammux)

    # Init multiple RTSP sources
    rtsp_blocks = {}
    for i, loc in enumerate(RTSP_SRCS):

        rtsp_bin = gsw.RTSPBin(builder_id=i, location=loc, compression=CODEC)
        rtsp_blocks.update({i: rtsp_bin})
        streammux_sinkpad = streammux.get_request_pad(f'sink_{i}')

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
            decoder_srcpad.link(streammux_sinkpad)
        else:
            pipeline.add(rtsp_bin.rtspsrc)
            pipeline.add(rtsp_bin.decoder)

            decodebin_handler = gsw.DecodeBinHandler(sink_pad=streammux_sinkpad)
            rtsp_bin.decoder.connect("pad-added", decodebin_handler.decodebin_pad_added)
            rtsp_bin.decoder.connect("pad-removed", decodebin_handler.decodebin_pad_removed)

        # Primary GPU inference engine - pgie
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        pgie.set_property('config-file-path', MODELS_CONFIG[1])
        pgie.set_property("batch-size", MODELS_BATCH[1])

        # Secondary GPU inference engine - sgie0
        sgie0 = Gst.ElementFactory.make("nvinfer", "secondary-gie0")
        sgie0.set_property('config-file-path', MODELS_CONFIG[2])
        sgie0.set_property("batch-size", MODELS_BATCH[2])

        # Secondary GPU inference engine - sgie1
        sgie1 = Gst.ElementFactory.make("nvinfer", "secondary-gie1")
        sgie1.set_property('config-file-path', MODELS_CONFIG[3])
        sgie1.set_property("batch-size", MODELS_BATCH[3])

        # Object tracking
        nvtracker = Gst.ElementFactory.make("nvtracker", "nvtracker0")
        nvtracker.set_property('tracker-width', TRACKER_WH)
        nvtracker.set_property('tracker-height', TRACKER_WH)
        nvtracker.set_property('ll-lib-file', TRACKER_PATH)
        nvtracker.set_property('enable-batch-process', 1)
        nvtracker.set_property('display-tracking-id', 1)

        # Combine streams to one array
        tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        tiler.set_property("rows", tiler_rows)
        tiler.set_property("columns", tiler_columns)
        tiler.set_property("width", WIDTH)
        tiler.set_property("height", HEIGHT)

        # nvvidconv convertor
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvidia_convertor")

        # Add capsfilter
        capsfilter = Gst.ElementFactory.make("capsfilter", "filter")
        capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=(string)RGBA"))

        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

        pipeline.add(pgie)
        pipeline.add(sgie0)
        pipeline.add(sgie1)
        pipeline.add(nvtracker)
        pipeline.add(tiler)
        pipeline.add(nvvidconv)
        pipeline.add(capsfilter)
        pipeline.add(nvosd)

        if DISPLAY:
            if nvutils.is_aarch64():
                transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
                pipeline.add(transform)
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
            sink.set_property('sync', 1)
        else:
            sink = Gst.ElementFactory.make("fakesink", 'fake_sink')
        pipeline.add(sink)

        # Link all elements in pipeline except source which has a dynamic source pad'''
        print("Linking elements in the Pipeline \n")

        streammux.link(pgie)
        pgie.link(sgie0)
        sgie0.link(sgie1)
        sgie1.link(nvtracker)
        nvtracker.link(tiler)
        tiler.link(nvvidconv)
        nvvidconv.link(capsfilter)

        if DISPLAY:
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
                                       muxer=streammux,
                                       **{'rtsp': {'addr': IP, 'port': PORT}}
                                       )
    try:
        rtsp_handler.loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)
    del pipeline
