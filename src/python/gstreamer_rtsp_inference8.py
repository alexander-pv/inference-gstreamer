import os
import socket
import sys
import time

import gi

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
                stream_fps['stream0'].get_fps()

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    return _buffer_probe


if __name__ == '__main__':

    """
    Combined inference with:
        1.TLT-pretrained PeopleNet model as primary GIE.
        2.TLT-pretrained DashCamNet model (without person class) as secondary GIE.
        3.Object tracker
          https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#low-level-tracker-library-comparisons-and-tradeoffs
    """

    DISPLAY = False
    IP = '10.42.0.1'
    PORT = 8554
    CODEC = "h264"
    RTSP_REC_SRC = f'rtsp://{IP}:{PORT}/stream'
    PGIE_CONFIG = r'configs/pgie_peoplenet.txt'
    SGIE0_CONFIG = r'configs/pgie_dashcamnet_no_person.txt'
    PGIE_BATCH = 1
    TRACKER_ALG = 'libnvds_nvdcf'  # Options: 'libnvds_nvdcf', 'libnvds_mot_iou', 'libnvds_mot_klt'
    TRACKER_CONFIG = r'configs/tracker_config.yml'
    MUXER_BATCH = 1
    WIDTH, HEIGHT = 1920, 1080
    TRACKER_WH = 1024

    stream_fps.update({'stream0': nvutils.GetFPS(stream_id=0,
                                                 seconds=5,
                                                 save_log=False,
                                                 model_name=__file__,
                                                 rtsp_fps=30,
                                                 )
                       }
                      )

    # GStreamer initialization
    # Gst.debug_set_active(True)
    # Debug levels: 0-7
    # Gst.debug_set_default_threshold(2)
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

    # Primary GPU inference engine - pgie
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', PGIE_CONFIG)
    pgie.set_property("batch-size", PGIE_BATCH)

    # Secondary GPU inference engine - sgie0
    sgie0 = Gst.ElementFactory.make("nvinfer", "secondary-gie0")
    sgie0.set_property('config-file-path', SGIE0_CONFIG)
    sgie0.set_property("batch-size", PGIE_BATCH)

    # Object tracking
    nvtracker = Gst.ElementFactory.make("nvtracker", "nvtracker0")
    nvtracker.set_property('tracker-width', TRACKER_WH)
    nvtracker.set_property('tracker-height', TRACKER_WH)
    nvtracker.set_property('ll-lib-file', f'/opt/nvidia/deepstream/deepstream/lib/{TRACKER_ALG}.so')
    nvtracker.set_property('enable-batch-process', 1)
    nvtracker.set_property('display-tracking-id', 1)

    if TRACKER_ALG == 'libnvds_nvdcf':
        nvtracker.set_property('ll-config-file', TRACKER_CONFIG)

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvidia_convertor")

    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    # Add capsfilter
    capsfilter = Gst.ElementFactory.make("capsfilter", "filter")
    capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=(string)RGBA"))

    if DISPLAY:
        # nv3dsink works with NVMM buffers and renders using the 3D graphics rendering API.
        # It performs better than nveglglessink with NVMM buffers.
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        sink.set_property('sync', 1)
    else:
        sink = Gst.ElementFactory.make("fakesink", 'fake_sink')

    pipeline.add(rtspsrc)
    pipeline.add(rtph26xdepay)
    pipeline.add(h26xparser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(sgie0)
    pipeline.add(nvtracker)
    pipeline.add(nvvidconv)
    if DISPLAY:
        pipeline.add(capsfilter)
        pipeline.add(nvosd)
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
    pgie.link(sgie0)
    sgie0.link(nvtracker)
    nvtracker.link(nvvidconv)
    if DISPLAY:
        nvvidconv.link(capsfilter)
        capsfilter.link(nvosd)
        nvosd.link(transform)
        capsfilter.link(transform)
        transform.link(sink)
    else:
        nvvidconv.link(sink)

    # Probe for inference description
    pgie_src_pad = pgie.get_static_pad("src")
    pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, det_buffer_probe('pgie'), 0)

    sgie0_src_pad = sgie0.get_static_pad("src")
    sgie0_src_pad.add_probe(Gst.PadProbeType.BUFFER, det_buffer_probe('sgie0'), 0)

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
