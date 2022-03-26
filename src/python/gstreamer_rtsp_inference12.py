import os
import sys
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

MODELS_CLASSES = {1: ("Vehicle", "TwoWheeler", "Person", "RoadSign"),
                  2: "Licence",
                  }

stream_fps = {}


def buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    # https://docs.nvidia.com/metropolis/deepstream/sdk-api/struct__NvDsFrameMeta.html#aed8c9ecaad8faecef490341c84c9133f
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    for frame_meta in gsw.object_generator(batch_meta.frame_meta_list, pyds.NvDsFrameMeta.cast):

        # Holds the current frame number of the source.
        frame_number = frame_meta.frame_num
        # Holds the ntp timestamp.
        ntp_timestamp = frame_meta.ntp_timestamp
        # Holds the number of object meta elements attached to current frame.
        num_rects = frame_meta.num_obj_meta
        # Holds the source IDof the frame in the batch, e.g. the camera ID. It need not be in sequential order.
        source_id = frame_meta.source_id
        # Holds the pad or port index of the Gst-streammux plugin for the frame in the batch.
        pad_index = frame_meta.pad_index
        # Holds a pointer to a list of pointers of type NvDsObjectMeta in use for the frame.
        l_obj = frame_meta.obj_meta_list
        # Holds the location of the frame in the batch.
        batch_id = frame_meta.batch_id

        frame = pyds.get_nvds_buf_surface(hash(gst_buffer), batch_id)

        for obj_meta in gsw.object_generator(l_obj, pyds.NvDsObjectMeta.cast):

            # Holds the index of the object class inferred by the primary detector/classifier.
            obj_class_id = obj_meta.class_id
            # Holds a confidence value for the object, set by the inference component.
            # confidence will be set to -0.1, if "Group Rectangles" mode of clustering is chosen
            # since the algorithm does not preserve confidence values.
            # Also, for objects found by tracker and not inference component, confidence will be set to -0.1
            obj_confidence = obj_meta.confidence
            # Holds a string describing the class of the detected object.
            obj_label = obj_meta.obj_label
            # Holds a unique ID for tracking the object.
            object_id = obj_meta.object_id
            # Holds a structure containing bounding box coordinates of the object when processed by tracker.
            tracker_bbox_info = obj_meta.tracker_bbox_info
            # Holds a pointer to a list of pointers of type NvDsClassifierMeta.
            classifier_meta_list = obj_meta.classifier_meta_list
            # Holds a unique component ID that identifies the metadata in this structure.
            model_uid = obj_meta.unique_component_id
            user_meta_list = obj_meta.obj_user_meta_list

            rect_params = obj_meta.rect_params
            top = int(rect_params.top)
            left = int(rect_params.left)
            width = int(rect_params.width)
            height = int(rect_params.height)

            if model_uid == 1:
                # Init obj_info dictionary
                obj_info = dict()
                obj_info.update({'frame_number': frame_number,
                                 'ntp_timestamp': ntp_timestamp,
                                 'source_id': source_id,
                                 'batch_id': batch_id,
                                 'pad_index': pad_index,
                                 'model_uid': model_uid,
                                 'detection_confidence': obj_confidence,
                                 'detection_class_id': obj_class_id,
                                 'detection_class_label': obj_label,
                                 'bbox': (top, left, width, height),
                                 'obj_id': object_id,
                                 'secondary_models': {},
                                 }
                                )
            else:
                # Add info about secondary detectors
                obj_info['secondary_models'].update({model_uid: {'detection_confidence': obj_confidence,
                                                                 'detection_class_id': obj_class_id,
                                                                 'detection_class_label': obj_label,
                                                                 'bbox': (top, left, width, height),
                                                                 'obj_id': object_id,
                                                                 'secondary_models': {},
                                                                 }
                                                     }
                                                    )

            if classifier_meta_list is not None:
                for cls_meta in gsw.object_generator(classifier_meta_list, pyds.NvDsClassifierMeta.cast):
                    if cls_meta.label_info_list is not None:
                        secondary_dict = {'cls_confidence': [],
                                          'cls_class_id': [],
                                          }
                        for meta in gsw.object_generator(cls_meta.label_info_list, pyds.NvDsLabelInfo.cast):
                            secondary_dict['cls_confidence'].append(meta.result_prob)
                            secondary_dict['cls_class_id'].append(meta.result_class_id)

                        obj_info['secondary_models'].update({cls_meta.unique_component_id: secondary_dict})

            print('\n', obj_info)

    return Gst.PadProbeReturn.OK


if __name__ == '__main__':

    """
    Combined inference with:
        1.PrimaryModel
        2.LPDNet model
        3.LPRNet model
        4.Object tracker
          https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html#low-level-tracker-library-comparisons-and-tradeoffs
    """

    DISPLAY = True
    IP = '127.0.0.1'
    PORT = 8554
    CODEC = "h264"
    RTSP_REC_SRC = f'rtsp://{IP}:{PORT}/stream'
    CONFIG_FOLDER = 'configs'
    MODELS_CONFIG = {1: os.path.join(CONFIG_FOLDER, 'pgie_det_config.txt'),
                     2: os.path.join(CONFIG_FOLDER, 'sgie_lpdnet.txt'),
                     3: os.path.join(CONFIG_FOLDER, 'sgie_custom_lpr.txt'),
                     }
    MODELS_BATCH = {1: 1,
                    2: 1,
                    3: 1,
                    }
    TRACKER_ALG = ['libnvds_nvdcf', 'libnvds_mot_iou', 'libnvds_mot_klt'][0]
    TRACKER_CONFIG = os.path.join(CONFIG_FOLDER, 'tracker_config.yml')
    BATCHED_PUSH_TIMEOUT = 10
    ATTACH_SYS_TS = 1
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
    mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)

    # GStreamer initialization
    # Gst.debug_set_active(True)
    # Debug levels: 0-7
    # Gst.debug_set_default_threshold(2)
    GObject.threads_init()
    Gst.init(None)

    # Gstreamer pipeline init
    pipeline = Gst.Pipeline.new('rtsp_client')

    # Set RTSP plugins container
    rtsp_bin = gsw.RTSPBin(builder_id=0, location=RTSP_REC_SRC, compression=CODEC)

    # Init streammuxer
    streammux = Gst.ElementFactory.make("nvstreammux", "stream_muxer")
    streammux.set_property('live-source', 1)
    streammux.set_property('width', WIDTH)
    streammux.set_property('height', HEIGHT)
    streammux.set_property('batch-size', MODELS_BATCH[1])
    streammux.set_property('batched-push-timeout', BATCHED_PUSH_TIMEOUT)
    streammux.set_property('attach-sys-ts', ATTACH_SYS_TS)

    # Primary GPU inference engine. UID=1
    pgie = Gst.ElementFactory.make("nvinfer", "primary_inference")
    pgie.set_property('config-file-path', MODELS_CONFIG[1])
    pgie.set_property("batch-size", MODELS_BATCH[1])

    # Secondary GPU inference engine. UID=2
    sgie2 = Gst.ElementFactory.make("nvinfer", "secondary_gie_2_lpdet")
    sgie2.set_property('config-file-path', MODELS_CONFIG[2])
    sgie2.set_property("batch-size", MODELS_BATCH[2])

    # Secondary GPU inference engine. UID=3
    #sgie3 = Gst.ElementFactory.make("nvinfer", "secondary_gie_3_lpr")
    #sgie3.set_property('config-file-path', MODELS_CONFIG[3])
    #sgie3.set_property("batch-size", MODELS_BATCH[3])

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
    nvvidconv.set_property("nvbuf-memory-type", mem_type)

    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    # Add capsfilter with RGBA for pgie_buffer_probe
    capsfilter = Gst.ElementFactory.make("capsfilter", "filter")
    capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=(string)RGBA"))

    if DISPLAY:
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
    pipeline.add(sgie2)
    #pipeline.add(sgie3)
    pipeline.add(nvtracker)
    pipeline.add(nvvidconv)
    pipeline.add(capsfilter)

    if DISPLAY:
        pipeline.add(nvosd)
        if nvutils.is_aarch64():
            pipeline.add(transform)
    pipeline.add(sink)

    # Create sink pad for streammux with a specified id and connect with decoder srcpad
    streammux_sinkpad = streammux.get_request_pad('sink_%u' % 0)

    if nvutils.is_aarch64():
        decoder_srcpad = rtsp_bin.decoder.get_static_pad("src")
        decoder_srcpad.link(streammux_sinkpad)
    else:
        decodebin_handler = gsw.DecodeBinHandler(sink_pad=streammux_sinkpad)
        rtsp_bin.decoder.connect("pad-added", decodebin_handler.decodebin_pad_added)
        rtsp_bin.decoder.connect("pad-removed", decodebin_handler.decodebin_pad_removed)

    # Link all elements in pipeline except source which has a dynamic source pad'''
    print("Linking elements in the Pipeline \n")
    if nvutils.is_aarch64():
        rtsp_bin.depayer.link(rtsp_bin.parser)
        rtsp_bin.parser.link(rtsp_bin.decoder)
    streammux.link(pgie)
    pgie.link(nvtracker)
    nvtracker.link(sgie2)
    #sgie2.link(sgie3)
    #sgie3.link(nvvidconv)
    sgie2.link(nvvidconv)

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

    sinkpad = sink.get_static_pad("sink")
    if not sinkpad:
        sys.stderr.write(" Unable to get sink pad of sink \n")
    sinkpad.add_probe(Gst.PadProbeType.BUFFER, buffer_probe, 0)

    # Init RTSPHandler
    rtsp_handler = gsw.RTSPHandler(pipeline=pipeline,
                                   loop=GObject.MainLoop(),
                                   basic_blocks={0: rtsp_bin},
                                   **{'rtsp': {'addr': IP, 'port': PORT}}
                                   )

    try:
        rtsp_handler.loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)
    del pipeline
