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

stream_fps = {}

COLORS = [[128, 128, 64], [0, 0, 128], [0, 128, 128], [128, 0, 0],
          [128, 0, 128], [128, 128, 0], [0, 128, 0], [0, 0, 64],
          [0, 0, 192], [0, 128, 64], [0, 128, 192], [128, 0, 64],
          [128, 0, 192], [128, 128, 128]]


def map_mask_as_display_bgr(mask):
    """ Assigning multiple colors as image output using the information
        contained in mask. (BGR is opencv standard.)
    """
    # getting a list of available classes
    m_list = list(set(mask.flatten()))

    shp = mask.shape
    bgr = np.zeros((shp[0], shp[1], 3))
    for idx in m_list:
        bgr[mask == idx] = COLORS[idx]
    return bgr


def seg_src_pad_buffer_probe(pad, info, u_data):
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
        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                # The casting is done by pyds.NvDsUserMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                seg_user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break
            if seg_user_meta and seg_user_meta.base_meta.meta_type == \
                    pyds.NVDSINFER_SEGMENTATION_META:
                try:
                    # Note that seg_user_meta.user_meta_data needs a cast to
                    # pyds.NvDsInferSegmentationMeta
                    # The casting is done by pyds.NvDsInferSegmentationMeta.cast()
                    # The casting also keeps ownership of the underlying memory
                    # in the C code, so the Python garbage collector will leave
                    # it alone.

                    # segmeta = pyds.NvDsInferSegmentationMeta.cast(seg_user_meta.user_meta_data)
                    pass
                except StopIteration:
                    break
                # Retrieve mask data in the numpy format from segmeta
                # Note that pyds.get_segmentation_masks() expects object of
                # type NvDsInferSegmentationMeta
                # masks = pyds.get_segmentation_masks(segmeta)
                # masks = np.array(masks, copy=True, order='C')
                # map the obtained masks to colors of 2 classes.
                # frame_image = map_mask_as_display_bgr(masks)
                # cv2.imwrite(folder_name + "/" + str(frame_number) + ".jpg", frame_image)
            try:
                l_user = l_user.next
            except StopIteration:
                break

        stream_fps['stream0'].get_fps()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


if __name__ == '__main__':

    """
    Inference with semantic segmentation model
    """

    DISPLAY = True
    IP = '127.0.0.1'
    PORT = 8554
    CODEC = "h264"
    RTSP_REC_SRC = f'rtsp://{IP}:{PORT}/stream'
    PGIE_CONFIG = os.path.join('configs', 'pgie_seg_config_semantic.txt')
    PGIE_BATCH = 1
    BATCHED_PUSH_TIMEOUT = 10
    ATTACH_SYS_TS = 1
    WIDTH, HEIGHT = 1920, 1080

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

    # Set RTSP plugins container
    rtsp_bin = gsw.RTSPBin(builder_id=0, location=RTSP_REC_SRC, compression=CODEC)

    # Init streammuxer
    streammux = Gst.ElementFactory.make("nvstreammux", "stream_muxer")
    streammux.set_property('live-source', 1)
    streammux.set_property('width', WIDTH)
    streammux.set_property('height', HEIGHT)
    streammux.set_property('batch-size', PGIE_BATCH)
    streammux.set_property('batched-push-timeout', BATCHED_PUSH_TIMEOUT)
    streammux.set_property('attach-sys-ts', ATTACH_SYS_TS)

    # Primary GPU inference engine
    pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference")
    pgie.set_property('config-file-path', PGIE_CONFIG)
    pgie.set_property("batch-size", PGIE_BATCH)

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvidia_convertor")

    # Add capsfilter
    capsfilter = Gst.ElementFactory.make("capsfilter", "filter")
    capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=(string)RGBA"))

    if DISPLAY:
        nvosd = Gst.ElementFactory.make("nvsegvisual", "nvsegvisual")
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
    if DISPLAY:
        pipeline.add(capsfilter)
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
    pgie.link(nvvidconv)

    if DISPLAY:
        nvvidconv.link(nvosd)
        if nvutils.is_aarch64():
            nvosd.link(transform)
            transform.link(sink)
        else:
            nvosd.link(sink)
    else:
        capsfilter.link(sink)

    # Probe for inference description
    #pgie_src_pad = pgie.get_static_pad("src")
    #pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, seg_src_pad_buffer_probe, 0)

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
