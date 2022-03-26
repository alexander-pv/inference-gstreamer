import sys
import gi
import os
import cv2
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
import pyds
from common import nvutils
from common import gstreamer_wrappers as gsw


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

                primary_obj_id = obj_info['obj_id']
                secondary_obj_id = obj_info['secondary_models'][2]['obj_id']
                side = "left" if (left + 0.5*width) < 1920/2 else "right"
                nvutils.save_img(img=frame[top:top + height, left:left + width, :],
                                 path='./temp',
                                 postfix=f"{frame_number}_{primary_obj_id}_{side}")

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


def main(imgs_path: str, display: bool) -> None:
    """
    Run Gstreamer pipeline
    :param imgs_path: str, path to images folder
    :param display:  bool, display images
    :return: None
    """

    config_fodler = 'configs'
    models_config = {1: os.path.join(config_fodler, 'pgie_det_config.txt'),
                     2: os.path.join(config_fodler, 'sgie_lpdnet.txt'),
                     }
    models_batch = {1: 1, 2: 1}
    width, height = 1920, 1080

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)
    Gst.debug_set_active(True)
    # Debug levels: 0-7
    Gst.debug_set_default_threshold(3)

    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)

    source = Gst.ElementFactory.make("multifilesrc")
    source.set_property('location', os.path.join(imgs_path, "img_%05d.jpg"))
    source.set_property('index', 0)

    jpegparser = Gst.ElementFactory.make("jpegparse", "jpegdec")

    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")

    # Primary GPU inference engine. UID=1
    pgie = Gst.ElementFactory.make("nvinfer", "primary_inference")
    pgie.set_property('config-file-path', models_config[1])
    pgie.set_property("batch-size", models_batch[1])

    # Secondary GPU inference engine. UID=2
    sgie2 = Gst.ElementFactory.make("nvinfer", "secondary_gie_2_lpdet")
    sgie2.set_property('config-file-path', models_config[2])
    sgie2.set_property("batch-size", models_batch[2])

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvvidconv.set_property("nvbuf-memory-type", mem_type)

    # Add capsfilter with RGBA for pgie_buffer_probe
    capsfilter = Gst.ElementFactory.make("capsfilter", "filter")
    capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=(string)RGBA"))

    if display:
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        if nvutils.is_aarch64():
            transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    else:
        sink = Gst.ElementFactory.make("fakesink", 'fake_sink')

    streammux.set_property('width', width)
    streammux.set_property('height', height)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)

    pipeline.add(source)
    pipeline.add(jpegparser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(sgie2)
    pipeline.add(nvvidconv)
    pipeline.add(capsfilter)

    if display:
        pipeline.add(nvosd)
        if nvutils.is_aarch64():
            pipeline.add(transform)
    pipeline.add(sink)

    # we link the elements together
    # file-source -> jpeg-parser -> nvh264-decoder ->
    # nvinfer -> nvvidconv -> nvosd -> video-renderer
    print("Linking elements in the Pipeline \n")
    source.link(jpegparser)
    jpegparser.link(decoder)

    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = decoder.get_static_pad("src")

    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(sgie2)
    sgie2.link(nvvidconv)
    nvvidconv.link(capsfilter)

    if display:
        capsfilter.link(nvosd)
        if nvutils.is_aarch64():
            nvosd.link(transform)
            transform.link(sink)
        else:
            nvosd.link(sink)
    else:
        capsfilter.link(sink)

    # Create an event loop and feed gstreamer bus messages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    # Lets add probe to get informed of the metadata generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    sinkpad = sink.get_static_pad("sink")
    sinkpad.add_probe(Gst.PadProbeType.BUFFER, buffer_probe, 0)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # Cleanup
    pipeline.set_state(Gst.State.NULL)
    del pipeline


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("No arguments was found. Args:\n" +
              "$1: images path. Example: {./temp}\n" +
              "$2: display or not. Example: {0}, {1}\n")
    else:
        sys.exit(main(imgs_path=sys.argv[1], display=bool(int(sys.argv[2]))))
