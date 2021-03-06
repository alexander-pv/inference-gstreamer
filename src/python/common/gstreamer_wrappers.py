import sys
import time
import logging
import cv2
import gi
import platform
from common import nvutils

gi.require_version('Gst', '1.0')
gi.require_version('GstRtsp', '1.0')
from gi.repository import Gst, GstRtsp

from abc import ABCMeta, abstractmethod


class RTSPBin:

    def __init__(self, builder_id: int, location: str, compression: str = 'h264',
                 retry: int = 25000, verbose: bool = True):
        """
        :param builder_id:  RTSP index, int
        :param location:    RTSP address with port and postfix, str
        :param compression: Video compression format
        :param retry:       rtspsrc number of retries
        :param verbose:
        """
        self.verbose = verbose
        self.builder_id = builder_id
        self.location = location
        self.compression = compression
        self.retry = retry

        self.available_compression = {'h264': {'depayer': 'rtph264depay',
                                               'parser': 'h264parse',
                                               },
                                      'h265': {'depayer': 'rtph265depay',
                                               'parser': 'h265parse',
                                               }
                                      }
        self.rtspsrc = None
        self.depayer = None
        self.parser = None
        self.decoder = None
        self.connect_plugin = None
        self.is_aarch64 = nvutils.is_aarch64()
        self.enable_max_performance = 1
        self.enable_frame_type_reporting = 0

        self._build()

    def _build(self):
        """
        Build basic GStreamer RTSP-block with elements:
                                if aarch64:
                                             RTSP packets reader    |src|->
                                    |sink|-> RTSP packets extractor |src|->
                                    |sink|-> Video parser           |src|->
                                    |sink|-> Video decoder          |src|->
                                else:
                                             RTSP packets reader    |src|->
                                    |sink|-> decodebin              |src|->
        :return: self
        """

        # Read RTSP packets
        cur_comp = self.available_compression[self.compression]
        self.rtspsrc = Gst.ElementFactory.make('rtspsrc', 'rtspsrc' + f'_{self.builder_id}')
        self.rtspsrc.set_property('location', self.location)
        self.rtspsrc.set_property('protocols', GstRtsp.RTSPLowerTrans.TCP)
        self.rtspsrc.set_property('retry', self.retry)
        if self.verbose:
            logging.info(f'Building for {platform.uname().machine}')

        if self.is_aarch64:
            # Extract video from RTSP
            self.depayer = Gst.ElementFactory.make(cur_comp['depayer'], cur_comp['depayer'] + f'_{self.builder_id}')

            # Parse video from compresion format
            self.parser = Gst.ElementFactory.make(cur_comp['parser'], cur_comp['parser'] + f'_{self.builder_id}')

            # Decode video stream via V4L2 API
            self.decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2decoder" + f'_{self.builder_id}')
            self.decoder.set_property("enable-max-performance", self.enable_max_performance)
            self.decoder.set_property("enable-frame-type-reporting", self.enable_frame_type_reporting)
            self.connect_plugin = self.depayer
        else:
            self.decoder = Gst.ElementFactory.make("decodebin", "decode_container" + f'_{self.builder_id}')
            self.connect_plugin = self.decoder


class RTSPHandler:
    def __init__(self, pipeline, loop, basic_blocks: dict, verbose: bool = True, **kwargs):
        """
        GStreamer rtsp handler with pipeline reconnection option
        :param pipeline:
        :param loop:
        :param basic_blocks:
        :param verbose:
        """

        self.pipeline = pipeline
        self.loop = loop
        self.basic_blocks = basic_blocks

        self.init_pipeline_callbacks()
        self.init_signal_watch()
        self.pipeline.set_state(Gst.State.PLAYING)
        self.kwargs = kwargs

        self.callback_delay = 5
        self.reconn_wait = 10
        self.get_state_wait = 5
        self.alive = True
        self.caught_eos = False
        self.check_flow_enabled = True
        self._rtspsrc_active = {}
        self._rtspsrc_flow_th = 0
        self._rtspsrc_flow_timer = 1e5

        self.verbose = verbose

    def init_pipeline_callbacks(self):
        """Note that source is an rtspsrc element which has a dynamically
        created source pad.  This means it can only be linked after the pad has
        been created.  Therefore, the linking is done with the callback function
        onPadAddedToRtspsrc(...):
        :return: None
        """
        for source_id in self.basic_blocks.keys():
            self.basic_blocks[source_id].rtspsrc.connect('pad-added', self.on_pad_added_to_rtspsrc)
            self.basic_blocks[source_id].rtspsrc.connect('pad-removed', self.on_pad_removed_from_rtspsrc)

    def init_signal_watch(self):
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus_watch_id = bus.connect("message", self.reconnection_callback, self.loop)

    def on_pad_added_to_rtspsrc(self, rtspsrc, pad):
        """
        This callback is required because rtspsrc elements have
        dynamically created pads.  So linking can only occur after a pad
        has been created.  Furthermore, only the rtspsrc for the currently
        selected camera is linked to the depayer.
        :param rtspsrc:
        :param pad:
        :return:
        """
        source_id = int(rtspsrc.get_name().split('_')[-1])
        if self.verbose:
            logging.info(f'Pad added to rtspsrc element. Source id: {source_id}')
        sink_pad = self.basic_blocks[source_id].connect_plugin.get_static_pad('sink')
        if not sink_pad.is_linked():
            pad.link(sink_pad)

    def on_pad_removed_from_rtspsrc(self, rtspsrc, pad):
        """
        Unlinks the rtspsrc element from the depayer
        :param rtspsrc:
        :param pad:
        :return:
        """
        source_id = int(rtspsrc.get_name().split('_')[-1])
        if self.verbose:
            logging.info(f'Pad removed from rtspsrc element. Source id: {source_id}')
        sink_pad = self.basic_blocks[source_id].connect_plugin.get_static_pad('sink')
        if sink_pad.is_linked():
            pad.unlink(sink_pad)

    def check_state(self, state_type):
        """
        Check GStreamer pipeline state
        :param state_type:
        :return:
        """
        result, state, pending = self.pipeline.get_state(self.get_state_wait)
        return (state == state_type) & (result == Gst.StateChangeReturn.SUCCESS)

    def check_eos(self, message):
        """
        Check GstMessage: EOS
        :param message: GstMessage
        :return: None
        """
        if not self.caught_eos:
            t = message.type
            if t == Gst.MessageType.EOS:
                if self.verbose:
                    logging.info('[Gst.MessageType] EOS')
                self.caught_eos = True
                self.alive = False
            else:
                self.alive = True

    def check_rtspsrc_flow(self):
        """
        A workaround fix for pad-added rtspsrc reconnection if a stream wakes up after connection lost.
        :return: None
        """
        if self.check_flow_enabled:
            if self._rtspsrc_flow_timer > self._rtspsrc_flow_th and len(self.basic_blocks.keys()) > 1:
                self._rtspsrc_flow_timer = 0
                for i in self.basic_blocks.keys():
                    if self.basic_blocks[i].rtspsrc:
                        loc = self.basic_blocks[i].rtspsrc.get_property("location")
                        cap = cv2.VideoCapture(loc)

                        if i not in self._rtspsrc_active.keys():
                            self._rtspsrc_active.update({i: cap.isOpened()})
                        else:

                            if not self._rtspsrc_active[i] and cap.isOpened():
                                # If the camera was off before and now it is on
                                self._rtspsrc_active.update({i: cap.isOpened()})
                                # Restart pipeline
                                self.alive = False

                            elif self._rtspsrc_active[i] and not cap.isOpened():
                                # If the camera was on before and now it is off
                                self._rtspsrc_active.update({i: cap.isOpened()})
                                # Do not restart pipeline

                        del cap
            else:
                self._rtspsrc_flow_timer += 1

    def reconnection_callback(self, bus, message, loop):
        """
        Reconnection callback for pipeline bus
        :param bus:
        :param message:
        :param loop:
        :return: None
        """

        self.check_eos(message)
        self.check_rtspsrc_flow()

        while not self.alive:
            time.sleep(self.reconn_wait)
            # Clean pipeline and set playing state
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline.set_state(Gst.State.READY)
            self.pipeline.set_state(Gst.State.PLAYING)
            # Restart time position
            self.pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, 0)
            # Set current frame to 0 ?
            time.sleep(self.reconn_wait)
            # Check playing state
            if self.check_state(Gst.State.PLAYING):
                # Setting playing state success
                self.alive = True
                self.caught_eos = False
            else:
                logging.info('Warning. Gst.State in pipeline could not be changed to PLAYING.')
                logging.info(self.pipeline.get_state(self.get_state_wait))


class ElementsConnectionHandler(metaclass=ABCMeta):

    def __init__(self, next_element, scr_pad=None, index: str or int = None, verbose: bool = True):
        """
        Abstract class handler for GStreamer elements dynamic generation based on pad-added and pad-removed signals.
        :param next_element: next element which needs to be connected to the previous one
        :param scr_pad:      src pad of the previous element
        :param index:        handler index
        :param loglevel:     logging
        """
        self.next_element = next_element
        self.next_element_sink_pad = None
        self.caps_string = None
        self.index = index
        self.sink_pad_name = "sink" if self.index is None else f"sink_{self.index}"
        self.video_formats = ['video/x-raw', 'video/h264', 'video/h265']
        self.verbose = verbose
        self.scr_pad = scr_pad

        if self.verbose:
            logging.info(f'Initializing {self.__class__.__name__}. self.scr_pad:{self.scr_pad}')

    def on_pad_added(self, element, src_pad) -> None:
        src_pad = self.prepare_src_pad(src_pad)
        self.caps_string = src_pad.query_caps(None).to_string()
        if self.verbose:
            logging.info(f'Linking: {self.caps_string}')
        self.make_sink_pad()
        if sum([self.caps_string.startswith(f) for f in self.video_formats]):
            src_pad.link(self.next_element_sink_pad)

    def on_pad_removed(self, element, src_pad) -> None:
        src_pad = self.prepare_src_pad(src_pad)
        if self.verbose:
            logging.info(f'Unlinking: {self.caps_string}')
        if sum([self.caps_string.startswith(f) for f in self.video_formats]):
            src_pad.unlink(self.next_element_sink_pad)
        self.destroy_sink_pad()

    def prepare_src_pad(self, src_pad):
        if self.verbose:
            logging.info(f'Predefined src_pad: {self.scr_pad}')
        src_pad = self.scr_pad if self.scr_pad is not None else src_pad
        return src_pad

    @abstractmethod
    def make_sink_pad(self):
        return

    @abstractmethod
    def destroy_sink_pad(self):
        return


class DecodeBinHandler(ElementsConnectionHandler):

    def __init__(self, *args, **kwargs):
        """
        Element handler basically for decodebin element
        :param args:
        :param kwargs:
        """
        super(DecodeBinHandler, self).__init__(*args, **kwargs)

    def make_sink_pad(self) -> None:
        if self.verbose:
            logging.info(f'Making sink pad: {self.sink_pad_name}')
        if not self.next_element_sink_pad:
            self.next_element_sink_pad = self.next_element.get_static_pad(self.sink_pad_name)

    def destroy_sink_pad(self) -> None:
        if self.verbose:
            logging.info(f'Destroying sink pad: {self.sink_pad_name}')
        pass


class StreamMuxHandler(ElementsConnectionHandler):

    def __init__(self, *args, **kwargs):
        """
         Element handler basically for nvstreammux element
        :param args:
        :param kwargs:
        """
        super(StreamMuxHandler, self).__init__(*args, **kwargs)

    def make_sink_pad(self) -> None:
        if self.verbose:
            logging.info(f'Making sink pad: {self.sink_pad_name}')
        if not self.next_element_sink_pad:
            self.next_element_sink_pad = self.next_element.get_request_pad(self.sink_pad_name)

    def destroy_sink_pad(self) -> None:
        if self.verbose:
            logging.info(f'Destroying sink pad: {self.sink_pad_name}')
        if self.next_element_sink_pad:
            self.next_element.remove_pad(self.next_element_sink_pad)
        self.next_element_sink_pad = None


def object_generator(obj, parse_function):
    """
    Note that batch_meta.frame_meta_list.data needs a cast to pyds.NvDsFrameMeta.cast()
    The casting also keeps ownership of the underlying memory in the C code,
    so the Python garbage collector will leave it alone.
    Args:
        obj:  batch_meta.frame_meta_list or another
        parse_function: pyds.NvDsFrameMeta.cast or another

    Returns:
    """

    try:
        while obj is not None:
            yield parse_function(obj.data)
            obj = obj.next
    except Exception as e:
        sys.stderr.write(e)
