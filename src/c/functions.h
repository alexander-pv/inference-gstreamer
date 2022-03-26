
#include "functions.c"

// Define platform
#if __arm__
    #define PLATFORM "aarch64"
#elif __x86_64__
    #define PLATFORM "x86_64"
#else
    #define PLATFORM "unknown"
#endif


// Make GStreamer element with check
GstElement *make_gst_element(char *factory_name, char *name);

// Check GStreamer element
int check_gst_element(GstElement *e, char *name);

// Handler for the pad-added signal
static void pad_added_handler(GstElement *src, GstPad *new_pad, GstPad *sink_pad);

// Unreference the sink pad and pad's caps
int unref_caps_and_pad(GstCaps *caps, GstPad *sink_pad);
