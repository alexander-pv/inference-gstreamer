
#include <assert.h>
#include <unistd.h>
#include <gst/gst.h>


int check_gst_element(GstElement *e, char *name) {
    assert(e != NULL);
    g_print("\nElement %s: Ok.", name);
    return 0;
}

GstElement *make_gst_element(char *factory_name, char *name) {
    GstElement *element = gst_element_factory_make(factory_name, name);
    g_print("\nCreating element: %s...", name);
    check_gst_element(element, name);
    return element;
}

int unref_caps_and_pad(GstCaps *caps, GstPad *sink_pad) {
    if (caps != NULL) {
        gst_caps_unref(caps);
    }
    gst_object_unref(sink_pad);
    return 0;
}

static void pad_added_handler(GstElement *src, GstPad *new_pad, GstPad *sink_pad) {

    GstPadLinkReturn ret;
    GstCaps *new_pad_caps = NULL;
    GstStructure *new_pad_struct = NULL;
    const gchar *new_pad_type = NULL;

    g_print("Received new pad '%s' from '%s':\n", GST_PAD_NAME (new_pad), GST_ELEMENT_NAME (src));

    /* If our converter is already linked, we have nothing to do here */
    if (gst_pad_is_linked(sink_pad)) {
        g_print("We are already linked. Ignoring.\n");
        unref_caps_and_pad(new_pad_caps, sink_pad);
    }

    /* Check the new pad's type */
    new_pad_caps = gst_pad_get_current_caps(new_pad);
    new_pad_struct = gst_caps_get_structure(new_pad_caps, 0);
    new_pad_type = gst_structure_get_name(new_pad_struct);
    if (!g_str_has_prefix(new_pad_type, "audio/x-raw")) {
        g_print("It has type '%s' which is not raw audio. Ignoring.\n", new_pad_type);
        unref_caps_and_pad(new_pad_caps, sink_pad);
    }

    /* Attempt the link */
    ret = gst_pad_link(new_pad, sink_pad);
    if (GST_PAD_LINK_FAILED (ret)) {
        g_print("Type is '%s' but link failed.\n", new_pad_type);
    } else {
        g_print("Link succeeded (type '%s').\n", new_pad_type);
    }
}
