//
// Created by Alexander Popkov
//

#include <stdio.h>
#include <string.h>
#include <gst/gst.h>
//#include <gstnvdsmeta.h>
//#include <gst-nvmessage.h>
//#include <nvdsmeta.h>

#include "argparser.h"
#include "functions.h"

#define true 1
#define false 0
#define ELEMENT_MAX_STR_LEN 20


int main(int argc, char *argv[]) {

    // Checking platform. Available: aarch64, x86_64.
    printf("\nPlatform: %s\n", PLATFORM);
    if (strcmp(PLATFORM, "unknown") == 0) {
        g_printerr("Warning. Unknown platform.\n");
        return -1;
    }

    // Handling arguments.
    ParsedArgs args;
    if (argc < 2) {
        printf("No arguments specified. Using default settings.\n");
        args = DefaultParsedArgs;
    } else {
        parse_arguments(&args, &argv[0]);
    }
    display_arguments(&args);

    //Making GStreamer pipeline.
    gst_init(&argc, &argv);

    GstBus *bus;
    GstCaps *caps;
    GstCapsFeatures *feature;
    GstMessage *msg;
    GstStateChangeReturn ret;
    GMainLoop *loop = g_main_loop_new(NULL, false);

    GstElement *pipeline, *rtspsrc, *capsfilter, *nvvidconv, *decoder, *sink, *depayer, *parser, *transform;
    pipeline = gst_pipeline_new("rtsp_client");
    check_gst_element(pipeline, "rtsp_client");

    // Creating elements
    // rtspsrc
    rtspsrc = make_gst_element("rtspsrc", "rtspsrc_0");
    g_object_set(GST_OBJECT(rtspsrc), "location", args.rtsp_address, NULL);
    g_object_set(GST_OBJECT(rtspsrc), "retry", 10, NULL);
    // nvvidconv
    nvvidconv = make_gst_element("nvvideoconvert", "nvidia_convertor");
    // capsfilter
    capsfilter = make_gst_element("capsfilter", "filter");
    caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGBA", NULL);
    feature = gst_caps_features_new("memory:NVMM", NULL);
    gst_caps_set_features(caps, 0, feature);
    g_object_set(G_OBJECT (capsfilter), "caps", caps, NULL);

    if (strcmp(PLATFORM, "aarch64") == 0) {

        // depayer
        char depayer_name[ELEMENT_MAX_STR_LEN] = "rtp\0";
        strcat(depayer_name, args.codec);
        strcat(depayer_name, "depay");
        depayer = make_gst_element(depayer_name, "depayer_0");
        // parser
        char *parser_name = args.codec;
        strcat(parser_name, "parse");
        parser = make_gst_element(parser_name, "parser_0");
        // decoder
        decoder = make_gst_element("nvv4l2decoder", "nvv4l2decoder_0");
        g_object_set(GST_OBJECT(decoder), "enable-max-performance", true, NULL);

        if (args.display) {
            transform = make_gst_element("nvegltransform", "nvegl-transform");
            sink = make_gst_element("nveglglessink", "nvvideo-renderer");
        } else {
            sink = make_gst_element("fakesink", "fake_sink");
        }

    } else {
        decoder = make_gst_element("decodebin", "decode_container_0");
        if (args.display) {
            sink = make_gst_element("nveglglessink", "nvvideo-renderer");
        } else {
            sink = make_gst_element("fakesink", "fake_sink");
        }

    }

    // Add elements to the pipeline
    g_print("\nAdding elements to the pipeline");
    if (strcmp(PLATFORM, "aarch64") == 0) {

        gst_bin_add_many(GST_BIN(pipeline),
                         rtspsrc, depayer, parser, decoder, nvvidconv, capsfilter, NULL);
        if (args.display) {
            gst_bin_add_many(GST_BIN(pipeline), transform, sink, NULL);
        } else {
            gst_bin_add(GST_BIN(pipeline), sink);
        }

    } else {
        gst_bin_add_many(GST_BIN(pipeline), rtspsrc, decoder, nvvidconv, capsfilter, sink, NULL);
    }

    //Linking elements
    g_print("\nLinking pipeline elements");
    if (strcmp(PLATFORM, "aarch64") == 0) {
        // rtspsrc -> [depayer -> parser -> decoder] -> nvvidconv -> capsfilter ->
        GstPad *depayer_sinkpad = gst_element_get_static_pad(depayer, "sink");
        g_signal_connect(rtspsrc, "pad-added", G_CALLBACK(pad_added_handler), depayer_sinkpad);
        gst_element_link_many(depayer, parser, decoder, nvvidconv, NULL);
    } else {
        // rtspsrc -> decoder(decodebin) -> nvvidconv -> capsfilter ->
        GstPad *decoder_sinkpad = gst_element_get_static_pad(decoder, "sink");
        g_signal_connect(rtspsrc, "pad-added", G_CALLBACK(pad_added_handler), decoder_sinkpad);
        GstPad *nvvidconv_sinkpad = gst_element_get_static_pad(nvvidconv, "sink");
        g_signal_connect (decoder, "pad-added", G_CALLBACK(pad_added_handler), nvvidconv_sinkpad);
    }
    gst_element_link(nvvidconv, capsfilter);

    if (args.display) {
        if (strcmp(PLATFORM, "aarch64") == 0) {
            // capsfilter -> transform -> sink
            gst_element_link_many(capsfilter, transform, sink, NULL);
        } else {
            // capsfilter -> sink
            gst_element_link(capsfilter, sink);
        }
    } else {
        // capsfilter -> sink
        gst_element_link(capsfilter, sink);
    }

    ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_printerr("Unable to set the pipeline to the playing state.\n");
        gst_object_unref(pipeline);
        return -1;
    }

    g_print("\nRunning RTSP stream: %s...\n", args.rtsp_address);
    g_main_loop_run(loop);

    gst_object_unref(bus);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    return 0;
}