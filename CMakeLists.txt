cmake_minimum_required(VERSION 3.21)
project(inference_gst)

set(CMAKE_CXX_STANDARD 14)

add_executable(
        inference_gst
        src/c/gst_read_rtsp.c
)

include_directories(
        /usr/include/gstreamer-1.0
        /usr/include/glib-2.0
        /usr/lib/glib-2.0/include
        /usr/lib/gstreamer-1.0/include
        /usr/lib/x86_64-linux-gnu/glib-2.0/include
)

find_package(PkgConfig)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
target_link_libraries(inference_gst ${GSTREAMER_LIBRARIES})