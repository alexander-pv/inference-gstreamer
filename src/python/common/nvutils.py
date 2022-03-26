################################################################################
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################


import time
import platform
import pandas as pd

start_time = time.time()
frame_count = 0


class GetFPS:
    def __init__(self, stream_id, model_name='default', rtsp_fps='', save_log=False, seconds=5):
        global start_time
        self.seconds = seconds
        self.start_time = start_time
        self.is_first = True
        global frame_count
        self.frame_count = frame_count
        self.stream_id = stream_id
        self.model_name = model_name
        self.rtsp_fps = rtsp_fps
        self.save_log = save_log

    def log_fps(self, fps):
        df = pd.DataFrame([fps])
        df.to_csv(f'fps_{self.model_name}_stream_{self.stream_id}_rtsp_fps_{self.rtsp_fps}.csv',
                  mode='a', header=False, index=False
                  )

    def get_fps(self):
        end_time = time.time()
        if self.is_first:
            self.start_time = end_time
            self.is_first = False
        if end_time - self.start_time > self.seconds:
            fps = float(self.frame_count) / self.seconds
            print("\n**********************FPS*****************************************")
            print("FPS of stream: %6.f is %3.2f" % (self.stream_id, fps))
            print("**********************FPS*****************************************\n")
            if self.save_log:
                self.log_fps(fps)
            self.frame_count = 0
            self.start_time = end_time
        else:
            self.frame_count = self.frame_count + 1

    def print_data(self):
        print('Frame_count=', self.frame_count)
        print('Start_time=', self.start_time)


def is_aarch64() -> bool:
    return platform.uname()[4] == 'aarch64'

