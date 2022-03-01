import io
import logging as log
import os
import subprocess
import sys
import urllib.request
from datetime import datetime
from threading import Thread

import cv2
import gphoto2 as gp
import numpy as np
from PIL import Image
import time

class DSLRCamera:
    def __init__(self, display=False, post_processing=[], return_img=False):
        """A video camera object that provides a stream of frames"""
        log.basicConfig(
            format='%(levelname)s: %(name)s: %(message)s', level=log.WARNING)
        callback_obj = gp.check_result(gp.use_python_logging())
        self.camera = gp.check_result(gp.gp_camera_new())
        gp.check_result(gp.gp_camera_init(self.camera))
        config = gp.check_result(gp.gp_camera_get_config(self.camera))
        # find the image format config item
        # camera dependent - 'imageformat' is 'imagequality' on some
        OK, image_format = gp.gp_widget_get_child_by_name(config, 'imageformat')
        if OK >= gp.GP_OK:
            # get current setting
            value = gp.check_result(gp.gp_widget_get_value(image_format))
            # make sure it's not raw
            if 'raw' in value.lower():
                print('Cannot preview raw images')
                return 
            else:
                print(f'Value : {value}')

        OK, capture_size_class = gp.gp_widget_get_child_by_name(
            config, 'capturesizeclass')
        if OK >= gp.GP_OK:
            # set value
            value = gp.check_result(gp.gp_widget_get_choice(capture_size_class, 2))
            gp.check_result(gp.gp_widget_set_value(capture_size_class, value))
            # set config
            gp.check_result(gp.gp_camera_set_config(self.camera, config))
        # capture preview image (not saved to camera memory card)
        self.display = display
        self.return_img = return_img
        self.post_processing = post_processing
        self.started = False
        self.should_stop = False
        self.should_read = False
        
    def start(self):
        if not self.started:
            # if self.display:
            self.capture_thread = Thread(target=self._capture_thread)
            self.capture_thread.start()
            self.started = True
        else:
            log.warn('DSLR already started')

    def _get_preview(self):
        camera_file = gp.check_result(gp.gp_camera_capture_preview(self.camera))
        file_data = gp.check_result(gp.gp_file_get_data_and_size(camera_file))
        image = Image.open(io.BytesIO(file_data))
        image = np.array(image)[..., ::-1]
        return image

    def _capture(self):
        """capture full res image"""
        file_path = self.camera.capture(gp.GP_CAPTURE_IMAGE)
        if not self.return_img:
            return file_path
        
        img = cv2.imread(file_path)[..., ::-1]
        for pos in self.post_processing:
            img = pos(img)
        return img

    def kill(self):
        self.should_stop = True
        if self.display:
            self.capture_thread.join()

    def read(self):
        self.should_read = True
        while self.should_read:
            time.sleep(0.1)
        return self.img, True, None

    def _capture_thread(self):
        if self.display:
            cv2.namedWindow(self.__class__.__name__, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(self.__class__.__name__, self.height, self.width)
        while not self.should_stop:
            if self.should_read:
                self.img = self._capture()
                self.should_read = False
                continue
            else:
                time.sleep(0.1)
            if self.display:
                img = self._get_preview()
                for pos in self.post_processing:
                    img = pos(img)
                cv2.imshow(self.__class__.__name__, img)
                cv2.waitKey(1)  