from threading import Thread
import logging as log
import numpy as np
from pyk4a import PyK4A
from pyk4a import Config, PyK4A, ColorResolution, DepthMode
from .depth_feed import DepthFeed

class KinectFeed(DepthFeed):
    
    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    
    
    def __init__(self, return_depth=True, depth_as_alpha=False, display=False, post_processing=[], thread=True):
        super().__init__(return_depth, depth_as_alpha, display, post_processing, thread=thread)
        # Load camera with the default config
        depth_mode = DepthMode.NFOV_2X2BINNED if return_depth else DepthMode.OFF 
        synchronized_images_only = False
        if return_depth:
            synchronized_images_only = True
        config = Config(color_resolution=ColorResolution.RES_2160P, depth_mode=depth_mode, synchronized_images_only=synchronized_images_only)
        k4a = PyK4A(config)
        self.capture = k4a

    def start(self):
        if not self.started:
            self.capture.connect()
            self.capture.exposure_mode_auto = False
            self.capture.exposure = 8330
            self.capture.whitebalance = 4050
            self.capture.brightness = 60
            self.capture.contrast = 6
            self.capture.saturation = 30
            self.capture.sharpness = 2
            self.capture.gain = 0
        super().start()

    def _get_image(self):
        if self.return_depth:
            try:
                color, depth = self.capture.get_capture(color_only=False, transform_depth_to_color=True)    
                depth = self.normalize_depth(depth)
                data = (color[..., :-1], depth)
            except:
                data = None
        else:
            try:
                color = self.capture.get_capture(color_only=True)
                data = (color[..., :-1], None)
            except:
                data = None
        return data

    def kill(self):
        self.capture.disconnect()
        super().kill()