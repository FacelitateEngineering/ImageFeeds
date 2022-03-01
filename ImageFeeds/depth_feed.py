from threading import Thread
import logging as log
import numpy as np
from pyk4a import PyK4A
from .image_feed import VideoFeed
from datetime import datetime
import cv2
from time import sleep

class DepthFeed(VideoFeed):
    def __init__(self, return_depth=True, depth_as_alpha=False, display=False, post_processing=[], thread=True, drop_frame=True):
        super().__init__(display, post_processing, thread, drop_frame)
        self.depth_img = None
        self.return_depth = return_depth
        self.depth_as_alpha = depth_as_alpha
        self._depth_min, self._depth_max = 10, 1200
        self._depth_mag = self._depth_max - self._depth_min

    def normalize_depth(self, depth):
        depth -= self._depth_min
        depth = np.minimum(depth, self._depth_mag)
        depth = depth / self._depth_mag
        return depth

    def set_depth_range(self, _min, _max):
        assert _max >= _min
        self._depth_min = _min
        self._depth_max = _max
        self._depth_mag = self._depth_max - self._depth_min

    def _update(self):
        while not self.should_stop:
            if self.drop_frame or (not self.dirty):
                self.__update()
            sleep(0.02)

    def __update(self):
        data = self._get_image()
        if data is None:
            return
        self.dirty = True
        rgb_img, depth_img = data
        for post in self.post_processing:
            rgb_img = post(rgb_img)
            try:
                depth_img = post(depth_img)
            except:
                pass
        self.img = rgb_img
        self.depth_img = depth_img
        self.timestamp = datetime.now()
    
    def read(self):
        assert self.started, 'Call start before reading image from feed'
        if not self.use_thread:
            self.__update()
        _dirty = self.dirty
        self.dirty = False
        if not self.return_depth:
            return self.img, _dirty, self.timestamp
        return self.img, self.depth_img, _dirty, self.timestamp

    def _display(self):
        cv2.namedWindow(self.__class__.__name__ + '_RGB', cv2.WINDOW_NORMAL)
        if self.return_depth:
            cv2.namedWindow(self.__class__.__name__ + '_Depth', cv2.WINDOW_NORMAL)
        while not self.should_stop:
            if self.dirty and self.img is not None:
                
                #add depth image
                if self.return_depth and self.depth_img is not None:
                    cv2.imshow(self.__class__.__name__ + '_Depth', self.depth_img)
                cv2.imshow(self.__class__.__name__ + '_RGB', self.img)
                cv2.waitKey(1)
                # print(f'Depth : {self.depth_img.shape}')
                # print(f'RGB : {self.img.shape}')
        
    def _get_image(self):
        return (None, None)