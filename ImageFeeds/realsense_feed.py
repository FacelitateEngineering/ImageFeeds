from threading import Thread
import logging as log
import numpy as np
import pyrealsense2 as rs
import cv2 
from .image_feed import VideoCamera
import matplotlib.pyplot as plt
from time import sleep
from datetime import datetime

class RealSenseFeed(VideoCamera):
    
    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1.6 #1 meter
    starting_distance_in_meters = 0.7
    def __init__(self, size=(1080, 1920), display=False, post_processing=[]):
        self.height, self.width = size
        self.display = display
        self.post_processing = post_processing        
        self.dirty = True
        self.started = False
        self.should_stop = False 
        self.img = None
        self.timestamp = None
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    def start(self):
        if not self.started:
            profile = self.pipeline.start(self.config)
            # Getting the depth sensor's depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            color_sensor = profile.get_device().first_color_sensor()
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
            color_sensor.set_option(rs.option.exposure, 400)
            color_sensor.set_option(rs.option.gain, 15)
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            depth_sensor.set_option(rs.option.enable_auto_white_balance, 0)
            depth_sensor.set_option(rs.option.exposure, 6400)
            depth_sensor.set_option(rs.option.gain, 10)
            
            depth_scale = depth_sensor.get_depth_scale()
            print("[start]: Depth Scale is: " , depth_scale)
            self.starting_distance = int(self.starting_distance_in_meters / depth_scale)
            self.clipping_distance = int(self.clipping_distance_in_meters / depth_scale)
            print(f"[start]: clipping_distance is: {self.starting_distance} {self.clipping_distance}")
            # Create an align object
            align_to = rs.stream.color
            self.align = rs.align(align_to)

            self.thread = Thread(target=self._update)
            self.thread.start()
            if self.display:
                self.display_thread = Thread(target=self._display)
                self.display_thread.start()
            self.started = True
        else:
            log.warn('Webcam stream already started')

    def _get_image(self):
        # frames.get_depth_frame() is a 640x360 depth image
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            return None

        depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(np.float)
        depth_image = depth_image.clip(self.starting_distance, self.clipping_distance)
        depth_image -= self.starting_distance
        depth_image /= (self.clipping_distance - self.starting_distance) / 255
        depth_image = depth_image.astype(np.uint8)
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image

    def _update(self):
        while not self.should_stop:
            img, depth_img = self._get_image()
            if img is None:
                continue
            self.dirty = True
            for post in self.post_processing:
                img = post(img)
                depth_img = post(depth_img)
            self.img = img
            self.depth_img = depth_img
            self.timestamp = datetime.now()

    def read(self):
        assert self.started, 'Call start before reading image from feed'
        _dirty = self.dirty
        self.dirty = False
        return self.img, self.depth_img, _dirty, self.timestamp

    def save(self, filename, ext='png'):
        img, depth = self.img.copy(), self.depth_img.copy()
        _filename = str(filename) + '.' + ext
        cv2.imwrite(_filename, img)
        _filename = str(filename) + '_depth.' + ext
        cv2.imwrite(_filename, depth)

if __name__ == '__main__':
    feed = RealSenseFeed(post_processing=[lambda x: np.rot90(x, k=1)])
    feed.start()
    img = None
    while img is None:
        print('---')
        sleep(0.4)
        img, _, _, _ = feed.read()
    plt.imshow(img.astype(np.uint8))
    plt.show()