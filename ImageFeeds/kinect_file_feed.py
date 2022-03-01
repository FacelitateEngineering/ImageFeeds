from pyk4a import PyK4APlayback, ImageFormat
from .depth_feed import DepthFeed
import cv2

def convert_to_bgra_if_required(color_format, color_image):
    # examples for all possible pyk4a.ColorFormats
    if color_format == ImageFormat.COLOR_MJPG:
        color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    elif color_format == ImageFormat.COLOR_NV12:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2RGBA_NV12)
    elif color_format == ImageFormat.COLOR_YUY2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2RGBA_YUY2)
    return color_image


class KinectFileFeed(DepthFeed):
    def __init__(self, filename, return_depth=True, depth_as_alpha=False, display=False, post_processing=[], thread=True, drop_frame=False):
        super().__init__(return_depth, depth_as_alpha, display, post_processing, thread, drop_frame)
        self.playback = PyK4APlayback(filename)
        return  
    
    def start(self):
        if not self.started:
            self.playback.open()
        super().start()

    def _get_image(self):
        try:
            capture = self.playback.get_next_capture()
            color = depth = None
            if capture.color is None:
                return None
            color = convert_to_bgra_if_required(self.playback.configuration["color_format"], capture.color)
            if self.return_depth:
                depth = self.normalize_depth(capture.transformed_depth)
            data = (color, depth)
        except EOFError as e:
            print('Data exhausted')
            data = None
            self.exhausted = True
        return data

    def kill(self):
        super().kill()
        self.playback.close()