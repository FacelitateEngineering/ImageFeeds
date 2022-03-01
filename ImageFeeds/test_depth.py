import time
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
import pyrealsense2 as rs

from XM540 import *
from realsense_feed import RealSenseFeed

parser = ArgumentParser()
parser.add_argument('--device', type=str,
                    default='/dev/ttyUSB0', help='tty device')
# parser.add_argument('--ip', type=str, default='192.168.100.25', help='IP camera')
parser.add_argument('--v', type=int, default=10, help='Speed of the turntable')
parser.add_argument('--n_images', type=int, default=720,
                    help='collect n images per set, shouldn"t go over 360')
parser.add_argument('--folder', type=str, default=None,
                    help='folder to output the images')
parser.add_argument('--display', action='store_true', help='show live video')
opt = parser.parse_args()

# feed = RealSenseFeed(display=opt.display, post_processing=[
#                      lambda x: np.rot90(x, k=1)])
# feed.start()

device = XM540(DEVICE_NAME=opt.device, velocity=opt.v)

if opt.folder is None:
    opt.folder = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")
folder = Path(opt.folder)
folder.mkdir(exist_ok=True, parents=True)


pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

align_to = rs.stream.color
align = rs.align(align_to)


def collect_one_set():
    device.reset_to_0()
    init_pos = device.get_pos_raw()
    print(f'Start collecting a set, starting at position {init_pos}')
    angles = (np.arange(opt.n_images) / opt.n_images * 4096).astype(np.int)
    paths = []
    try:
        for angle in tqdm(angles):
            device.set_pos_raw(angle)
            time.sleep(0.5)
            # img, _, _ = feed.read()
            cur_pos = device.get_pos_raw()
            file_path_color = folder / (str(cur_pos).zfill(4) + '_color' + '.png')
            file_path_seg = folder / (str(cur_pos).zfill(4) + '_seg' + '.png')
            file_path_depth = folder / (str(cur_pos).zfill(4) + '_depth' + '.png')

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.rot90(np.asanyarray(aligned_depth_frame.get_data()), 3)
            color_image = np.rot90(np.asanyarray(color_frame.get_data()), 3)

            mask = np.where((depth_image > clipping_distance) | (depth_image <= 0), 0, 255)
            mask = np.array(mask, dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            foreground_img = cv2.bitwise_and(color_image, color_image, mask=mask)

            # display
            # images = np.hstack((color_image, depth_image_3d, bg_removed))
            # cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('Align Example', color_image)
            # key = cv2.waitKey(1)
            # # Press esc or 'q' to close the image window
            # if key & 0xFF == ord('q') or key == 27:
            #     cv2.destroyAllWindows()
            #     break

            cv2.imwrite(str(file_path_color), color_image)
            cv2.imwrite(str(file_path_seg), mask)
            cv2.imwrite(str(file_path_depth), depth_image)
    finally:
        # feed.kill()
        if len(paths) == opt.n_images:
            print('Capturing done, copying image to from SDCard')
        else:
            print(f'Capture terminated [{len(paths)}/{opt.n_images}]')
        print('Collection finished')
            

print('Ready for Command')
while True:
    c = input()
    words = c.split(" ")
    if words[0] == "R":
        device.reset_to_0()
    elif words[0] == "O":
        collect_one_set()
    elif words[0] == 'P':
        position = device.get_pos_raw()
        print(f'Position : {position}')
    elif words[0] == 'M':
        try:
            pos_degree = int(words[1])
            device.set_pos_raw(pos_degree)
        except ValueError:
            print(f'"{words[1]}" cannot be parsed as string')
    elif words[0] == 'V':
        try:
            v = int(words[1])
            device.set_vel_raw(v)
        except ValueError:
            print(f'"{words[1]}" cannot be parsed as string')
    elif words[0] == 'Q':
        exit(0)
    else:
        print("COMMAND ERROR")
