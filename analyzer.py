import rosbag
import cv2
import numpy as np
import math
import os
import matplotlib
from matplotlib import pyplot as plt
import util
from digit_recognition import plt_showim


HOST_NAME = 'csc22917'


TIME_CUTOFF_MIN = 1675038119 #526362452
TIME_CUTOFF_MAX = math.inf


def compute_dim(im_hsv, reference_color, reference_std):
    diff_im = np.abs((im_hsv - reference_color[np.newaxis, np.newaxis, :]) / reference_std[np.newaxis, np.newaxis, :])
    diff_im = np.sum(diff_im, axis=2)
    return diff_im


def select_over_dataset(bag, roi_txt_path, im_dict):
    out_file = open(roi_txt_path, 'w')
    image_count = 0
    for topic, msg, t in bag.read_messages(topics=[f'/{HOST_NAME}/camera_node/image/compressed']):
        if t.secs < TIME_CUTOFF_MIN:
            continue
        elif t.secs > TIME_CUTOFF_MAX:
            break

        im = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        image_count += 1

        if image_count % 20 == 0:
            roi = (-1, -1, -1, -1)
            while roi != (0, 0, 0, 0):
                if roi != (-1, -1, -1, -1):
                    # new line
                    out_file.write(f'{image_count} {roi[0]} {roi[1]} {roi[2]} {roi[3]}')
                    out_file.write(f'\n')
                    if image_count not in im_dict:
                        im_dict[image_count] = im

                roi = cv2.selectROI('selectROI', im)
                print(roi)
    out_file.close()


def save_images_by_ids(bag, output_dir, im_dict):
    out_file = open(f'{output_dir}/im_ids.txt', 'w')
    image_count = 0
    for topic, msg, t in bag.read_messages(topics=[f'/{HOST_NAME}/camera_node/image/compressed']):
        if t.secs < TIME_CUTOFF_MIN:
            continue
        elif t.secs > TIME_CUTOFF_MAX:
            break

        im = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        image_count += 1

        if image_count in im_dict:
            out_file.write(f'{image_count}\n')
            cv2.imwrite(f'{output_dir}/im_{image_count}.jpg', im)
    out_file.close()


def main():
    OUTPUT_DIR = input('give your output directory a name:')
    if not os.path.exists(f'{OUTPUT_DIR}'):
        os.makedirs(f'{OUTPUT_DIR}')

    bag = rosbag.Bag('project_test_run.bag')
    im_dict = {}
    select_over_dataset(bag, f'{OUTPUT_DIR}/bot.txt', im_dict)
    select_over_dataset(bag, f'{OUTPUT_DIR}/duckie.txt', im_dict)
    save_images_by_ids(bag, OUTPUT_DIR, im_dict)

    bag.close()


if __name__ == '__main__':
    main()

