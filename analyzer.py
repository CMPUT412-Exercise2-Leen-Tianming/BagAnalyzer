import rosbag
import cv2
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
import util
import digit_recognition
import tensorflow as tf
import tag_contour


# matplotlib.use('Agg')


HOSTNAME = 'csc22917'


bag = rosbag.Bag('data.bag')


TIME_CUTOFF_MIN = 1675038119 #526362452
TIME_CUTOFF_MAX = math.inf


def plt_showim(im):
    plt.imshow(im)
    plt.show()
    plt.cla()


def mask_to_grayscale(im, min_value, max_value):
    im = (max_value - np.minimum(max_value, np.maximum(min_value, im))) * \
           (255.9 / (max_value - min_value))
    im = np.array(im, dtype=np.uint8)
    return im


def mask_by_hsv_bounds(im_hsv):
    # TAG_MASK = ((90, 0, 30), (130, 100, 60))
    TAG_MASK = ((100, 130, 80), (107, 220, 140))
    im = cv2.inRange(im_hsv, TAG_MASK[0], TAG_MASK[1])
    return im


def main():
    WIDTH = 96

    recognizer = digit_recognition.Recognizer()
    # recognizer.train_data()
    recognizer.load_model()
    # recognizer.save_model(None)
    recognizer.test_data()
    image_count = 0

    for topic, msg, t in bag.read_messages(topics=[f'/{HOSTNAME}/camera_node/image/compressed']):
        if t.secs < TIME_CUTOFF_MIN:
            continue
        elif t.secs > TIME_CUTOFF_MAX:
            break

        compressed_image = np.frombuffer(msg.data, np.uint8)
        im = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)
        # im = cv2.pyrDown(im)
        # im = cv2.flip(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 1)
        image_count += 1

        if image_count % 20 == 0:
            selectim = np.copy(im)
            pltim = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            ihom2d_pts = util.selectManyPoints(selectim)
            npts = ihom2d_pts.shape[0]
            for i in range(npts):
                ihom_pt = ihom2d_pts[i, :]
                cx, cy = ihom_pt[0].item(), -ihom_pt[1].item()
                xmin, ymin = int(cx - WIDTH * .5), int(cy - WIDTH * .5)
                xmax, ymax = xmin + WIDTH, ymin + WIDTH

                # make a box and crop the image
                crop = im[ymin:ymax, xmin:xmax]
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                REFERENCE_COLOR = np.array((103, 175, 135), dtype=np.float32)
                STD = np.array((5, 40, 60), dtype=np.float32)
                dim = np.abs((hsv - REFERENCE_COLOR[np.newaxis, np.newaxis, :]) / STD[np.newaxis, np.newaxis, :])
                dim = np.sum(dim, axis=2)
                crop = np.array(dim < 3, dtype=np.uint8)  # crop = mask_to_grayscale(dim, 3., 3.8)

                plt_showim(crop)
                plt_showim(hsv)

                # handling contours`
                contours, hierarchy = cv2.findContours(crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                square_idx = tag_contour.find_largest_contour_idx(contours, hierarchy, 0)
                if square_idx == -1:
                    print(f'no contour detected in the input!')
                    continue
                digit_idx = tag_contour.find_largest_contour_idx(contours, hierarchy, hierarchy[0][square_idx][2])
                if digit_idx == -1:
                    print(f'no digit detected in the input!')
                    continue

                bound_x, bound_y, bound_w, bound_h = cv2.boundingRect(contours[square_idx])
                # digit_x, digit_y, digit_w, digit_h = cv2.boundingRect(contours[square_idx])
                # cv2.rectangle(crop, (bound_x, bound_y), (bound_x + bound_w, bound_y + bound_h), (0, 255, 0), 2)
                # cv2.rectangle(crop, (digit_x, digit_y), (digit_x + digit_w, digit_y + digit_h), (0, 255, 0), 2)

                digit_im = np.zeros(crop.shape, dtype=np.uint8)
                cv2.drawContours(digit_im, contours, digit_idx, 255, cv2.FILLED, cv2.LINE_8, hierarchy, 1)
                # crop again to get only the digit
                digit_im = digit_im[bound_y:bound_y + bound_h, bound_x:bound_x + bound_w]
                digit_im = cv2.resize(digit_im, (28, 28), cv2.INTER_AREA)
                plt_showim(digit_im)

                digit = int(recognizer.detect_digit(digit_im))
                impath = f'../plots/im_{image_count}_{i}_{digit}.png'
                cv2.imwrite(impath, crop)
                print(f'recognized digit:{digit}')
                pltim = cv2.putText(pltim, str(digit), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                pltim = cv2.polylines(pltim, np.array(
                    (((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)), ), dtype=np.int32), True, (255, 0, 0))
            # if npts > 0:
            #     plt_showim(pltim)
            #     # plt.savefig(f'./images/test{image_count}.png')

    bag.close()


if __name__ == '__main__':
    main()

