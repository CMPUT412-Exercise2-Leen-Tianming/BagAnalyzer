import rosbag
import cv2
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
import util
import digit_recognition
import tensorflow as tf


# matplotlib.use('Agg')


HOSTNAME = 'csc22917'


bag = rosbag.Bag('data.bag')


TIME_CUTOFF_MIN = 1675038119 #526362452
TIME_CUTOFF_MAX = math.inf


recognizer = digit_recognition.Recognizer()
# recognizer.train_data()
recognizer.load_model()
# recognizer.save_model(None)
image_count = 0
WIDTH = 64


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
            crop = cv2.resize(crop, (28, 28), cv2.INTER_AREA)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # TAG_MASK = [(90, 0, 30), (130, 100, 60)]
            # crop = cv2.inRange(hsv, TAG_MASK[0], TAG_MASK[1])
            REFERENCE_COLOR = np.array((110, 50, 60), dtype=np.float32)
            STD = np.array((20, 50, 30), dtype=np.float32)
            dim = np.abs((hsv - REFERENCE_COLOR[np.newaxis, np.newaxis, :]) / STD[np.newaxis, np.newaxis, :])
            crop = np.sum(dim, axis=2) < 3
            plt.imshow(crop)
            plt.show()
            # plt.savefig(f'./images/test{image_count}.png')
            plt.cla()
            plt.imshow(hsv)
            plt.show()
            # plt.savefig(f'./images/test{image_count}.png')
            plt.cla()
            # contours, hierarchy = cv2.findContours(crop,
            #                                cv2.RETR_EXTERNAL,
            #                                cv2.CHAIN_APPROX_NONE)
            # cv2.drawContours(crop, contours, -1, (0, 255, 0), 3)

            digit = int(recognizer.detect_digit(crop))
            print(f'recognized digit:{digit}')
            pltim = cv2.putText(pltim, str(digit), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            pltim = cv2.polylines(pltim, np.array(
                (((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)), ), dtype=np.int32), True, (255, 0, 0))
        if npts > 0:
            plt.imshow(pltim)
            plt.show()
            # plt.savefig(f'./images/test{image_count}.png')
            plt.cla()

# plt.plot(errors)
# plt.savefig('fig.png')
# plt.cla()

bag.close()