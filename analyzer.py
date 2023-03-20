import rosbag
import cv2
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt


matplotlib.use('Agg')


HOSTNAME = 'csc22917'


bag = rosbag.Bag('data.bag')


TIME_CUTOFF_MIN = 1675038119 #526362452
TIME_CUTOFF_MAX = math.inf


image_count = 0


for topic, msg, t in bag.read_messages(topics=[f'/{HOSTNAME}/camera_node/image/compressed']):
    if t.secs < TIME_CUTOFF_MIN:
        continue
    elif t.secs > TIME_CUTOFF_MAX:
        break

    compressed_image = np.frombuffer(msg.data, np.uint8)
    im = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)
    im = cv2.flip(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 1)
    image_count += 1

    if image_count % 20 == 0:
        plt.imshow(im)
        plt.show(im)
        # plt.savefig(f'./images/test{image_count}.png')
        plt.cla()

# plt.plot(errors)
# plt.savefig('fig.png')
# plt.cla()

bag.close()