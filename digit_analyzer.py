import rosbag
import cv2
import numpy as np
import math
import digit_recognition
from digit_bag_util import read_dict

# matplotlib.use('Agg')


HOST_NAME = 'csc22917'


bag = rosbag.Bag('collected_digit.bag')


TIME_CUTOFF_MIN = 1675038119 #526362452
TIME_CUTOFF_MAX = math.inf


def main():
    recognizer = digit_recognition.Recognizer()

    LOAD_MODEL = True
    if LOAD_MODEL:
        recognizer.load_model()
    else:
        recognizer.train_data()
        recognizer.save_model(None)

    # recognizer.test_data()
    # recognizer.test_png()
    label_table = read_dict()
    image_count = 0
    correct_count = 0
    image_count_by_digit = [0] * 10
    correct_count_by_digit = [0] * 10

    cv2.namedWindow('image')

    for topic, msg, t in bag.read_messages(topics=[]):
        if t.secs < TIME_CUTOFF_MIN:
            continue
        elif t.secs > TIME_CUTOFF_MAX:
            break

        compressed_image = np.frombuffer(msg.data, np.uint8)
        seq = msg.header.seq
        if seq not in label_table:
            continue
        label = label_table[seq]

        im = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)[:, :, 0]
        image_count += 1
        image_count_by_digit[label] += 1
        digit = int(recognizer.detect_digit(im))
        if digit == label:
            correct_count += 1
            correct_count_by_digit[label] += 1
        else:
            impath = f'../plots/im_{digit}_{label}_{image_count}.png'
            cv2.imwrite(impath, im)
            cv2.imshow('image', im)
            key = cv2.waitKey(16)
            if key % 256 == 27:
                break
    print(f'accuracy: {correct_count}/{image_count}')
    for i in range(10):
        print(f'digit {i} acc: {correct_count_by_digit[i]}/{image_count_by_digit[i]}')

    bag.close()


if __name__ == '__main__':
    main()

