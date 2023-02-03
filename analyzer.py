import rosbag
import cv2
import numpy as np
import math

bag = rosbag.Bag('keyboard_control_record_all_topics.bag')

output_files = {}

TIME_CUTOFF_MIN = 1675038119 #526362452
TIME_CUTOFF_MAX = math.inf
SILENCED_TOPICS = ('/csc22917/camera_node/image/compressed', 
                   '/csc22917/display_driver_node/fragments', 
                   '/csc22917/camera_node/camera_info',
                   '/csc22917/diagnostics/ros/links',
                   '/diagnostics',
                   #'/csc22917/diagnostics/ros/topics',
                   # '/csc22917/front_center_tof_driver_node/range',
                   # '/csc22917/diagnostics/code/profiling',
                   '/my_publisher_node/diagnostics/ros/topics',
                   '/my_publisher_node/diagnostics/ros/links',
                   # '/csc22917/wheels_driver_node/wheels_cmd',
                   # '/csc22917/wheels_driver_node/wheels_cmd_executed',
                   '/rosout_agg')

for topic, msg, t in bag.read_messages(topics=[]):
    if t.secs < TIME_CUTOFF_MIN:
        continue
    elif t.secs > TIME_CUTOFF_MAX:
        break

    if topic not in output_files:
        output_files[topic] = open(f"{topic.replace('/', 'S')}.txt", 'w')

    if topic not in SILENCED_TOPICS:
        if topic == '/tf':
            pass
            # rot = msg.transforms[0].transform.rotation
            # if msg.transforms[0].header.frame_id == 'csc22917/left_wheel_axis':
            #     print(f'tf left: y={rot.y} w={rot.w}')
            # else:
            #     print(f'tf right: y={rot.y} w={rot.w}')
        elif topic == '/csc22917/left_wheel_encoder_node/tick':
            print(f'left wheel:{msg.data} at time {t}')
        elif topic == '/csc22917/right_wheel_encoder_node/tick':
            print(f'right wheel:{msg.data} at time {t}')
        else:
            print(f'reading a topic at time {t}: {str(topic)}')
            print(str(msg))

    # if topic == '/csc22917/camera_node/image/compressed':
    #     if key == 27:
    #         break

    #     compressed_image = np.frombuffer(msg.data, np.uint8)
    #     im = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)
    #     cv2.imshow('test', im)
bag.close()

for file in output_files.values():
    file.close()