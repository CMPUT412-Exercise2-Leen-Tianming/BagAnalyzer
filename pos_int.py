import rosbag
import cv2
import numpy as np
import math
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import cv_plot
import wheel_int


bag = rosbag.Bag('keyboard_control_record_all_topics.bag')

output_files = {}

TIME_CUTOFF_MIN = 1675038126 #526362452
TIME_CUTOFF_MAX = 1675039000
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

x, y, theta, time = 0.0, 0.0, 0.0, None

xList, yList = [], []
int_wheel = wheel_int.WheelPositionIntegration(40)
int_wheel_state = int_wheel.get_state()
fig, ax = plt.subplots(1, 1)
i = 0
ax.set_xlim(-1000, 2000)
ax.set_ylim(-1000, 2000)
ax.set_aspect('equal', adjustable='box')

for topic, msg, t in bag.read_messages(topics=['/csc22917/kinematics_node/velocity', '/csc22917/camera_node/image/compressed', 
                                               '/csc22917/left_wheel_encoder_node/tick', '/csc22917/right_wheel_encoder_node/tick']):
    if t.secs < TIME_CUTOFF_MIN:
        continue
    elif t.secs > TIME_CUTOFF_MAX:
        break

    if topic == '/csc22917/camera_node/image/compressed':
        key = cv2.waitKey(8)
        if key == 27:
            break

        compressed_image = np.frombuffer(msg.data, np.uint8)
        im = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)
        cv2.imshow('test', im)
    # elif topic == '/csc22917/kinematics_node/velocity':
    #     if time == None:
    #         time = t
    #         continue
    #     dt = (t.secs - time.secs) + (t.nsecs - time.nsecs) * 1e-9
    #     time = t
    #     oldx, oldy = x, y
    #     v, omega = msg.v, msg.omega * 8
    #     distance, dtheta = v * dt, (omega / 180 * math.pi) * dt
    #     x, y = x + distance * math.cos(theta), y + distance * math.sin(theta)
    #     theta += dtheta

    #     # xList.append(x)
    #     # yList.append(y)
    #     i += 1

    #     cv_plot.plot_path(ax, (oldx, x), (oldy, y))

    #     if i % 20 == 0:
    #         cv_plot.cv_show_plot(fig)

    #     print(f'cur state: x_{x: .3f} y_{y: .3f} theta_{theta: .3f} dt_{dt: .3f}')
    elif topic == '/csc22917/left_wheel_encoder_node/tick':
        int_wheel.update_left(msg.data, t)
        x, y, theta = int_wheel_state
        int_wheel_state = int_wheel.get_state()
        newx, newy, newtheta = int_wheel_state
        cv_plot.plot_path(ax, (x, newx), (y, newy))
        i += 1
        if i % 20 == 0:
            cv_plot.cv_show_plot(fig)
    elif topic == '/csc22917/right_wheel_encoder_node/tick':
        int_wheel.update_right(msg.data, t)
        x, y, theta = int_wheel_state
        int_wheel_state = int_wheel.get_state()
        newx, newy, newtheta = int_wheel_state
        cv_plot.plot_path(ax, (x, newx), (y, newy))
        i += 1
        if i % 20 == 0:
            cv_plot.cv_show_plot(fig)
        


bag.close()

for file in output_files.values():
    file.close()