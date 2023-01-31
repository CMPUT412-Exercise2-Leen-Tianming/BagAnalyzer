# reference: https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array

import cv2
import numpy as np
import matplotlib
import io


def fig_to_im(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180)
    buf.seek(0)
    im_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    im = cv2.imdecode(im_arr, 1)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # not sure why this is in the reference code, adding this mixes up R and B
    return im


def plot_path(ax, xList, yList):
    ax.plot(xList, yList, color='r')

def cv_show_plot(fig):
    im = fig_to_im(fig)
    cv2.imshow('est', im)
