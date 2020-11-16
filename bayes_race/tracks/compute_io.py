""" Compute inner and outer track boundaries.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np


def ComputeIO(track):

    x_center = track.x_center
    y_center = track.y_center
    x_raceline = track.x_raceline
    y_raceline = track.y_raceline

    ns = x_center.shape[0]
    x_inner = np.zeros([ns])
    y_inner = np.zeros([ns])
    x_outer = np.zeros([ns])
    y_outer = np.zeros([ns])

    track_width = track.track_width
    eps = 0.01

    for idx in range(ns):
        
        theta = track.xy_to_param(x_center[idx], y_center[idx])
        x_, y_ = track.param_to_xy(theta+eps)
        _x, _y = track.param_to_xy(theta-eps)
        x, y = track.param_to_xy(theta)

        norm = np.sqrt((y_-_y)**2 + (x_-_x)**2)

        width = track_width/2
        x_inner[idx] = x - width*(y_-_y)/norm
        y_inner[idx] = y + width*(x_-_x)/norm

        width = -track_width/2
        x_outer[idx] = x - width*(y_-_y)/norm
        y_outer[idx] = y + width*(x_-_x)/norm

    return x_inner, y_inner, x_outer, y_outer