""" Runge Kutta sixth order integration.
    Uses same syntax as scipy.integrate.odeint
    `fun` should be of the form fun(t, y, *args)
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np


def odeintRK6(fun, y0, t, args=()):
    gamma = np.asarray([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    y_next = np.zeros([len(t)-1, len(y0)])

    for i in range(len(t)-1):
        h = t[i+1]-t[i]
        k1 = h*fun(t[i], y0, *args)
        k2 = h*fun(t[i]+h/4, y0+k1/4, *args)
        k3 = h*fun(t[i]+3/8*h, y0+3/32*k1+9/32*k2, *args)
        k4 = h*fun(t[i]+12/13*h, y0+1932/2197*k1-7200/2197*k2+7296/2197*k3, *args)
        k5 = h*fun(t[i]+h, y0+439/216*k1-8*k2+3680/513*k3-845/4104*k4, *args)
        k6 = h*fun(t[i]+h/2, y0-8/27*k1+2*k2-3544/2565*k3+1859/4104*k4-11/40*k5, *args)
        K = np.asarray([k1, k2, k3, k4, k5, k6])
        y_next[i,:] = y0 + gamma@K
        y0 = y0 + gamma@K
    return y_next