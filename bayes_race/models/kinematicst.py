""" Single track model.
    Adapted from https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/blob/master/vehicleModels_commonRoad.pdf
    Original file: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/PYTHON/vehiclemodels/vehicle_dynamics_ks.py
    
    Copyright 2019 Technical University of Munich, Professorship of Cyber-Physical Systems.
    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.    
"""

import math


class SteerSettings(object):
    def __init__(self, params):
        self.min = params['min_steer']
        self.max = params['max_steer']
        self.v_min = -params['max_steer_vel']
        self.v_max = params['max_steer_vel']


class AccSettings(object):
    def __init__(self, params):
        self.v_min = params['min_v']
        self.v_max = params['max_v']
        self.v_switch = params['switch_v']
        self.a_min = params['min_acc']
        self.a_max = params['max_acc']


def KinematicST(x, u_init, p):

    #create equivalent kinematic single-track parameters
    lwb = p['lf'] + p['lr']

    #states
    #x1 = x-position in a global coordinate system
    #x2 = y-position in a global coordinate system
    #x3 = steering angle of front wheels
    #x4 = velocity in x-direction
    #x5 = yaw angle

    #u1 = steering angle velocity of front wheels
    #u2 = longitudinal acceleration
    
    u = u_init

    #system dynamics
    f = [x[3]*math.cos(x[4]), 
        x[3]*math.sin(x[4]), 
        u[0], 
        u[1], 
        x[3]/lwb*math.tan(x[2])]
    
    return f