import numpy as np
import scipy as sp
import gmsh
import splines
import matplotlib.pyplot as plt
from intersect import intersection
from helper import plot_spline_2d
import random
from numpy import pi as PI

'''Defining global variables'''
MAX_RADIUS= 7               # Maximum absolute distance one point on the track can be away from the previous point on the track
MIN_RADIUS = 2              # Minimum absolute distance from one point on the track to the previous one
MAX_ANGLE = 3*PI/4          # Maximum angle between the ingoing vector to one point and the outgoing vector from that point
MAX_KNOTS = 20              # Number of points defining the track
MAX_DEVIATION = 10          # Twice the maximum deviation, it gets centered around 0
TRACK_HEIGHT = 3            # Height of the simulated track above the ground anchors
MAX_CONNECTOR_LENGTH = 5    # Maximum allowed length for connections between ground anchors and points on the track
EVALUATE_INTERVALS = 100    # Number of intervals used to approximate intersections between the spline and the connection-circle

circle_radius = np.sqrt(MAX_CONNECTOR_LENGTH**2 - TRACK_HEIGHT**2)

'''Defining functions'''

def get_random_track(debug_info = True, max_radius = MAX_RADIUS, min_radius = MIN_RADIUS, max_angle = MAX_ANGLE, max_knots = MAX_KNOTS, max_deviation = MAX_DEVIATION):
    vertices = np.array([[0,0]])

    while len(vertices) < MAX_KNOTS:
        current_point = vertices[-1:]
        if(debug_info): print(current_point)
        dx = max_deviation*random.random()- max_deviation/2 #Generate random x and y deviaten to add onto the current point, centered on 0
        dy = max_deviation*random.random()- max_deviation/2
        deviation = np.array([dx,dy])
        random_point = current_point + deviation
        if(debug_info): print(random_point)
        if(len(vertices) == 1): # If there is only one point so far, the next one is automatically valid
            vertices = np.append(vertices, random_point,axis=0)
            if(debug_info): print('First point is always valid')
            continue
        previous_point = vertices[-2]
        if(debug_info): print(previous_point)
        previous_segment =current_point-previous_point
        if(debug_info): print(previous_segment)

        next_segment = random_point - current_point
        if(debug_info): print(next_segment)

        next_len = np.linalg.vector_norm(next_segment)
        if(debug_info): print(next_len)
        segments_dotproduct = np.dot(previous_segment[0], next_segment[0])
        segments_lenproduct = np.linalg.vector_norm(previous_segment)*np.linalg.vector_norm(next_segment)
        theta = np.arccos(segments_dotproduct/segments_lenproduct)
        if (MIN_RADIUS <= next_len <= MAX_RADIUS):
            if(debug_info): print(next_len)
            if(debug_info): print("Vector length ok")
        else:
            if(debug_info): print(next_len)
            if(debug_info): print("Vector either too short or too long")
            continue
        if (np.absolute(theta) < MAX_ANGLE):
            if(debug_info): print(theta)
            if(debug_info): print("Angle ok")
        else:
            if(debug_info): print(theta)
            if(debug_info): print("Angle too big")
            continue
        vertices = np.append(vertices, random_point, axis=0)
    if(debug_info): print(vertices)
    return vertices

def get_random_point_on_track(track, debug_info = True, number_knots = MAX_KNOTS):
    random_progress = number_knots*random.random()
    if(debug_info): print(random_progress)
    random_point = track.evaluate(random_progress)
    if(debug_info): print(random_point)
    return random_point

def get_random_segment(number_knots = MAX_KNOTS):
    random_segment = (number_knots-1)*random.random()
    return random_segment



track = get_random_track(debug_info=False)
s1 = splines.CatmullRom(track, endconditions='closed')

point = get_random_point_on_track(s1, debug_info=False)
is_inside = (np.linalg.norm(point - s1.evaluate(s1.grid), axis=1)) <= circle_radius
inside_vertices = np.where(is_inside==True)


fig, ax = plt.subplots()
ax.axis('equal')
circle = plt.Circle((point[0],point[1]),circle_radius,color='g',fill=False)
#plt.plot(vertices)
plot_spline_2d(s1, ax=ax,dots_per_second=30)
ax.plot(point[0],point[1],'o')
ax.add_patch(circle)
plt.show()