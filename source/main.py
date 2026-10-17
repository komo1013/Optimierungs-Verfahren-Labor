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
CIRCLE_EPSILON = 0.2        # Arbitrary epsilon-value to also find vertices close to the anchor-circle
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

def check_surrounding_points(spline, circle_center, radius, vertex, previous_segment = True, next_segment = True, num_intervals = EVALUATE_INTERVALS):
    if(previous_segment==next_segment==False): return None
    previous_times = np.array([])
    next_times = np.array([])
    #print(vertex)
    if(previous_segment):
        previous_times = np.arange(num_intervals)/num_intervals + vertex - 1
    if(next_segment):
        next_times = np.arange(num_intervals)/num_intervals + vertex
    crossing_times = np.array([])
    #print(previous_times)
    print(next_times)
    for time in previous_times:
        if(time == previous_times[-1]) : continue
        if((np.linalg.norm(circle_center - spline.evaluate(time))<=circle_radius)!=(np.linalg.norm(circle_center - spline.evaluate(time + 1/num_intervals))<= circle_radius)):
            if(np.linalg.norm(circle_center - spline.evaluate(time))<=circle_radius):
                crossing_times = np.append(crossing_times, time)
            else:
                crossing_times = np.append(crossing_times, time + 1/num_intervals)
    for time in next_times:
        if(time == next_times[-1]) : continue
        if((np.linalg.norm(circle_center - spline.evaluate(time))<=circle_radius)!=(np.linalg.norm(circle_center - spline.evaluate(time + 1/num_intervals))<= circle_radius)):
            if(np.linalg.norm(circle_center - spline.evaluate(time))<=circle_radius):
                crossing_times = np.append(crossing_times, time)
            else:
                crossing_times = np.append(crossing_times, time + 1/num_intervals)

    return crossing_times

def get_inside_vertices(anchor_point, spline, radius, epsilon = CIRCLE_EPSILON):
    is_inside = (np.linalg.norm(anchor_point - spline.evaluate(s1.grid), axis=1)) <= radius + epsilon
    inside_vertices = np.where(is_inside==True)
    inside_vertices = inside_vertices[0]
    return inside_vertices


track = get_random_track(debug_info=False)
s1 = splines.CatmullRom(track, endconditions='closed')

anchor_point = get_random_point_on_track(s1, debug_info=False)
#anchor_point = s1.evaluate(s1.grid[-2])
inside_vertices = get_inside_vertices(anchor_point, s1, circle_radius)

if(len(np.where(inside_vertices == s1.grid[-2])[0])==0):
    inside_vertices = np.append(inside_vertices, s1.grid[-2])

crossing_times = np.array([])
for vertex in inside_vertices:
    crossing_times = np.append(crossing_times, check_surrounding_points(s1, anchor_point, circle_radius, vertex, previous_segment=vertex!=0, next_segment=vertex<MAX_KNOTS, num_intervals=EVALUATE_INTERVALS if vertex == MAX_KNOTS-1 else EVALUATE_INTERVALS*3))

crossing_points = np.array(s1.evaluate(crossing_times))



fig, ax = plt.subplots()
ax.axis('equal')
circle = plt.Circle((anchor_point[0],anchor_point[1]),circle_radius,color='g',fill=False)
#plt.plot(vertices)
plot_spline_2d(s1, ax=ax,dots_per_second=30)
ax.plot(anchor_point[0],anchor_point[1],'o')
for point in crossing_points:
    #print(point)
    x = point[0]
    y = point[1]
    ax.plot(x, y, 'o')
ax.add_patch(circle)
plt.show()