import numpy as np
import scipy as sp
import gmsh
import splines
import matplotlib.pyplot as plt
from intersect import intersection
from helper import plot_spline_2d
import random
from numpy import pi as PI
import time

start = time.perf_counter() # starting with a Timestamp to measure the time it takes to run the code

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
INTERSECT_EPSILON = 2-3     # Arbitrary epsilon-value used to find the intersections between connector-circles and the track
DEBUGING_EVALUATE = False          # Boolean to turn on/off printing of debuging information for the evaluate-function

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

def check_surrounding_points(spline, circle_center, radius, locations, previous_segment = True, next_segment = True, num_intervals = EVALUATE_INTERVALS):

    if(previous_segment==next_segment==False): return None
    previous_times = np.array([])
    next_times = np.array([])
    current_location = locations[-2]
    
    if previous_segment:
        previous_location = locations[0]
        total_previous_distance = current_location - previous_location
        previous_increment = total_previous_distance/num_intervals
        


    if next_segment: 
        next_location = locations[-1]
        total_next_distance = next_location - current_location
        next_increment = total_next_distance/num_intervals


    if(previous_segment):
        previous_times = np.arange(num_intervals)*previous_increment + previous_location
        if next_segment: previous_times = np.append(previous_times, current_location)
    if(next_segment):
        next_times = np.arange(num_intervals)*next_increment + current_location



    crossing_times = np.array([])
    #print(previous_times)
    #print(next_times)
    #print(len(next_times))
    avg_time_1 = avg_time_2 = 0
    #print(spline.evaluate(previous_times))
    #idx = -1
    #if(len(previous_times) != 0):
    #    points = spline.evaluate(previous_times)
    #    points -= circle_center
    #    dist = np.linalg.norm(points, axis=1)
    #    dist = np.abs(dist-radius)
    #    print(dist)
    #    idx = np.argmin(dist) if dist[idx] <= INTERSECT_EPSILON else -1
    ##if len(previous_times)!=0 : idx = np.argmin(np.abs(spline.evaluate(previous_times) - circle_center)-radius)
    #print(idx)
    #if idx != -1 :print(spline.evaluate(idx))
    for location in previous_times:

        if(location == previous_times[-1]) : continue
        start = time.perf_counter()
        current_distance = np.linalg.norm(circle_center - spline.evaluate(location))
        next_distance = np.linalg.norm(circle_center - spline.evaluate(location + previous_increment))
        step1 = time.perf_counter() - start
        if((current_distance<=radius)!=(next_distance<= radius)):
            if(np.linalg.norm(circle_center - spline.evaluate(location))<=circle_radius):
                crossing_times = np.append(crossing_times, location)
            else:
                crossing_times = np.append(crossing_times, location + previous_increment)
        step2 = time.perf_counter() - step1
        avg_time_1 += (step2 + step1)
    if len(previous_times)!= 0 : avg_time_1 = avg_time_1/len(previous_times)
    for location in next_times:
        if(location == next_times[-1]) : continue
        start = time.perf_counter()
        current_distance = np.linalg.norm(circle_center - spline.evaluate(location))
        next_distance = np.linalg.norm(circle_center - spline.evaluate(location + next_increment))
        step1 = time.perf_counter() - start
        if((current_distance<=radius)!=(next_distance<= radius)):
            if(current_distance<=radius):
                crossing_times = np.append(crossing_times, location)
            else:
                crossing_times = np.append(crossing_times, location + next_increment)
        step2 = time.perf_counter() - step1
        avg_time_2 += step2 + step1
    if len(previous_times)!= 0 : avg_time_2 = avg_time_2/len(previous_times)
    if (len(next_times)) != 0 : avg_time_2 = avg_time_2/len(next_times)
    if (DEBUGING_EVALUATE== False):
        print("Average step 1:",avg_time_1)   #Anschalten
        print("Average step 2:",avg_time_2)



    return crossing_times

def get_inside_vertices(anchor_point, spline, radius, epsilon = CIRCLE_EPSILON):
    is_inside = (np.linalg.norm(anchor_point - spline.evaluate(spline.grid), axis=1)) <= radius + epsilon
    inside_vertices = np.where(is_inside==True)
    inside_vertices = inside_vertices[0]
    inside_times = np.array([])
    for vertex in inside_vertices:
        if(vertex !=20):inside_times = np.append(inside_times, spline.grid[vertex])

    return inside_times

def segment_polynomials(spline):
    """Pro Segment: Koeffizienten (4,2), höchste Potenz zuerst, lokaler Parameter u in [0,1]."""
    g = np.asarray(spline.grid, dtype=float)
    u = np.linspace(0, 1, 4)
    V = np.vander(u, 4)                      # Spalten: u^3, u^2, u, 1
    polys = []
    for a, b in zip(g[:-1], g[1:]):
        pts = np.asarray(spline.evaluate(a + u * (b - a)))   # (4, 2)
        polys.append(np.linalg.solve(V, pts))                # (4, 2)
    return g, polys

def segment_boxes(polys):
    """Achsenparallele Box pro Segment über die Bézier-Kontrollpunkte (garantiert umschließend)."""
    boxes = []
    for A, B, C, D in polys:                 # p(u) = A u^3 + B u^2 + C u + D
        P = np.array([D, D + C/3, D + 2*C/3 + B/3, A + B + C + D])
        boxes.append((P.min(axis=0), P.max(axis=0)))
    return boxes

def filter_segments(boxes, center, radius):
    """Indizes der Segmente, deren Box die Quadrat-Hülle des Kreises überlappt."""
    c = np.asarray(center, dtype=float)
    lo, hi = c - radius, c + radius
    return [i for i, (bmin, bmax) in enumerate(boxes)
            if np.all(bmax >= lo) and np.all(bmin <= hi)]

def circle_crossings(g, polys, boxes, center, radius, tol=1e-9):
    """Parameterwerte t (im Original-Parameter des Splines), bei denen die Kurve den Kreis schneidet."""
    c = np.asarray(center, dtype=float)
    times = []
    for i in filter_segments(boxes, c, radius):
        q = polys[i].copy()                  # (4, 2)
        q[3] -= c                            # p(u) - c
        f = np.polymul(q[:, 0], q[:, 0]) + np.polymul(q[:, 1], q[:, 1])
        f[-1] -= radius**2                   # |p-c|^2 - r^2, Grad 6
        for root in np.roots(f):
            if abs(root.imag) < tol and -tol <= root.real <= 1 + tol:
                u = min(max(root.real, 0.0), 1.0)
                times.append(g[i] + u * (g[i+1] - g[i]))
    return np.unique(np.round(times, 9))     # Doppelte an Segmentgrenzen entfernen

def mo_function(spline, g, polys, boxes, circle_radius):
    fig, ax = plt.subplots()
    ax.axis('equal')
    plot_spline_2d(spline, ax=ax,dots_per_second=30)
    all_crossing_times_mb = np.array([])
    
    for idx in spline.grid:
        anchor_point = spline.evaluate(idx)
        crossing_times_mb = circle_crossings(g, polys, boxes, anchor_point, circle_radius)
        all_crossing_times_mb = np.append(all_crossing_times_mb, crossing_times_mb)
        crossing_points_mb = spline.evaluate(crossing_times_mb) if len(crossing_times_mb) else []

        circle = plt.Circle((anchor_point[0],anchor_point[1]),circle_radius,color='g',fill=False)
        ax.plot(anchor_point[0],anchor_point[1],'o')
        
        for point in crossing_points_mb:
            #print(point)
            x = point[0]
            y = point[1]
            ax.plot(x, y, 'o')
        ax.add_patch(circle)
    return all_crossing_times_mb

while True:
    print("Creating a random track with the following parameters:")
    track = get_random_track(debug_info=False)
    s1 = splines.CatmullRom(track, endconditions='closed')
    s2 = splines.UnitSpeedAdapter(s1)
    print(s1.grid)
    print(s2.grid)
    #print(s2.evaluate(s2.grid[0]))
    print("s2_grid\n",s2.grid[-1],"\n")
    
    g, polys = segment_polynomials(s1)
    boxes = segment_boxes(polys)
    

    L = s2.grid[-1]
    print("evaluation of grid points:\n",s2.evaluate(np.mod(s2.grid, L)),s2.evaluate(s2.grid[0]),"\n")

    #print(s1.evaluate(s1.grid))
    #print((0,0) - s2.evaluate(s2.grid))
    start_mv = time.perf_counter()
    fig, ax = plt.subplots()
    ax.axis('equal')
    plot_spline_2d(s1, ax=ax,dots_per_second=30)
    all_crossing_times = np.array([])

    for index in s1.grid:
        if index == s1.grid[-1]: continue
        anchor_point = s1.evaluate(index)

        inside_vertices = get_inside_vertices(anchor_point, s1, circle_radius)

        if(len(np.where(inside_vertices == s1.grid[-2])[0])==0):
            inside_vertices = np.append(inside_vertices, s1.grid[-2])

        crossing_times = np.array([])
        for vertex in inside_vertices:
            is_last_vertex = vertex == (s1.grid[-2])
            is_first_vertex = vertex == (s1.grid[0])
            index = np.where(s1.grid == vertex)[0]
            index = index[0]
            if is_first_vertex:
                locations = np.array([s1.grid[index], s1.grid[index+1]])

            elif (is_last_vertex): locations = np.array([s1.grid[index-1], s1.grid[index]])
            else: locations = np.array([s1.grid[index-1], s1.grid[index], s1.grid[index+1]])

            crossing_times = np.append(crossing_times, check_surrounding_points(s1, anchor_point, circle_radius, locations, 
                                                                                previous_segment=not is_first_vertex, next_segment=not is_last_vertex, 
                                                                                num_intervals=EVALUATE_INTERVALS if not is_last_vertex else EVALUATE_INTERVALS*3))
        all_crossing_times = np.append(all_crossing_times, crossing_times)
        crossing_points = np.array(s1.evaluate(crossing_times))

        circle = plt.Circle((anchor_point[0],anchor_point[1]),circle_radius,color='g',fill=False)

        ax.plot(anchor_point[0],anchor_point[1],'o')

        for point in crossing_points:
            #print(point)
            x = point[0]
            y = point[1]
            ax.plot(x, y, 'o')
        ax.add_patch(circle)

    all_crossing_times = np.unique(all_crossing_times, sorted=True)
    end_mv = time.perf_counter()
    
    if (DEBUGING_EVALUATE== False):
        print(all_crossing_times)
    
    #Trying to find the crossing points with the new method
    start_mb = time.perf_counter()

    

    all_crossing_times_mb = mo_function(s1, g, polys, boxes, circle_radius)
    all_crossing_times_mb = np.unique(all_crossing_times_mb, sorted=True)
    print("All crossing times (method 2):", all_crossing_times_mb)
    end_mb = time.perf_counter()
    
    print("Time for the old method:\n",end_mv-start_mv)
    print("Time for the new method:\n",end_mb-start_mb)

    '''
    for index in s2.grid:
        if index == s2.grid[-1]: continue
        anchor_point = s2.evaluate(index)

        inside_vertices = get_inside_vertices(anchor_point, s2, circle_radius)

        if(len(np.where(inside_vertices == s2.grid[-2])[0])==0):
            inside_vertices = np.append(inside_vertices, s2.grid[-2])

        crossing_times = np.array([])
        for vertex in inside_vertices:
            is_last_vertex = vertex == (s2.grid[-2])
            is_first_vertex = vertex == (s2.grid[0])
            index = np.where(s2.grid == vertex)[0]
            index = index[0]
            if is_first_vertex:
                locations = np.array([s2.grid[index], s2.grid[index+1]])

            elif (is_last_vertex): locations = np.array([s2.grid[index-1], s2.grid[index]])
            else: locations = np.array([s2.grid[index-1], s2.grid[index], s2.grid[index+1]])

            crossing_times = np.append(crossing_times, check_surrounding_points(s2, anchor_point, circle_radius, locations, 
                                                                                previous_segment=not is_first_vertex, next_segment=not is_last_vertex, 
                                                                                num_intervals=EVALUATE_INTERVALS if not is_last_vertex else EVALUATE_INTERVALS*3))
        all_crossing_times = np.append(all_crossing_times, crossing_times)
        crossing_points = np.array(s2.evaluate(crossing_times))

        circle = plt.Circle((anchor_point[0],anchor_point[1]),circle_radius,color='g',fill=False)

        #ax.plot(anchor_point[0],anchor_point[1],'o')

        for point in crossing_points:
            #print(point)
            x = point[0]
            y = point[1]
            #ax.plot(x, y, 'o')
        #ax.add_patch(circle)

    '''
    end = time.perf_counter()

    print("Time for the entire code:\n",end-start)

    plt.show()
    print("I'm doney with the runny")
    a= input("do you want to run it again? (y/n): ")
    if a.lower() == 'n':
        break
    elif(DEBUGING_EVALUATE==False):
        break
        
