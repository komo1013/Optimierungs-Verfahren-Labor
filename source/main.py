from timeit import main

import numpy as np
import scipy as sp
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
import gmsh
import splines
import matplotlib.pyplot as plt
from intersect import intersection
from helper import plot_spline_2d
import random
from numpy import pi as PI
import time
import sys

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
INTERSECT_EPSILON = 2e-2    # Arbitrary epsilon-value used to find the intersections between connector-circles and the track
NUM_ANCHORS = np.array([10,9,8,7,6,5,4])            # Number of anchors to be used (will get removed later)
DEBUGGING_EVALUATE = False          # Boolean to turn on/off printing of debugging information for the evaluate-function
F_VAR = 0.4
MAX_GEN = 100
TARGET = 10
CR = 0.7
rng = np.random.default_rng(42)
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

def get_distance_interpolator(spline, num_intervals = EVALUATE_INTERVALS):
    time_values = np.array([])
    distance_values = np.array([])
    cumulative_distance = 0

    
    for time0, time1 in zip(spline.grid[:-1], spline.grid[1:]):
        intermediate_times = np.linspace(time0, time1, num_intervals+1)

        velocity_vectors = spline.evaluate(intermediate_times, 1)
        velocities = np.linalg.norm(velocity_vectors, axis=1)

        intermediate_distances = cumulative_trapezoid(
            velocities,
            intermediate_times,
            initial=0
        )
        if len(time_values)>0:
            intermediate_times = intermediate_times[1:]
            intermediate_distances = intermediate_distances[1:]
        time_values = np.append(time_values, intermediate_times)
        distance_values = np.append(distance_values, intermediate_distances + cumulative_distance)
        cumulative_distance += intermediate_distances[-1]

    time_to_distance = PchipInterpolator(time_values, distance_values)

    return time_to_distance 

def get_fitness(intersections, interpolator, spline):
    s_points = np.array(interpolator(intersections))
    last_point = interpolator(spline.grid[-1])
    distances = np.array([])
    for i,s_point in enumerate(s_points[1:]):

        distances = np.append(distances, s_point - s_points[i])

        if s_point == s_points[-1]:
            remaining_distance = last_point - s_point
            lap_distance = remaining_distance + s_points[0]
            distances = np.append(distances, lap_distance)
    fitness = distances[np.argmax(distances)]
    return fitness

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

def mo_function(spline, g, polys, boxes, anchor_points, circle_radius,plotten=False):
    if plotten:
        fig, ax = plt.subplots()
        ax.grid()
        ax.axis('equal')
        plot_spline_2d(spline, ax=ax,dots_per_second=30)
    all_crossing_times_mb = np.array([])
    
    for anchor_point in anchor_points:
        
        crossing_times_mb = circle_crossings(g, polys, boxes, anchor_point, circle_radius)
        all_crossing_times_mb = np.append(all_crossing_times_mb, crossing_times_mb)
        crossing_points_mb = spline.evaluate(crossing_times_mb) if len(crossing_times_mb) else []

        if plotten:
            circle = plt.Circle((anchor_point[0],anchor_point[1]),circle_radius,color='g',fill=False)
            ax.plot(anchor_point[0],anchor_point[1],'o')
            for point in crossing_points_mb:
                #print(point)
                x = point[0]
                y = point[1]
                ax.plot(x, y, 'o')
            ax.add_patch(circle)
    return all_crossing_times_mb

def create_multiple_agents(anchor_number, s1, g, polys, boxes, circle_radius, distance_interpolator,total_track_length):
    fitnesses = np.array([])
    state_vectors = np.array([])
    track_increments = total_track_length/anchor_number
    while True:
        # Initialize the array of anchor points
        anchor_points = np.array([np.array(s1.evaluate(track_increments))])
        random_anc_points = np.sort(np.random.uniform(s1.grid[0],s1.grid[-1],size=anchor_number))
        anchor_points = np.asarray(s1.evaluate(random_anc_points))
        # print("size Anchor points: ", anchor_points.size)
        all_crossing_times = mo_function(s1, g, polys, boxes,anchor_points, circle_radius)
        all_crossing_times = np.unique(all_crossing_times, sorted=True)
        fitness = get_fitness(all_crossing_times, distance_interpolator, s1)
        state_vector = np.ndarray.flatten(anchor_points)
        state_vector = np.array(anchor_points)
                
        print("Fitness: ", fitness)
        fitnesses = np.append(fitnesses, fitness)  
        state_vectors = np.append(state_vectors, state_vector)  
        if len(fitnesses) >= 30:
            break
        
    # print("size Anchor points: ", anchor_points.size)
    # anchor_points = np.append(anchor_points, np.array([s1.evaluate(track_increments)]),axis=0)   
    # # print("Anchor points: ", anchor_points) 
    # print("size Anchor points: ", anchor_points.size)
    # all_crossing_times = mo_function(s1, g, polys, boxes,anchor_points, circle_radius)
    # all_crossing_times = np.unique(all_crossing_times, sorted=True)
    # fitness = get_fitness(all_crossing_times, distance_interpolator, s1)
    # state_vector = np.ndarray.flatten(anchor_points)
    # print("Fitness: ", fitness)
    # fitnesses = np.append(fitnesses, fitness)  
    # state_vectors = np.append(state_vectors, state_vector)  

    return fitnesses, state_vectors

def crossover_T_vec(agent_v, student_agent):
    CR_r = np.random.random(size=agent_v.size)
    # print("CR_r: ", CR_r)
    agent_T =np.array([])
    for i in np.arange(0,agent_v.size):
        if CR_r[i] < 0.7:
            agent_T = np.append(agent_T, agent_v[0][i])
        else:
            agent_T = np.append(agent_T, student_agent[i])
    return agent_T

def evaluate(anchor_points, s1, g, polys, boxes, circle_radius, distance_interpolator):
    crossings = np.unique(mo_function(s1, g, polys, boxes, anchor_points, circle_radius))
    return get_fitness(crossings, distance_interpolator, s1)

def de_generation(pop, fit, F, CR, rng, lower, upper, s1, g, polys, boxes, circle_radius, distance_interpolator):
    N, n, _ = pop.shape
    for i in range(N):
        others = [j for j in range(N) if j != i]
        a, b, c = rng.choice(others, size=3, replace=False)

        v = pop[a] + F * (pop[b] - pop[c])            # Mutation, Form (n, 2)

        mask = rng.random((n, 2)) < CR                 # Crossover pro Komponente
        mask[rng.integers(n), rng.integers(2)] = True  # mindestens eine aus v
        trial = np.where(mask, v, pop[i])
        trial = np.clip(trial, lower, upper)           # Grenzen

        f_trial = evaluate(trial, s1, g, polys, boxes, circle_radius, distance_interpolator)
        if f_trial <= fit[i]:                          # Selektion
            pop[i] = trial
            fit[i] = f_trial
    return pop, fit

def check_optimization_completion(fitnesses):
    print("check_optimization_completion")
    if np.min(fitnesses) < 10:
        idx_best_fitness = np.argmin(fitnesses)
        print("best idx: ", idx_best_fitness)
        return True, idx_best_fitness
    else:
        return False, None

'All of these only need to run once per spline'
def main():
    track = get_random_track(debug_info=False)
    s1 = splines.CatmullRom(track, endconditions='closed')
    total_track_length = s1.grid[-1] - s1.grid[0]
    distance_interpolator = get_distance_interpolator(s1)
    g, polys = segment_polynomials(s1)
    boxes = segment_boxes(polys)
    lower = track.min(axis=0) - circle_radius   # Rahmen um die Strecke
    upper = track.max(axis=0) + circle_radius
    'Needs to be recalculated every time the number of anchors is changed (for now constant)'
    start_all_time = time.perf_counter()
    time_mv = np.array([])
    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.grid()
    plot_spline_2d(s1, ax=ax,dots_per_second=30)
    plt.title(f"Track with {NUM_ANCHORS} anchors")
    plt.ion()
    plt.show()
    best_of_all = np.array([])
    for i in np.arange(0,NUM_ANCHORS.size):
    # for i in np.arange(0,2):
        print(f"Processing {NUM_ANCHORS[i]} anchors")


        'All of the following code is needed for the actual optimiziation, needs to run for every agent'
        start_mv = time.perf_counter()

        'The anchors wouldn\'t need to be regenerated, they are defined by the agent itself.'
        'First generation should have some amount of randomness, otherwise it\'s a bit pointless.'
        Agents_list_fitnesses, Agent_state_vectors = create_multiple_agents(
            NUM_ANCHORS[i], s1, g, polys, boxes, circle_radius, distance_interpolator, total_track_length)
        Agent_state_vectors = np.reshape(Agent_state_vectors, (-1, NUM_ANCHORS[i], 2))

        pop, fit = Agent_state_vectors, Agents_list_fitnesses

        for gen in range(MAX_GEN):
            pop, fit = de_generation(pop, fit, F=F_VAR, CR=CR, rng=rng, lower=lower, upper=upper,s1=s1, g=g, polys=polys, boxes=boxes, circle_radius=circle_radius, distance_interpolator=distance_interpolator)
            print(f"Anker {NUM_ANCHORS[i]}, Generation {gen}: beste Fitness {fit.min():.3f}")
            if fit.min() < TARGET:
                break

        Agent_state_vectors, Agents_list_fitnesses = pop, fit
        idx_best_fitness = np.argmin(fit)

        print("Optimization completed for ", NUM_ANCHORS[i], " anchors")
        print("Best fitness: ", Agents_list_fitnesses[idx_best_fitness])
        best_of_all = np.append(best_of_all, Agents_list_fitnesses[idx_best_fitness])
        print("The best agent is: ", Agent_state_vectors[idx_best_fitness])
        # best_of_all = np.append(best_of_all, [Agent_state_vectors[idx_best_fitness],Agents_list_fitnesses[idx_best_fitness], NUM_ANCHORS[i]])

        # save_best_agent = input("Do you want to save the best agent? (y/n): ")
        save_best_agent = "n"
        if save_best_agent == "y":
            np.savetxt(f"best_agent_anchors.npy", Agent_state_vectors[idx_best_fitness])
            np.savetxt(f"best_fitness_anchors.npy", Agents_list_fitnesses[idx_best_fitness])
            print("Best agent saved.")


        print("\n continue to next number of anchors")

        end_mv = time.perf_counter()
        time_mv =np.append(time_mv, end_mv - start_mv)

    end_all_time = time.perf_counter()
    print("Total time for all anchors: ", end_all_time - start_all_time)
    for i in np.arange(0,NUM_ANCHORS.size):
        print(f"{NUM_ANCHORS[i]} anchors best fitness is: ", best_of_all[i])

    print("Best fitness: ", np.min(best_of_all))
    min_idx = np.argmin(best_of_all)
    print("Anchor number with best fitness: ", NUM_ANCHORS[min_idx])
    print("Best fitness: ", np.max(best_of_all))
    max_idx = np.argmax(best_of_all)
    print("Anchor number with worst fitness: ", NUM_ANCHORS[max_idx])



    print("\nTime for this number of anchors: ", end_mv - start_mv)

if __name__ == "__main__":
    main()