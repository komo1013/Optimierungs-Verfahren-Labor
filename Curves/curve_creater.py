import os
import secrets

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=4)

import splines

from helper import plot_spline_2d, plot_tangent_2d

def shape_creator(points_number):
    # Ein eingangiger Parameter t, 2 Ausgänge x(t) und y(t)
    t = np.linspace(0, 2*np.pi, points_number)
    # x = [(2+np.sin(5*t))*np.cos(t), (2+np.sin(5*t))*np.sin(t)]
    # x = t*[np.cos(t), np.sin(t)]
    x = [20*np.sin(t) ,10 * np.cos(t)]
    x_1 = x[0].reshape(-1, 1)
    x_2 = x[1].reshape(-1, 1)
    p = np.hstack((x_1, x_2))
    p= p[:-1]

    return p

def point_changer(p_original,level):
    p_neu = []
    if(level == 1):
        for i in range(len(p_original)):
            p_neu.append((p_original[i][0], p_original[i][1]+(secrets.randbits(3))*2))
    if(level == 2):
        for i in range(len(p_original)):
            p_neu.append((p_original[i][0]+(secrets.randbits(3))*2, p_original[i][1]+(secrets.randbits(3))*2))        
    return p_neu,p_original

def plot_spline(spline, label=None):

    fig, ax = plt.subplots()
    for i in range(len(spline)):
        plot_spline_2d(spline[i], ax=ax, label=label[i])

    # plot_spline_2d(spline, ax=ax, label=label)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.title('Catmull-Rom Spline Interpolation')
    plt.show()
    save_print = bool(input("Do you want to save the spline plot? (y/n): ").lower() == 'y')
    if save_print:
        name = input("Enter the name of the file to save the plot (without extension): ")
        output_dir = os.path.join(os.path.dirname(__file__), 'Possible_curves')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{name}.png'))
        save_spline(spline, label)

    return 0

def plot_tangent(spline, label=None):
    t = np.linspace(0, 1, 10)
    fig, ax = plt.subplots() 
    # plot_tangent_2d(spline[i], t, ax=ax, label=label[i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.title('Catmull-Rom Spline Interpolation_tangent')
    plt.show()
    return 0

def save_spline(spline,label):
    output_dir = os.path.join(os.path.dirname(__file__), 'Possible_curves')
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(spline)):
        plot_spline_2d(spline[i], label=label[i])
        if bool(input(f"Do you want to save the {label[i]}? (y/n): ").lower() == 'y'):
            file_name = f"{label[i]}.txt"
            np.savetxt(os.path.join(output_dir, file_name), spline[i].points, delimiter=',')

    return 0

def main():
    p1= np.array([
    (-1, -0.5),
    (0, 2.3),
    (1, 1),
    (4, 1.3),
    (3.8, -0.2),
    (2.5, 0.1),
    ])

    points_number= int(input("Enter the number of points for the ellipse: \n"))

    p2= shape_creator(points_number)
    print("p1:", p1)
    print("shape_p1:", p1.shape)
    print("p2:", p2)
    print("shape_p2:", p2.shape)

    p1_changed, p1_original = point_changer(p1*5, int(input("Enter the level of change for p1 (1 or 2): \n")))
    p2_changed,p2_original = point_changer(p2, int(input("Enter the level of change for p2 (1 or 2): \n")))
    # print("p1_changed:", p1_changed)

    s1_0 = splines.CatmullRom(p1_original, endconditions='closed')
    s2_0= splines.CatmullRom(p2_original, endconditions='closed')
    
    s1 = splines.CatmullRom(p1_changed, endconditions='closed')
    s2 = splines.CatmullRom(p2_changed, endconditions='closed')
    
    spline_p1 = [s1_0, s1]
    label_unchanged = ['original Catmull-Rom Spline', 'changed Catmull-Rom Spline']
    
    spline_p2 = [s2_0, s2]
    label = ['original_ellipse-spline', 'changed_ellipse-spline']

    spline = np.vstack((spline_p1, spline_p2))
    label = np.vstack((label_unchanged, label))
    for i in range(len(spline)):
        plot_spline(spline[i], label=label[i])

    # plot_spline(spline, label=label)

    return 0

if __name__ == "__main__":
    main()
