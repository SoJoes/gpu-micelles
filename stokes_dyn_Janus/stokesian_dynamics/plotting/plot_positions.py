#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Adam Townsend, adam@adamtownsend.com, 07/06/2017

"""Plot particles at a given frame number for an NPZ file specified in the
script.

Does not plot any periodic copies. If you want to do this, see the code in
plot_particle_positions_video.py.
"""

# Useful command:  scp -r dbxl46@ncc1.clients.dur.ac.uk:pytential_stokes/pytential_stokes/stokes_dyn_Janus/stokesian_dynamics/frame_output /mnt/c/Users/sj000/Desktop/frame_output

import matplotlib.pyplot as plt
import numpy as np
import zarr
import os
import sys
from pylab import rcParams
from sumpy.visualization import FieldPlotter
sys.path.append("../stokesian_dynamics")  # Allows importing from SD directory

from functions.graphics import (plot_all_spheres, plot_all_dumbbells,
                                plot_all_torque_lines, plot_all_velocity_lines,
                                plot_all_angular_velocity_lines)


def get_latest_file(folder_path, n=1):
    # Get all the files in the folder
    files = os.listdir(folder_path)

    # Sort the files by their last modified time
    files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))

    # Get the latest file
    latest_file = files[-num_files:]

    return latest_file

args = sys.argv[1:]
num_files = 1
start_frame = 0
frame_folder = "frame_output"
folder_path = 'output'

if len(args) > 0:
    num_files = int(args[0])
if len(args) > 1:
    start_frame = int(args[1])
if len(args) > 2:
    frame_folder = str(args[2])
if len(args) > 3:
    folder_path = str(args[3])

latest_file = get_latest_file(folder_path, n=num_files)
print("LATEST:" + str(num_files))
print(latest_file)

graph_title = "testing"
viewing_angle = (0, -90)
viewbox_bottomleft_topright = np.array([[-5, -5, -5], [5, 5, 5]])
two_d_plot = True
view_labels = False
trace_paths = 0

# Naming the folders like this means you can run this script from any directory
#this_folder = os.path.dirname(os.path.abspath(__file__))
#output_folder = this_folder + "/../output/"

#data1 = np.load(output_folder + filename + ".npz")

pos_centres_list = []
pos_deltax_list = []
Fa_out_list = []
Fb_out_list = []
DFb_out_list = []
particle_rotations_list = []

pot_list = []
indicator_list = []
nabla_pot_x_list = []
nabla_pot_y_list = []
T_xx_list = []
T_yy_list = []
T_xy_list = []

for file in latest_file:
    filename = folder_path + '/' + file
    data1 = zarr.open(filename, mode="r")

    pos_centres_list.append(data1['centres'][:])
    pos_deltax_list.append(data1['deltax'][:])
    Fa_out_list.append(data1['Fa'][:])
    Fb_out_list.append(data1['Fb'][:])
    DFb_out_list.append(data1['DFb'][:])
    particle_rotations_list.append(data1['sphere_rotations'][:])

    pot_list.append(data1['pot'][:])
    indicator_list.append(data1['indicator'][:])
    nabla_pot_x_list.append(data1['nabla_pot_x'][:])
    nabla_pot_y_list.append(data1['nabla_pot_y'][:])
    T_xx_list.append(data1['T_xx'][:])
    T_yy_list.append(data1['T_yy'][:])
    T_xy_list.append(data1['T_xy'][:])

    print(file)
    print("has number of frames:", data1['centes'][:].shape)

positions_centres = np.concatenate(pos_centres_list, axis=0)
positions_deltax = np.concatenate(pos_deltax_list, axis=0)
Fa_out = np.concatenate(Fa_out_list, axis=0)
Fb_out = np.concatenate(Fb_out_list, axis=0)
DFb_out = np.concatenate(DFb_out_list, axis=0)
particle_rotations = np.concatenate(particle_rotations_list, axis=0)

pot = np.concatenate(pot_list, axis=0)
indicator = np.concatenate(indicator_list, axis=0)
nabla_pot_x = np.concatenate(nabla_pot_x_list, axis=0)
nabla_pot_y = np.concatenate(nabla_pot_y_list, axis=0)
T_xx = np.concatenate(T_xx_list, axis=0)
T_xy = np.concatenate(T_xy_list, axis=0)
T_yy = np.concatenate(T_yy_list, axis=0)

num_frames = positions_centres.shape[0]
num_particles = positions_centres.shape[1]
num_dumbbells = positions_deltax.shape[1]
num_spheres = num_particles - num_dumbbells

fplot = FieldPlotter(np.zeros(2), extent=5, npoints=500)
frameno = T_xx.shape[0]

for frame in range(frameno):
    sphere_positions = positions_centres[frame, 0:num_spheres, :]
    sphere_rotations = particle_rotations[frame, 0:num_spheres, :]
    dumbbell_positions = positions_centres[frame, num_spheres:num_particles, :]
    dumbbell_deltax = positions_deltax[frame, :, :]

    sphere_sizes = np.array([1 for _ in range(num_spheres)])
    dumbbell_sizes = np.array([0.1 for _ in range(num_dumbbells)])
    Ta_out = [[0, 0, 0] for _ in range(num_spheres)]
    Oa_out = [[0, 0, 0] for _ in range(num_spheres)]
    Ua_out = [[0, 0, 0] for _ in range(num_spheres)]

    posdata = [sphere_sizes, sphere_positions, sphere_rotations, dumbbell_sizes,
               dumbbell_positions, dumbbell_deltax]
    previous_step_posdata = posdata

    # Pictures initialise
    rcParams.update({'font.size': 11})
    rcParams.update({'figure.dpi': 120, 'figure.figsize': [6, 6],
                     'savefig.dpi': 140})
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(viewing_angle[0], viewing_angle[1])
    spheres = list()
    dumbbell_lines = list()
    dumbbell_spheres = list()
    force_lines = list()
    force_text = list()
    torque_lines = list()
    velocity_lines = list()
    velocity_text = list()
    sphere_labels = list()
    angular_velocity_lines = list()
    sphere_lines = list()
    sphere_trace_lines = list()
    dumbbell_trace_lines = list()
    v = viewbox_bottomleft_topright.transpose()
    ax.auto_scale_xyz(v[0], v[1], v[2])
    ax.set_xlim3d(v[0, 0], v[0, 1])
    ax.set_ylim3d(v[1, 0], v[1, 1])
    ax.set_zlim3d(v[2, 0], v[2, 1])
    ax.set_box_aspect((1, 1, 1), zoom=1.4)
    if two_d_plot:
        ax.set_proj_type('ortho')
        ax.set_yticks([])
    else:
        ax.set_ylabel("$y$")
    ax.set_xlabel("$x$")
    ax.set_zlabel("$z$")
    fig.tight_layout()

    # Pictures
    if num_spheres > 0:
        (spheres, sphere_lines, sphere_trace_lines) = plot_all_spheres(
            ax, frame, posdata, previous_step_posdata, trace_paths,
            sphere_trace_lines, Fa_out[frame])
    if num_dumbbells > 0:
        (dumbbell_spheres, dumbbell_lines, dumbbell_trace_lines) = plot_all_dumbbells(
            ax, frame, posdata, trace_paths, dumbbell_trace_lines,
            Fb_out[frame], DFb_out[frame])
    if view_labels:
        torque_lines = plot_all_torque_lines(ax, posdata, Ta_out)
        (velocity_lines, velocity_text, sphere_labels) = plot_all_velocity_lines(
            ax, posdata, Ua_out)  # Velocity in green
        angular_velocity_lines = plot_all_angular_velocity_lines(
            ax, posdata, Oa_out)  # Angular velocity in white with green edging

    for q in (dumbbell_lines):
        q.remove()

    ax.set_title("  frame "
                 + ("{:" + str(len(str(num_frames))) + ".0f}").format(start_frame + frame)
                 + "/" + str(num_frames-1), loc='left', y=0.97, fontsize=11)
    ax.set_title(graph_title, loc='center', y=1.055, fontsize=11)
    plt.savefig(frame_folder + "/frame"+str(start_frame + frame)+".png")
    plt.close()


    fplot.write_vtk_file(frame_folder + "/frame"+str(start_frame + frame)+".vts", [
        ("potential", pot[frame,:]),
        ("indicator", indicator[frame,:]),
        ("nabla_pot_x", nabla_pot_x[frame,:]),
        ("nabla_pot_y", nabla_pot_y[frame,:]),
        ("T_xx_component", T_xx[frame,:]),
        ("T_xy_component", T_xy[frame,:]),
        ("T_yy_component", T_yy[frame,:]),
        ])