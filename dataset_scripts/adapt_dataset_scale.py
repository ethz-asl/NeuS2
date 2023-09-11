#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import copy
import glob
import json
import numpy as np
import open3d as o3d
import os
import pandas as pd
import sys
import time

from pyntcloud import PyntCloud
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.common import ROOT_DIR, repl

import pyngp as ngp  # noqa

parser = argparse.ArgumentParser(description=(
    "Run neural graphics primitives testbed with additional configuration & "
    "output options"))

parser.add_argument("--name", type=str, required=True)
parser.add_argument(
    "--scene",
    type=str,
    required=True,
    help=("The scene to load. Can be the scene's name or a full path to the "
          "training data."))
parser.add_argument("--n_steps",
                    type=int,
                    required=True,
                    help="Number of steps to train for before quitting.")

parser.add_argument(
    '--bound_extent',
    type=float,
    required=True,
    help=("Extent of the scene bounds (between 0 and 1) that should contain "
          "the object."))
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--should_convert_to_mm', action='store_true')

args = parser.parse_args()

output_path = os.path.join('output', args.name)
os.makedirs(os.path.join(output_path, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(output_path, "mesh"), exist_ok=True)

time_name = time.strftime("%m_%d_%H_%M", time.localtime())

mode = ngp.TestbedMode.Nerf
configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")

base_network = os.path.join(configs_dir, "base.json")

testbed = ngp.Testbed(mode)

scene = args.scene
testbed.load_training_data(scene)

testbed.reload_network_from_file(base_network)

ref_transforms = {}

testbed.shall_train = True

testbed.nerf.render_with_camera_distortion = True

near_distance = -1

if (near_distance >= 0.0):
    print("NeRF training ray near_distance ", near_distance)
    testbed.nerf.training.near_distance = near_distance

old_training_step = 0
tqdm_last_update = 0
n_steps = args.n_steps

snapshot_path = os.path.join(output_path, "checkpoints", f"{n_steps}.msgpack")

with tqdm(desc="Training", total=n_steps, unit="step") as t:
    while testbed.frame():
        if testbed.want_repl():
            repl(testbed)
        # What will happen when training is done?
        if testbed.training_step >= n_steps:
            break
        # Update progress bar
        if testbed.training_step < old_training_step or old_training_step == 0:
            old_training_step = 0
            t.reset()

        now = time.monotonic()
        if now - tqdm_last_update > 0.1:
            t.update(testbed.training_step - old_training_step)
            t.set_postfix(loss=testbed.loss)
            old_training_step = testbed.training_step
            tqdm_last_update = now

print("Saving snapshot ", snapshot_path)
testbed.save_snapshot(snapshot_path, False)

pc_mc = testbed.compute_marching_cubes_pc(resolution=np.array([512, 512, 512]))

pc_mc_o3d = o3d.geometry.PointCloud()

# There are some degenerate points assigned to the origin. -> Remove them.
points_not_assigned_origin = np.where(
    np.logical_or.reduce((pc_mc['V'][:, 0] != 0, pc_mc['V'][:, 1]
                          != 0, pc_mc['V'][:, 2] != 0)))

# Center point cloud in the origin (`instant-ngp` uses [0, 1] bounds).
pc_mc_points = pc_mc['V'][points_not_assigned_origin] - testbed.aabb.center()
pc_mc_colors = pc_mc['C'][points_not_assigned_origin]

num_points_downsampled = 75000
random_permutation_mc = np.random.permutation(len(pc_mc_points))

_J_bop = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.], [0., -1., 0., 0.],
                   [0., 0., 0., 1.]])
W_BOP_T_W_NeRF = np.linalg.inv(_J_bop)

# Bring to NeRF coordinates.
pc_mc_points = (pc_mc_points[random_permutation_mc[:num_points_downsampled]]
                @ W_BOP_T_W_NeRF[:3, :3].T)
pc_mc_colors = pc_mc_colors[random_permutation_mc[:num_points_downsampled]]

# Save the point cloud.
point_cloud_NeRF_dict = {
    "x": pc_mc_points[..., 0],
    "y": pc_mc_points[..., 1],
    "z": pc_mc_points[..., 2]
}
# - Include color if present.
assert (pc_mc_colors.ndim == 2)
if (pc_mc_colors.shape[-1] != 0):
    assert (pc_mc_colors.shape[-1] == 3)
    point_cloud_NeRF_dict["red"] = (255 * pc_mc_colors[..., 0]).astype(np.uint8)
    point_cloud_NeRF_dict["green"] = (255 * pc_mc_colors[..., 1]).astype(
        np.uint8)
    point_cloud_NeRF_dict["blue"] = (255 * pc_mc_colors[..., 2]).astype(
        np.uint8)
point_cloud_NeRF = PyntCloud(pd.DataFrame(data=point_cloud_NeRF_dict))
point_cloud_NeRF_path = os.path.join(output_path, "full_point_cloud.ply")
point_cloud_NeRF.to_file(point_cloud_NeRF_path)

print(f"\033[94mSaved point cloud in NeRF scale to '{point_cloud_NeRF_path}'."
      "\033[0m")

if (args.should_convert_to_mm):
    pc_mc_points = pc_mc_points / testbed.nerf.training.dataset.scale * 1000.
    point_cloud_mm_dict = {
        "x": pc_mc_points[..., 0],
        "y": pc_mc_points[..., 1],
        "z": pc_mc_points[..., 2]
    }
    for color in ["red", "green", "blue"]:
        point_cloud_mm_dict[color] = point_cloud_NeRF_dict[color]
    point_cloud_mm = PyntCloud(pd.DataFrame(data=point_cloud_mm_dict))
    point_cloud_mm_path = os.path.join(output_path, "full_point_cloud_mm.ply")
    point_cloud_mm.to_file(point_cloud_mm_path)

    print(f"\033[94mSaved point cloud in mm to '{point_cloud_mm_path}'.\033[0m")

B = 0.5

assert (np.all(testbed.aabb.center() == testbed.nerf.training.dataset.offset))
assert (np.all(testbed.aabb.center() == [B, B, B]))
assert (np.all(testbed.aabb.min == [0., 0., 0.]) and
        np.all(testbed.aabb.max == [2 * B, 2 * B, 2 * B]))

if (args.visualize):
    assert (not args.should_convert_to_mm)
    pc_mc_o3d.points = o3d.utility.Vector3dVector(pc_mc_points)
    pc_mc_o3d.colors = o3d.utility.Vector3dVector(pc_mc_colors)

    points = [[-B, -B, -B], [-B, -B, B], [-B, B, -B], [-B, B, B], [B, -B, -B],
              [B, -B, B], [B, B, -B], [B, B, B]]

    lines = []
    for idx_i in range(len(points)):
        for idx_j in range(idx_i + 1, len(points)):
            if (np.count_nonzero(
                    np.array(points[idx_i]) - np.array(points[idx_j])) == 1):
                lines.append([idx_i, idx_j])

    colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw([pc_mc_o3d, line_set])

# Find the bounds of the (downsampled) point cloud.
coords_max = pc_mc_points.max(axis=0)
coords_min = pc_mc_points.min(axis=0)

coords_center = (coords_max + coords_min) / 2.

one_uom_scene_to_m = 1. / testbed.nerf.training.dataset.scale

coords_center_in_m = coords_center * one_uom_scene_to_m

old_scale_to_new_scale = 2 * args.bound_extent * B / (coords_max -
                                                      coords_min).max()

# Create a transformed version of the dataset.
all_json_paths = testbed.get_json_paths()
assert (len(all_json_paths))
json_path = all_json_paths[0]
with open(json_path, "r") as f:
    orig_transform = json.load(f)

new_transform = copy.deepcopy(orig_transform)
new_transform['one_uom_scene_to_one_m'] /= old_scale_to_new_scale
new_transform['scale'] *= old_scale_to_new_scale
new_transform['integer_depth_scale'] /= old_scale_to_new_scale

for frame in new_transform['frames']:
    curr_W_T_C = np.asarray(frame['transform_matrix'])
    frame['transform_matrix'] = curr_W_T_C.tolist()

# Copy all the dataset data into the temporary folder.
new_dataset_folder = os.path.join(output_path, "new_dataset")
print(f"Copying original dataset to '{new_dataset_folder}'...")
os.makedirs(new_dataset_folder)

for folder_or_file in sorted(
        glob.glob(os.path.join(os.path.dirname(json_path), "*"))):
    if (folder_or_file[-5:] == ".json"):
        continue
    else:
        os.symlink(
            folder_or_file,
            os.path.join(new_dataset_folder, os.path.basename(folder_or_file)))

# Add the JSON file.
output_json_path = os.path.join(new_dataset_folder, os.path.basename(json_path))

with open(output_json_path, "w") as f:
    json.dump(new_transform, f, indent=4)