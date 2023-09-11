import argparse
from chamferdist import ChamferDistance
import numpy as np
import os
from pyntcloud import PyntCloud
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--nerf_pc_path',
                    type=str,
                    required=True,
                    help=("Path to the NeRF point cloud, assumed to be already "
                          "in the BOP coordinate frame and scale."))
parser.add_argument(
    '--bop_gt_pc_path',
    type=str,
    required=True,
    help=("Path to the ground-truth point cloud from BOP (e.g., generated "
          "through SurfEmb scripts). NOTE: The unit of measure of this point "
          "cloud is assumed to be mm."))
parser.add_argument('--visualize', action='store_true')

args = parser.parse_args()

# Load point clouds.
source_pc = PyntCloud.from_file(args.nerf_pc_path)
target_pc = PyntCloud.from_file(args.bop_gt_pc_path)

if (args.visualize):
    import open3d as o3d
    source_pc_o3d = o3d.geometry.PointCloud()
    source_pc_o3d.colors = o3d.utility.Vector3dVector(
        np.stack([
            source_pc.points["red"], source_pc.points["green"],
            source_pc.points["blue"]
        ],
                 axis=-1) / 255.)
    target_pc_o3d = o3d.geometry.PointCloud()
    target_pc_o3d.colors = o3d.utility.Vector3dVector(
        np.ones_like(np.asarray(target_pc_o3d.points)))

source_pc = np.stack(
    [source_pc.points["x"], source_pc.points["y"], source_pc.points["z"]],
    axis=-1)
target_pc = np.stack(
    [target_pc.points["x"], target_pc.points["y"], target_pc.points["z"]],
    axis=-1)

if (args.visualize):
    source_pc_o3d.points = o3d.utility.Vector3dVector(source_pc)
    target_pc_o3d.points = o3d.utility.Vector3dVector(target_pc)

    o3d.visualization.draw([source_pc_o3d, target_pc_o3d])

# Compute the Chamfer distance.
with torch.no_grad():
    source_pc = torch.from_numpy(source_pc)[None].float().cuda()
    target_pc = torch.from_numpy(target_pc)[None].float().cuda()
    chamferDist = ChamferDistance()
    dist_forward = chamferDist(source_pc, target_pc).item() / len(source_pc[0])
    dist_backward = chamferDist(target_pc, source_pc).item() / len(target_pc[0])
    total_dist = dist_forward + dist_backward
    print(
        f"\033[1mForward distance = {dist_forward:.1f} mm\033[0m, backward "
        f"distance = {dist_backward:.1f} mm, total distance = {total_dist:.1f} "
        "mm.")

with open(
        os.path.join(os.path.dirname(args.nerf_pc_path),
                     "chamfer_distance.txt"), "w") as f:
    f.write(f"{args.bop_gt_pc_path}\n{dist_forward:.1f}\n{total_dist:.1f}\n")