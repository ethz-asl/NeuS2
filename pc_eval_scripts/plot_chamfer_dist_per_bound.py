import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from pyntcloud import PyntCloud

parser = argparse.ArgumentParser()

parser.add_argument('--obj_id', type=str, required=True)

args = parser.parse_args()

obj_id = args.obj_id

obj_id_to_name = {
    "000001": "ape",
    "000005": "can",
    "000006": "cat",
    "000008": "driller",
    "000009": "duck",
    "000010": "eggbox",
    "000011": "glue",
    "000012": "holepuncher"
}

assert (obj_id in obj_id_to_name)

chamfer_dist = {}
total_chamfer_dist = {}
chamfer_dist_valid_only = {}
total_chamfer_dist_valid_only = {}
num_points_in_pc = {}

experiment_root_folder_prefix = ("/cluster/scratch/fmilano/new_datasets/"
                                 f"{obj_id}")

all_bounds = [
    os.path.basename(p).split("_bound_")[-1] for p in sorted(
        glob.glob(f"{experiment_root_folder_prefix}_trained_on_bound*"))
]

MAX_CHAMFER_DIST_FOR_DISPLAY = 100.
MIN_POINTS_FOR_VALID_DIST = 75000

for bound in all_bounds:
    experiment_name = (f"{experiment_root_folder_prefix}_trained_on_bound_"
                       f"{bound}")
    if (not os.path.exists(os.path.join(experiment_name,
                                        "chamfer_distance.txt"))):
        continue
    with open(os.path.join(experiment_name, "chamfer_distance.txt"), "r") as f:
        lines = f.readlines()
        curr_chamfer_dist = float(lines[1].strip())
        try:
            curr_total_chamfer_dist = float(lines[2].strip())
        except IndexError:
            import pdb
            pdb.set_trace()
        if (curr_chamfer_dist < MAX_CHAMFER_DIST_FOR_DISPLAY and
                curr_total_chamfer_dist < MAX_CHAMFER_DIST_FOR_DISPLAY):
            chamfer_dist[float(bound)] = curr_chamfer_dist
            total_chamfer_dist[float(bound)] = curr_total_chamfer_dist

    # Read point cloud and count number of points.
    point_cloud = PyntCloud.from_file(
        os.path.join(experiment_name, "full_point_cloud.ply"))

    num_points_in_pc[float(bound)] = len(point_cloud.points)

    if (num_points_in_pc[float(bound)] >= MIN_POINTS_FOR_VALID_DIST):
        chamfer_dist_valid_only[float(bound)] = curr_chamfer_dist
        total_chamfer_dist_valid_only[float(bound)] = curr_total_chamfer_dist

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel("Point cloud extent relative to 2*(scene bounds).")
ax1.set_ylabel("Chamfer distance [mm]", color=color)
ax1.plot(chamfer_dist.keys(),
         chamfer_dist.values(),
         '--',
         color=color,
         label="Forward Chamfer distance")
ax1.plot(total_chamfer_dist.keys(),
         total_chamfer_dist.values(),
         color=color,
         label="Total Chamfer distance")

ax2 = ax1.twinx()

color = 'tab:orange'
ax2.set_ylabel("#points in point cloud", color=color)
ax2.plot(num_points_in_pc.keys(), num_points_in_pc.values(), color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax1.grid(which='both')
ax1.set_title(f"Object {obj_id} ({obj_id_to_name[obj_id]})")
fig.legend()
ax1.minorticks_on()
fig.tight_layout()

plt.grid(which='both')
plt.title(f"Object {obj_id} ({obj_id_to_name[obj_id]})")
plt.ylim(bottom=0,
         top=np.max([
             1.25 * np.min(list(total_chamfer_dist.values())).astype(int),
             np.min([5,
                     np.max(list(total_chamfer_dist.values())).astype(int)])
         ]))
plt.legend()
plt.minorticks_on()
plt.show()
fig.set_size_inches(12, 8)
plt.savefig(os.path.join(os.path.dirname(experiment_root_folder_prefix),
                         f"{args.obj_id}_chamfer_dist.png"),
            dpi=300)
