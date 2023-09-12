import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from pyntcloud import PyntCloud

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

MAX_CHAMFER_DIST_FOR_DISPLAY = 100.
MIN_POINTS_FOR_VALID_DIST = 75000

chamfer_dist = {}
total_chamfer_dist = {}
chamfer_dist_valid_only = {}
total_chamfer_dist_valid_only = {}
num_points_in_pc = {}

for obj_id in obj_id_to_name.keys():
    assert (obj_id in obj_id_to_name)

    chamfer_dist[obj_id] = {}
    total_chamfer_dist[obj_id] = {}
    chamfer_dist_valid_only[obj_id] = {}
    total_chamfer_dist_valid_only[obj_id] = {}
    num_points_in_pc[obj_id] = {}

    experiment_root_folder_prefix = ("/cluster/scratch/fmilano/new_datasets/"
                                     f"{obj_id}")

    all_bounds = [
        os.path.basename(p).split("_bound_")[-1] for p in sorted(
            glob.glob(f"{experiment_root_folder_prefix}_trained_on_bound*"))
    ]

    for bound in all_bounds:
        experiment_name = (f"{experiment_root_folder_prefix}_trained_on_bound_"
                           f"{bound}")
        if (not os.path.exists(
                os.path.join(experiment_name, "chamfer_distance.txt"))):
            continue
        with open(os.path.join(experiment_name, "chamfer_distance.txt"),
                  "r") as f:
            lines = f.readlines()
            curr_chamfer_dist = float(lines[1].strip())
            try:
                curr_total_chamfer_dist = float(lines[2].strip())
            except IndexError:
                import pdb
                pdb.set_trace()
            if (curr_chamfer_dist < MAX_CHAMFER_DIST_FOR_DISPLAY and
                    curr_total_chamfer_dist < MAX_CHAMFER_DIST_FOR_DISPLAY):
                chamfer_dist[obj_id][float(bound)] = curr_chamfer_dist
                total_chamfer_dist[obj_id][float(
                    bound)] = curr_total_chamfer_dist

        # Read point cloud and count number of points.
        point_cloud = PyntCloud.from_file(
            os.path.join(experiment_name, "full_point_cloud.ply"))

        num_points_in_pc[obj_id][float(bound)] = len(point_cloud.points)

        if (num_points_in_pc[obj_id][float(bound)]
                >= MIN_POINTS_FOR_VALID_DIST):
            chamfer_dist_valid_only[obj_id][float(bound)] = curr_chamfer_dist
            total_chamfer_dist_valid_only[obj_id][float(
                bound)] = curr_total_chamfer_dist

diff_fw_per_bound_per_obj = {}
diff_tot_per_bound_per_obj = {}
for selected_bound in range(40, 96, 5):
    selected_bound = float(f"{selected_bound / 100.:.2f}")
    curr_str = f"Bound {selected_bound}:\n"
    try:
        diff_fw_per_bound_per_obj[selected_bound] = {}
        diff_tot_per_bound_per_obj[selected_bound] = {}
        for obj_id in obj_id_to_name.keys():
            diff_fw_per_bound_per_obj[selected_bound][obj_id] = (
                chamfer_dist_valid_only[obj_id][selected_bound] -
                np.array(list(chamfer_dist_valid_only[obj_id].values())).min())
            diff_tot_per_bound_per_obj[selected_bound][
                obj_id] = total_chamfer_dist_valid_only[obj_id][
                    selected_bound] - np.array(
                        list(total_chamfer_dist_valid_only[obj_id].values())
                    ).min()
            curr_diff_fw = diff_fw_per_bound_per_obj[selected_bound][obj_id]
            curr_tot_diff_fw = diff_tot_per_bound_per_obj[selected_bound][
                obj_id]
            curr_str += (
                f"\t`{obj_id_to_name[obj_id]}`: FW: {curr_diff_fw:.1f} mm - "
                f"TOT: {curr_tot_diff_fw:.1f} mm\n")
        curr_avg_fw_per_obj = np.array(
            list(diff_fw_per_bound_per_obj[selected_bound].values())).mean()
        curr_avg_tot_per_obj = np.array(
            list(diff_tot_per_bound_per_obj[selected_bound].values())).mean()
        print(f"{curr_str}AVG FW = {curr_avg_fw_per_obj}, AVG TOT = "
              f"{curr_avg_tot_per_obj}")
    except KeyError:
        continue
