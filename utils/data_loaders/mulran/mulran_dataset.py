import os
import sys
import glob
import random
import numpy as np
import logging
import json
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from utils.o3d_tools import *
from utils.data_loaders.pointcloud_dataset import *


class MulRanDataset(PointCloudDataset):
    r"""
    Generate single pointcloud frame from MulRan dataset.
    """

    def __init__(self, config=None):

        self.root = config.mulran_dir

        PointCloudDataset.__init__(self)

        self.sequence_dir = os.path.join(self.root, config.mulran_eval_seq)
        self.velodyne_dir = os.path.join(self.sequence_dir, "Ouster/")

        self.files = sorted(glob.glob(self.velodyne_dir + "*.bin"))
        self.scan_timestamps = [int(os.path.basename(t).split(".")[0]) for t in self.files]
        self.gt_poses = self.load_gt_poses(os.path.join(self.sequence_dir, "global_pose.csv"))

    def load_gt_poses(self, poses_file: str):
        """MuRan has more poses than scans, therefore we need to match 1-1 timestamp with pose"""

        def read_csv(poses_file: str):
            poses = np.loadtxt(poses_file, delimiter=",")
            timestamps = poses[:, 0]
            poses = poses[:, 1:]
            n = poses.shape[0]
            poses = np.concatenate(
                (poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)),
                axis=1,
            )
            poses = poses.reshape((n, 4, 4))  # [N, 4, 4]
            return poses, timestamps

        # Read the csv file
        poses, timestamps = read_csv(poses_file)
        # Extract only the poses that has a matching Ouster scan
        poses = poses[[np.argmin(abs(timestamps - t)) for t in self.scan_timestamps]]

        # Convert from global coordinate poses to local poses
        first_pose = poses[0, :, :]
        poses = np.linalg.inv(first_pose) @ poses

        T_lidar_to_base, T_base_to_lidar = self._get_calibration()
        return T_lidar_to_base @ poses @ T_base_to_lidar

    def _get_calibration(self):
        # Apply calibration obtainbed from calib_base2ouster.txt
        #  T_lidar_to_base[:3, 3] = np.array([1.7042, -0.021, 1.8047])
        #  T_lidar_to_base[:3, :3] = tu_vieja.from_euler(
        #  "xyz", [0.0001, 0.0003, 179.6654], degrees=True
        #  )
        T_lidar_to_base = np.array(
            [
                [-9.9998295e-01, -5.8398386e-03, -5.2257060e-06, 1.7042000e00],
                [5.8398386e-03, -9.9998295e-01, 1.7758769e-06, -2.1000000e-02],
                [-5.2359878e-06, 1.7453292e-06, 1.0000000e00, 1.8047000e00],
                [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
            ]
        )
        T_base_to_lidar = np.linalg.inv(T_lidar_to_base)
        return T_lidar_to_base, T_base_to_lidar

    def get_pointcloud_tensor(self, pc_id):
        xyzr = np.fromfile(self.files[pc_id], dtype=np.float32).reshape(-1, 4)
        range = np.linalg.norm(xyzr[:, :3], axis=1)
        range_filter = np.logical_and(range > 0.1, range < 80)
        xyzr = xyzr[range_filter]
        return xyzr

    def __getitem__(self, idx):
        xyz0_th = self.get_pointcloud_tensor(idx)
        return xyz0_th


#####################################################################################
# Load poses
#####################################################################################


def load_poses_from_csv(file_name):
    with open(file_name, newline="") as f:
        reader = csv.reader(f)
        data_poses = list(reader)

    transforms = []
    positions = []
    for cnt, line in enumerate(data_poses):
        line_f = [float(i) for i in line]
        P = np.vstack((np.reshape(line_f[1:], (3, 4)), [0, 0, 0, 1]))
        transforms.append(P)
        positions.append([P[0, 3], P[1, 3], P[2, 3]])
    return np.asarray(transforms), np.asarray(positions)


#####################################################################################
# Load timestamps
#####################################################################################


def load_timestamps_csv(file_name):
    with open(file_name, newline="") as f:
        reader = csv.reader(f)
        data_poses = list(reader)
    data_poses_ts = np.asarray([float(t) / 1e9 for t in np.asarray(data_poses)[:, 0]])
    return data_poses_ts
