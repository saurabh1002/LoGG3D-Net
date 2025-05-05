from scipy.spatial.distance import cdist
import logging
import matplotlib.pyplot as plt
import pickle
import os
import sys
import numpy as np
import math
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from models.pipelines.pipeline_utils import *
from utils.data_loaders.make_dataloader import *
from utils.misc_utils import *
from utils.data_loaders.mulran.mulran_dataset import (
    load_poses_from_csv,
    load_timestamps_csv,
)

__all__ = ["evaluate_sequence_reg"]


def save_pickle(data_variable, file_name):
    dbfile2 = open(file_name, "ab")
    pickle.dump(data_variable, dbfile2)
    dbfile2.close()
    logging.info(f"Finished saving: {file_name}")


def load_gt_poses(poses_file: str):
    """MuRan has more poses than scans, therefore we need to match 1-1 timestamp with pose"""

    self.scan_timestamps = [
        int(os.path.basename(t).split(".")[0]) for t in self.scan_files
    ]

    def read_csv(poses_file: str):
        poses = np.loadtxt(poses_file, delimiter=",")
        timestamps = poses[:, 0]
        poses = poses[:, 1:]
        n = poses.shape[0]
        poses = np.concatenate(
            (
                poses,
                np.zeros((n, 3), dtype=np.float32),
                np.ones((n, 1), dtype=np.float32),
            ),
            axis=1,
        )
        poses = poses.reshape((n, 4, 4))  # [N, 4, 4]
        return poses, timestamps

    # Read the csv file
    poses, timestamps = read_csv(poses_file)
    # Extract only the poses that has a matching Ouster scan
    correct_ids = [np.argmin(abs(timestamps - t)) for t in timestamps]
    poses = poses[correct_ids]
    timestamps = timestamps[correct_ids]

    # Convert from global coordinate poses to local poses
    first_pose = poses[0, :, :]
    poses = np.linalg.inv(first_pose) @ poses

    return poses, timestamps


def evaluate_sequence_reg(model, cfg):
    save_descriptors = cfg.eval_save_descriptors
    eval_seq = cfg.mulran_eval_seq

    test_loader, positions_database, timestamps = make_data_loader(cfg)

    iterator = test_loader.__iter__()
    logging.info(f"len_dataloader {len(test_loader.dataset)}")

    num_queries = len(positions_database)

    # Databases of previously visited/'seen' places.
    seen_poses, seen_descriptors, seen_feats = [], [], []

    # Store results of evaluation.
    min_min_dist = 1.0
    max_min_dist = 0.0
    start_time = timestamps[0]

    for query_idx in tqdm(range(num_queries)):
        input_data = next(iterator)
        lidar_pc = input_data[0]  # .cpu().detach().numpy()
        if not len(lidar_pc) > 0:
            logging.info(f"Corrupt cloud id: {query_idx}")
            continue
        sparse_input = make_sparse_tensor(lidar_pc, cfg.voxel_size).cuda()

        output_desc = model(sparse_input)  # .squeeze()
        global_descriptor = output_desc.cpu().detach().numpy()

        global_descriptor = np.reshape(global_descriptor, (1, -1))
        query_pose = positions_database[query_idx]
        query_time = timestamps[query_idx]

        if len(global_descriptor) < 1:
            continue
        
        seen_descriptors.append(global_descriptor)
        seen_poses.append(query_pose)

        if (query_idx) < 100:
            continue

        # Build retrieval database using entries 30s prior to current query.
        db_seen_descriptors = np.asarray(seen_descriptors[:-100])
        db_seen_descriptors = db_seen_descriptors.reshape(
            -1, np.shape(global_descriptor)[1]
        )

        # Find top-1 candidate.
        nearest_idx = 0
        min_dist = math.inf

        feat_dists = cdist(
            global_descriptor, db_seen_descriptors, metric=cfg.eval_feature_distance
        ).reshape(-1)
        min_dist, nearest_idx = np.min(feat_dists), np.argmin(feat_dists)

        place_candidate = seen_poses[nearest_idx]

        if min_dist < min_min_dist:
            min_min_dist = min_dist
        if min_dist > max_min_dist:
            max_min_dist = min_dist

    if save_descriptors:
        save_dir = os.path.join(os.path.dirname(__file__), str(eval_seq))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        desc_file_name = "/logg3d_descriptor.pickle"
        save_pickle(seen_descriptors, save_dir + desc_file_name)

    return None
