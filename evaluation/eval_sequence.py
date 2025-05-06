from scipy.spatial.distance import cdist
import os
import sys
import numpy as np
import math
from tqdm.auto import trange

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from models.pipelines.pipeline_utils import make_sparse_tensor


def scan_to_map(scan_query, scan_ref, local_maps_scan_range):
    map_query = np.where(
        (scan_query >= local_maps_scan_range[:, 0])
        & (scan_query < local_maps_scan_range[:, 1])
    )[0][0]
    map_ref = np.where(
        (scan_ref >= local_maps_scan_range[:, 0])
        & (scan_ref < local_maps_scan_range[:, 1])
    )[0][0]
    return map_query, map_ref


def evaluate_sequence_reg(model, dataset, cfg):
    # Databases of previously visited/'seen' places.
    seen_descriptors = []
    closures_list = []
    distances_list = []

    for query_idx in trange(0, len(dataset), unit=" frames", dynamic_ncols=True):
        sparse_input = make_sparse_tensor(dataset[query_idx].astype(np.float32), cfg.voxel_size).cuda()
        global_descriptor = model(sparse_input).cpu().detach().numpy().reshape(1, -1)
        seen_descriptors.append(global_descriptor)

        if (query_idx) < 100:
            continue

        # Build retrieval database using entries 30s prior to current query.
        db_seen_descriptors = np.asarray(seen_descriptors[:-100])
        db_seen_descriptors = db_seen_descriptors.reshape(
            -1, np.shape(global_descriptor)[1]
        )

        # Find top-1 candidate.
        nearest_idx = 0
        feat_dists = cdist(
            global_descriptor, db_seen_descriptors, metric=cfg.eval_feature_distance
        ).reshape(-1)
        min_dist, nearest_idx = np.min(feat_dists), np.argmin(feat_dists)

        map_query, map_ref = scan_to_map(
            query_idx, nearest_idx, dataset.local_maps_scan_range
        )
        if map_query - map_ref > 3:
            closures_list.append([map_ref, map_query])
            distances_list.append(min_dist)

    return seen_descriptors, closures_list, distances_list
