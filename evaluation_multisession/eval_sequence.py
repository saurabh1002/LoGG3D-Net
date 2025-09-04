from scipy.spatial.distance import cdist
import os
import sys
import numpy as np
import pickle
from tqdm.auto import trange

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from models.pipelines.pipeline_utils import make_sparse_tensor


def scan_to_map(
    scan_query, scan_ref, query_local_maps_scan_range, ref_local_maps_scan_range
):
    map_query = np.where(
        (scan_query >= query_local_maps_scan_range[:, 0])
        & (scan_query < query_local_maps_scan_range[:, 1])
    )[0][0]
    map_ref = np.where(
        (scan_ref >= ref_local_maps_scan_range[:, 0])
        & (scan_ref < ref_local_maps_scan_range[:, 1])
    )[0][0]
    return map_query, map_ref


def evaluate_sequence_reg(model, dataset_ref, dataset_query, cfg):
    closures_list = []
    distances_list = []

    # Databases of previously visited/'seen' places.
    ref_descriptors_file = os.path.join(
        dataset_ref.sequence_dir, "logg3d_descriptors.pkl"
    )
    with open(ref_descriptors_file, "rb") as dbfile2:
        ref_descriptors = pickle.load(dbfile2)

    db_descriptors = np.asarray(ref_descriptors).reshape(-1, 256)

    for query_idx in trange(0, len(dataset_query), unit=" frames", dynamic_ncols=True):
        sparse_input = make_sparse_tensor(
            dataset_query[query_idx].astype(np.float32), cfg.voxel_size
        ).cuda()
        global_descriptor = model(sparse_input).cpu().detach().numpy().reshape(1, -1)

        # Find top-1 candidate.
        feat_dists = cdist(
            global_descriptor, db_descriptors, metric=cfg.eval_feature_distance
        ).reshape(-1)
        min_dist, ref_idx = np.min(feat_dists), np.argmin(feat_dists)

        map_query, map_ref = scan_to_map(
            query_idx,
            ref_idx,
            dataset_query.local_maps_scan_range,
            dataset_ref.local_maps_scan_range,
        )
        closures_list.append((map_ref, map_query))
        distances_list.append(min_dist)

    return closures_list, distances_list
