from scipy.spatial.distance import cdist
import os
import sys
import numpy as np
import pickle
from tqdm.auto import trange

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from models.pipelines.pipeline_utils import make_sparse_tensor


def evaluate_sequence_reg(
    model, dataset_ref, dataset_query, ref_map_indices, query_map_indices, cfg
):
    closures = {}

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
        candidate_indices = np.where(feat_dists <= cfg.cd_thresh_max)[0]
        map_refs = ref_map_indices[candidate_indices]
        distances = feat_dists[candidate_indices]

        if len(candidate_indices) > 0:
            closures[query_map_indices[query_idx]] = (map_refs, distances)

    return closures
