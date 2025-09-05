from scipy.spatial.distance import cdist
import os
import sys
import numpy as np
from tqdm.auto import trange

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from models.pipelines.pipeline_utils import make_sparse_tensor


def evaluate_sequence_reg(model, dataset, map_indices, cfg):
    # Databases of previously visited/'seen' places.
    global_descriptors = np.zeros((len(dataset), 256))
    closures = {}

    for query_idx in trange(0, len(dataset), unit=" frames", dynamic_ncols=True):
        sparse_input = make_sparse_tensor(
            dataset[query_idx].astype(np.float32), cfg.voxel_size
        ).cuda()
        global_descriptor = model(sparse_input).cpu().detach().numpy().reshape(1, -1)
        global_descriptors[query_idx] = global_descriptor

        map_query = map_indices[query_idx]
        if (query_idx) < 100:
            continue

        feat_dists = cdist(
            global_descriptor,
            global_descriptors[: query_idx - 100 + 1],
            metric=cfg.eval_feature_distance,
        ).reshape(-1)
        candidate_indices = np.where(feat_dists <= cfg.cd_thresh_max)[0]
        map_refs = map_indices[candidate_indices]
        distances = feat_dists[candidate_indices]

        keep_indices = np.where(map_query - map_refs > 3)[0]

        if len(keep_indices) > 0:
            closures[map_query] = (map_refs[keep_indices], distances[keep_indices])

    return global_descriptors, closures
