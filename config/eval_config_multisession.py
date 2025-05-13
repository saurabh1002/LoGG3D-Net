import argparse
from pathlib import Path

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Evaluation
eval_arg = add_argument_group('Eval')
eval_arg.add_argument('--eval_pipeline', type=str, default='LOGG3D')
eval_arg.add_argument('--checkpoint_name', type=str,
                      default='/mulran_10cm/2021-09-14_08-59-00_3n24h_MulRan_v10_q29_4s_263039.pth')
eval_arg.add_argument("--cd_thresh_min", default=0.001,
                      type=float, help="Thresholds on cosine-distance to top-1.")
eval_arg.add_argument("--cd_thresh_max", default=1.0,
                      type=float, help="Thresholds on cosine-distance to top-1.")
eval_arg.add_argument("--num_thresholds", default=1000, type=int,
                      help="Number of thresholds. Number of points on PR curve.")


# Dataset specific configurations
data_arg = add_argument_group('Data')

data_arg.add_argument('--voxel_size', type=float, default=0.10)
data_arg.add_argument('--eval_feature_distance', type=str,
                      default='cosine')  # cosine#euclidean

data_arg.add_argument('--dataloader_ref', type=str, required=True, help="Dataloader name")
data_arg.add_argument('--data_dir_ref', type=Path, required=True, help="Path to the dataset")
data_arg.add_argument('--sequence_ref', type=str, default=None, help="Sequence Name")
data_arg.add_argument('--dataloader_query', type=str, required=True, help="Dataloader name")
data_arg.add_argument('--data_dir_query', type=Path, required=True, help="Path to the dataset")
data_arg.add_argument('--sequence_query', type=str, default=None, help="Sequence Name")
data_arg.add_argument('--results_dir', type=Path, required=True, help="Path to the results directory")

def get_config_eval():
    args = parser.parse_args()
    return args
