import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


# Evaluation
eval_arg = add_argument_group('Eval')
eval_arg.add_argument('--eval_pipeline', type=str, default='LOGG3D')
eval_arg.add_argument('--mulran_eval_seq', type=str,
                      default='KAIST01')
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
# KittiDataset #MulRanDataset
data_arg.add_argument('--eval_dataset', type=str, default='MulRanDataset')
data_arg.add_argument('--collation_type', type=str,
                      default='default')  # default#sparcify_list
data_arg.add_argument("--eval_save_descriptors", type=str2bool, default=False)
data_arg.add_argument('--num_points', type=int, default=80000)
data_arg.add_argument('--voxel_size', type=float, default=0.10)
data_arg.add_argument('--eval_feature_distance', type=str,
                      default='cosine')  # cosine#euclidean

data_arg.add_argument('--mulran_dir', type=str,
                      default='/mnt/gupta/datasets/MulRAN/', help="Path to the MulRan dataset")

def get_config_eval():
    args = parser.parse_args()
    return args
