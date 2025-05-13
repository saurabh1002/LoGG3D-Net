import datetime
import os
import sys
import torch
import pickle
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from evaluation_multisession.eval_sequence import evaluate_sequence_reg
from rich import box
from rich.console import Console
from rich.table import Table


def create_results_dir(cfg, dataset_query, dataset_ref):
    query_dataset_name = (
        dataset_query.sequence_id
        if hasattr(dataset_query, "sequence_id")
        else os.path.basename(dataset_query.data_dir)
    )
    ref_dataset_name = (
        dataset_refsequence_id
        if hasattr(dataset_ref, "sequence_id")
        else os.path.basename(dataset_ref.data_dir)
    )
    results_dir = os.path.join(
        cfg.results_dir, f"{query_dataset_name}", f"{ref_dataset_name}"
    )
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

class EvaluationMetrics:
    def __init__(self, true_positives, false_positives, false_negatives):
        self.tp = true_positives
        self.fp = false_positives
        self.fn = false_negatives

        try:
            self.precision = self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            self.precision = np.nan

        try:
            self.recall = self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            self.recall = np.nan

        try:
            self.F1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        except ZeroDivisionError:
            self.F1 = np.nan


if __name__ == "__main__":
    from models.pipelines.LOGG3D import LOGG3D
    from config.eval_config_multisession import get_config_eval
    from utils.datasets import dataset_factory

    cfg = get_config_eval()

    # Get model
    model = LOGG3D(feature_dim=16)

    save_path = os.path.join(os.path.dirname(__file__), "../", "checkpoints")
    save_path = str(save_path) + cfg.checkpoint_name
    print("Loading checkpoint from: ", save_path)
    checkpoint = torch.load(save_path)  # ,map_location='cuda:0')
    model.load_state_dict(checkpoint["model_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    model = model.cuda()
    model.eval()

    dataset_ref = dataset_factory(
        dataloader=cfg.dataloader_query,
        data_dir=cfg.data_dir_query,
        sequence=cfg.sequence_query,
    )
    dataset_query = dataset_factory(
        dataloader=cfg.dataloader_query,
        data_dir=cfg.data_dir_query,
        sequence=cfg.sequence_query,
    )

    closures_list, distances_list = evaluate_sequence_reg(
        model, dataset_ref, dataset_query, cfg
    )
    results_dir = create_results_dir(cfg, dataset_query, dataset_ref)

    print("[INFO] Computing Loop Closure Evaluation Metrics")
    thresholds = np.linspace(
        cfg.cd_thresh_min, cfg.cd_thresh_max, int(cfg.num_thresholds)
    )

    metrics = []
    gt_closures = set(map(lambda x: tuple(sorted(x)), dataset.gt_closure_indices))
    for threshold in thresholds:
        closures = set()
        for closure_indices, distance in zip(closures_list, distances_list):
            if distance < threshold:
                closures.add(closure_indices)

        tp = len(gt_closures.intersection(closures))
        fp = len(closures) - tp
        fn = len(gt_closures) - tp
        metrics.append(EvaluationMetrics(tp, fp, fn))

    table = Table(box=box.HORIZONTALS)
    for threshold, metric in zip(thresholds, metrics):
        table.add_row(
            f"{threshold:.4f} {metric.tp} {metric.fp} {metric.fn} {metric.precision:.4f} {metric.recall:.4f} {metric.F1:.4f}"
        )
    console = Console()
    console.print(table)

    with open(os.path.join(results_dir, "metrics.txt"), "wt") as logfile:
        console = Console(file=logfile, width=100, force_jupyter=False)
        console.print(table)
