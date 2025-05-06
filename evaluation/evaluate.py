import datetime
import os
import sys
import torch
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from evaluation.eval_sequence import evaluate_sequence_reg
from rich import box
from rich.console import Console
from rich.table import Table


def create_results_dir(cfg, dataset):
    def get_timestamp() -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    results_dir = os.path.join(
        cfg.results_dir, f"{dataset.sequence_id}_results", get_timestamp()
    )
    latest_dir = os.path.join(
        cfg.results_dir, f"{dataset.sequence_id}_results", "latest"
    )
    os.makedirs(results_dir, exist_ok=True)
    (
        os.unlink(latest_dir)
        if os.path.exists(latest_dir) or os.path.islink(latest_dir)
        else None
    )
    os.symlink(results_dir, latest_dir)

    return results_dir


def save_pickle(data_variable, file_name):
    dbfile2 = open(file_name, "ab")
    pickle.dump(data_variable, dbfile2)
    dbfile2.close()


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
    from config.eval_config import get_config_eval
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

    dataset = dataset_factory(dataloader=cfg.dataloader, data_dir=cfg.data_dir, sequence=cfg.sequence)

    descriptors, closures_list, distances_list = evaluate_sequence_reg(
        model, dataset, cfg
    )
    results_dir = create_results_dir(cfg, dataset)
    save_pickle(descriptors, os.path.join(results_dir, "descriptors.pkl"))

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
