import os
import yaml
from itertools import combinations

import gridparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from legm.argparse_utils import add_arguments
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
)

from cot_priors import SemEval2018Task1EcDataset, GoEmotionsDataset, MFRCDataset

DATASET = dict(
    SemEval=SemEval2018Task1EcDataset,
    GoEmotions=GoEmotionsDataset,
    MFRC=MFRCDataset,
)


def parse_args():
    parser = gridparse.ArgumentParser()

    sp = parser.add_subparsers(dest="task")

    for task in DATASET:
        sp_task = sp.add_parser(task)

        add_arguments(
            sp_task, DATASET[task].argparse_args(), replace_underscores=True
        )
        sp_task.add_argument(
            "--experiments",
            type=str,
            nargs="+",
            required=True,
            help="experiments whose distributions to compare",
        )
        sp_task.add_argument(
            "--output-dir",
            type=str,
            default=".",
            help="directory to save plots",
        )
        sp_task.add_argument(
            "--alternative-experiment-names",
            type=str,
            nargs="+",
            help="rename experiments in plots",
        )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    elif os.listdir(args.output_dir):
        raise ValueError(
            f"Output directory {args.output_dir} already exists. "
            "Please delete it or specify a different directory."
        )

    # dev is probably smallest split to load
    dataset = DATASET[args.task](init__namespace=args, splits="dev")

    ### LOAD DATA

    labels = {}
    data = {}
    params = {}

    additional_experiments = {}

    for i, exp in enumerate(args.experiments):
        with open(os.path.join(exp, "indexed_metrics.yml")) as f:
            data[exp] = yaml.safe_load(f)

        with open(os.path.join(exp, "params.yml")) as f:
            exp_params = yaml.safe_load(f)
            if "experiment_0" in exp_params:
                exp_params = exp_params["experiment_0"]
            params[exp] = exp_params

        for exp_no, exp_data in data[exp].items():
            description = exp_data.pop("description")
            if description:
                description = "".join(
                    [d[0] for d in description.split("-") if d != "shot"]
                )
                new_exp = exp + "-" + description
                additional_experiments[new_exp] = exp
                params[new_exp] = params[exp]
                print(f"Renaming {exp} to {new_exp}")
            else:
                new_exp = exp

            for example in exp_data:
                gt_key = (
                    "test_gt" if "test_gt" in exp_data[example] else "test_true"
                )
                pred_key = (
                    "test_preds"
                    if "test_preds" in exp_data[example]
                    else "test_pred"
                )
                if (
                    i == 0
                ):  # to avoid contamination of examples from traindev priors
                    labels.setdefault("gt", {}).setdefault("experiment_0", {})[
                        example
                    ] = dataset.get_label_from_str(exp_data[example][gt_key])
                labels.setdefault(new_exp, {}).setdefault(exp_no, {})[
                    example
                ] = dataset.get_label_from_str(exp_data[example][pred_key])

    experiments = args.experiments + list(additional_experiments)

    label_lists = {}
    examples = list(labels["gt"][next(iter(labels["gt"]))])

    for exp in labels:
        for exp_no in labels[exp]:
            for example in examples:
                label_lists.setdefault(exp, {}).setdefault(exp_no, []).append(
                    labels[exp][exp_no][example]
                )
            label_lists[exp][exp_no] = np.array(label_lists[exp][exp_no])

    ### COMPUTE METRICS ACROSS AND WITHIN DISTRIBUTIONS

    metric_compute = {
        "macro f1": lambda x, y: f1_score(
            x, y, average="macro", zero_division=0
        ),
        "micro f1": lambda x, y: f1_score(
            x, y, average="micro", zero_division=0
        ),
        "jaccard score": lambda x, y: jaccard_score(
            x, y, average="samples", zero_division=1
        ),
        "macro precision": lambda x, y: precision_score(
            x, y, average="macro", zero_division=0
        ),
        "micro precision": lambda x, y: precision_score(
            x, y, average="micro", zero_division=0
        ),
        "macro recall": lambda x, y: recall_score(
            x, y, average="macro", zero_division=0
        ),
        "micro recall": lambda x, y: recall_score(
            x, y, average="micro", zero_division=0
        ),
    }

    pairwise_metrics = {m: {} for m in metric_compute}
    individual_metrics = {m: {} for m in metric_compute}

    for exp in experiments:
        for exp_no1, exp_no2 in combinations(label_lists[exp], 2):
            for metric in metric_compute:
                individual_metrics[metric].setdefault(exp, []).append(
                    metric_compute[metric](
                        label_lists[exp][exp_no1], label_lists[exp][exp_no2]
                    )
                )

    for exp1, exp2 in reversed(list(combinations(labels, 2))):
        for exp1_no in label_lists[exp1]:
            for exp2_no in label_lists[exp2]:
                for metric in metric_compute:
                    pairwise_metrics[metric].setdefault(exp1, {}).setdefault(
                        exp2, []
                    ).append(
                        metric_compute[metric](
                            label_lists[exp1][exp1_no],
                            label_lists[exp2][exp2_no],
                        )
                    )

    pairwise_metrics_std = {metric: {} for metric in pairwise_metrics}
    individual_metrics_std = {metric: {} for metric in individual_metrics}

    for metric in pairwise_metrics:
        for exp in labels:
            if exp in individual_metrics[metric]:
                individual_metrics_std[metric][exp] = np.std(
                    individual_metrics[metric][exp]
                )
                individual_metrics[metric][exp] = np.mean(
                    individual_metrics[metric][exp]
                )
            else:
                individual_metrics[metric][exp] = np.nan
                individual_metrics_std[metric][exp] = np.nan

        exp1s = list(pairwise_metrics[metric])
        exp2s = {exp1: list(pairwise_metrics[metric][exp1]) for exp1 in exp1s}
        for exp1 in exp1s:
            for exp2 in exp2s[exp1]:
                pairwise_metrics_std[metric].setdefault(exp1, {})[exp2] = (
                    f"{np.mean(pairwise_metrics[metric][exp1][exp2]):.3f}"
                    f"\n±{np.std(pairwise_metrics[metric][exp1][exp2]):.3f}"
                )
                pairwise_metrics[metric][exp1][exp2] = np.mean(
                    pairwise_metrics[metric][exp1][exp2]
                )

                pairwise_metrics_std[metric].setdefault(exp2, {})[exp1] = (
                    pairwise_metrics_std[metric][exp1][exp2]
                )
                pairwise_metrics[metric].setdefault(exp2, {})[exp1] = (
                    pairwise_metrics[metric][exp1][exp2]
                )

        for exp in labels:

            pairwise_metrics_std[metric].setdefault(exp, {})[exp] = (
                f"{individual_metrics[metric][exp]:.3f}"
                f"\n±{individual_metrics_std[metric][exp]:.3f}"
            )

            pairwise_metrics[metric].setdefault(exp, {})[exp] = (
                individual_metrics[metric][exp]
            )

    experiment_order = list(reversed(labels.keys()))

    # reorder for plotting
    pairwise_metrics = {
        metric: {
            exp1: {
                exp2: pairwise_metrics[metric][exp1][exp2]
                for exp2 in experiment_order
                if exp2 in pairwise_metrics[metric][exp1]
            }
            for exp1 in experiment_order
        }
        for metric in pairwise_metrics
    }

    pairwise_metrics_std = {
        metric: {
            exp1: {
                exp2: pairwise_metrics_std[metric][exp1][exp2]
                for exp2 in experiment_order
                if exp2 in pairwise_metrics_std[metric][exp1]
            }
            for exp1 in experiment_order
        }
        for metric in pairwise_metrics_std
    }

    individual_metrics = {
        metric: {
            exp: individual_metrics[metric][exp]
            for exp in experiment_order
            if exp in individual_metrics[metric]
        }
        for metric in individual_metrics
    }

    individual_metrics_std = {
        metric: {
            exp: individual_metrics_std[metric][exp]
            for exp in experiment_order
            if exp in individual_metrics_std[metric]
        }
        for metric in individual_metrics_std
    }

    # plot metrics
    sns.set_theme(style="whitegrid", font_scale=0.8)

    if args.alternative_experiment_names:
        name_mapping = {
            exp: name
            for exp, name in zip(
                args.experiments, args.alternative_experiment_names
            )
        }
        for add_exp in additional_experiments:
            name_mapping[add_exp] = (
                name_mapping[additional_experiments[add_exp]]
                + "-"
                + add_exp[len(additional_experiments[add_exp]) + 1 :]
            )

        name_mapping["gt"] = "Ground Truth"
    else:
        name_mapping = {}
        for exp in labels:

            if exp in additional_experiments:
                description = exp[len(additional_experiments[exp]) + 1 :]
            else:
                description = ""

            if exp == "gt":
                name = "Ground Truth"
            elif params[exp].get("train_pred_log_dir", None) is not None:
                with open(
                    os.path.join(
                        params[exp]["train_pred_log_dir"], "params.yml"
                    )
                ) as f:
                    train_pred_log_dir_params = yaml.safe_load(f)

                label_mode = train_pred_log_dir_params.get(
                    "label_randomization", ""
                ) or train_pred_log_dir_params.get("label_mode", "")
                label_mode = label_mode + "-" if label_mode else ""

                name = "Priors-prompt-" + label_mode + str(params[exp]["shot"])
            elif (
                params.get(exp, {}).get("label_randomization", None)
                or params.get(exp, {}).get("label_mode", None)
            ) not in (None, "none"):
                name = "Prior-" + (
                    params[exp].get("label_randomization", "")
                    or params[exp].get("label_mode", "")
                )
            else:
                try:
                    name = str(params[exp]["shot"]) + "-shot"
                except:
                    name = exp

            if description:
                name += "-" + description
            name_mapping[exp] = name

    for metric in pairwise_metrics:
        fig, ax = plt.subplots(figsize=(len(experiments), len(experiments)))
        df = pd.DataFrame(pairwise_metrics[metric])
        df.rename(index=name_mapping, columns=name_mapping, inplace=True)

        df_values = pd.DataFrame(pairwise_metrics_std[metric])
        df_values.rename(index=name_mapping, columns=name_mapping, inplace=True)

        sns.heatmap(df, annot=df_values, cmap="Blues", ax=ax, fmt="s")
        plt.xticks(rotation=80)
        ax.set_title(metric.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, f"{metric.replace(' ', '_')}.pdf"),
            dpi=300,
        )

        df_values.to_csv(
            os.path.join(args.output_dir, f"{metric.replace(' ', '_')}.csv")
        )

        index = individual_metrics[metric].keys()
        individual_df = pd.DataFrame(
            {
                "metric_mean": individual_metrics[metric].values(),
                "metric_std": [
                    individual_metrics_std[metric][exp] for exp in index
                ],
            },
            index=[name_mapping[e] for e in index],
        )

        individual_df.to_csv(
            os.path.join(
                args.output_dir, f"{metric.replace(' ', '_')}_individual.csv"
            )
        )

        ax = individual_df.plot.barh(
            y="metric_mean",
            xerr="metric_std",
            # rot=80,
            title="Self " + metric.replace('_', ' ').title() + " across runs",
            legend=False,
            xlim=(
                max(np.nanmin(individual_df.metric_mean.values) - 0.05, 0),
                np.nanmax(individual_df.metric_mean.values) + 0.05,
            ),
            figsize=(len(individual_df) / 4, len(individual_df) / 4),
        )

        fig = ax.get_figure()
        plt.tight_layout()
        fig.savefig(
            os.path.join(
                args.output_dir, f"{metric.replace(' ', '_')}_individual.pdf"
            ),
            dpi=300,
        )

        with open(os.path.join(args.output_dir, "name_mapping.yml"), "w") as fp:
            yaml.dump(name_mapping, fp)


if __name__ == "__main__":
    main()
