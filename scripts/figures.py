import os
import sys
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["jaccard_score", "micro_f1", "macro_f1"]
MODELS = [
    "meta-llama--Llama-2-7b-chat-hf",
    "meta-llama--Llama-2-70b-chat-hf",
    "meta-llama--Meta-Llama-3-8B-Instruct",
    "meta-llama--Meta-Llama-3-70B-Instruct",
    "gpt-3.5-turbo",
    "gpt-4o-mini-2024-07-18",
]
DATASET_NAMES = ["MFRC", "GoEmotions"]


def cot_vs_icl(analyses_folder, log_folder, basename=None):

    shots = {"ICL": [5, 25, 45], "CoT": [5, 15]}

    results = {}

    height_range = [
        [float("inf"), -float("inf")] for _ in range(len(DATASET_NAMES))
    ]

    for i, dataset in enumerate(DATASET_NAMES):
        results[dataset] = {}
        for model in MODELS:
            results[dataset][model] = {}
            for method in shots:
                results[dataset][model][method] = {}

                if method == "ICL":
                    template = "{model}-annotator-True-None-{s}-shot_0/aggregated_metrics.yml"
                else:
                    # TODO: read _1 too because of different chains
                    template = "{model}-cot-annotator-True-None-{s}-shot_0/aggregated_metrics.yml"

                for s in shots[method]:
                    try:
                        with open(
                            os.path.join(
                                log_folder,
                                (
                                    dataset
                                    if "gpt" not in model.lower()
                                    else dataset + "OpenAI"
                                ),
                                template.format(model=model, s=s),
                            )
                        ) as fp:
                            values = yaml.safe_load(fp)[""]

                        for m in METRICS:
                            mean, std = values[f"test_{m.lower()}"].split("+-")
                            results[dataset][model][method].setdefault(
                                m, []
                            ).append((float(mean), float(std)))

                            height_range[i][0] = min(
                                height_range[i][0],
                                float(mean) - float(std),
                            )
                            height_range[i][1] = max(
                                height_range[i][1],
                                float(mean) + float(std),
                            )

                    except FileNotFoundError:
                        for m in METRICS:
                            results[dataset][model][method].setdefault(
                                m, []
                            ).append((None, None))

    fig, ax = plt.subplots(
        ncols=len(MODELS),
        nrows=len(DATASET_NAMES),
        figsize=(2.5 * len(MODELS), 2 * len(DATASET_NAMES)),
        sharex=True,
    )

    colors = [
        "skyblue",
        "limegreen",
        "orange",
        "violet",
        "red",
        "peru",
        "dimgray",
    ]

    # results[dataset][model][method][metric]
    for i, dataset in enumerate(DATASET_NAMES):
        for j, model in enumerate(MODELS):
            for k, metric in enumerate(METRICS):
                ax[i, j].plot(
                    shots["ICL"],
                    [
                        results[dataset][model]["ICL"][metric][i][0]
                        for i in range(len(shots["ICL"]))
                    ],
                    color=colors[k],
                    marker=".",
                    linestyle=":",
                )
                ax[i, j].errorbar(
                    shots["ICL"],
                    [
                        results[dataset][model]["ICL"][metric][i][0] or 0
                        for i in range(len(shots["ICL"]))
                    ],
                    yerr=[
                        results[dataset][model]["ICL"][metric][i][1] or 0
                        for i in range(len(shots["ICL"]))
                    ],
                    color=colors[k],
                    fmt=".",
                )

                ax[i, j].plot(
                    shots["CoT"],
                    [
                        results[dataset][model]["CoT"][metric][i][0]
                        for i in range(len(shots["CoT"]))
                    ],
                    color=colors[k],
                    marker="x",
                    linestyle="--",
                )
                ax[i, j].errorbar(
                    shots["CoT"],
                    [
                        results[dataset][model]["CoT"][metric][i][0] or 0
                        for i in range(len(shots["CoT"]))
                    ],
                    yerr=[
                        results[dataset][model]["CoT"][metric][i][1] or 0
                        for i in range(len(shots["CoT"]))
                    ],
                    color=colors[k],
                    fmt="x",
                )

    for i, dataset in enumerate(DATASET_NAMES):
        ax[i, 0].set_ylabel(dataset)
        for j, model in enumerate(MODELS):
            ax[i, j].grid(axis="y", linestyle="dashed")
            ax[i, j].set_ylim(
                top=height_range[i][1] * 1.03,
                bottom=height_range[i][0] * 0.97,
            )

    for k, metric in enumerate(METRICS):
        ax[0, 0].plot(
            [],
            [],
            "-.",
            color=colors[k],
            label=metric.replace("_", " ").title(),
        )
    ax[0, 0].plot([], [], ".", color="black", label="ICL")
    ax[0, 0].plot([], [], "x", color="black", label="CoT")

    for j, model in enumerate(MODELS):
        name = model.split("--")[1] if "--" in model else model
        if "gpt-4o-mini" in name:
            name = "gpt-4o-mini"
        ax[0, j].set_title(name)

    fig.suptitle("Comparison of ICL and CoT", fontsize=16)
    fig.supxlabel("Shots")
    plt.subplots_adjust(top=0.7)
    fig.legend(
        fontsize=10,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(1, -0.08),
    )
    fig.tight_layout()
    plt.savefig(
        os.path.join(
            analyses_folder,
            basename or f"cot_vs_icl.pdf",
        ),
        bbox_inches="tight",
    )


def cot_priors(analyses_folder, log_folder, basename=None):

    height_range = [
        [float("inf"), -float("inf")] for _ in range(len(DATASET_NAMES))
    ]

    labels = [
        "Ground Truth",
        "ICL Prior",
        "CoT Random Labels",
        "CoT Random Reasoning",
        "CoT Random Labels + Reasoning",
        "Different Valid Reasoning",
    ]
    metric_names = ["JS", "Mic", "Mac"]

    results = {
        dataset: {model: {metric: [] for metric in METRICS} for model in MODELS}
        for dataset in DATASET_NAMES
    }

    for i, dataset in enumerate(DATASET_NAMES):
        results[dataset] = {}
        for model in MODELS:
            results[dataset][model] = {}
            for metric in METRICS:

                try:
                    df = pd.read_csv(
                        os.path.join(
                            log_folder,
                            f"{dataset.lower()}-{model}",
                            f"{metric}.csv",
                        ),
                        index_col=0,
                    )
                except FileNotFoundError:
                    results[dataset][model][metric] = dict(
                        gt=(None, None),
                        icl=(None, None),
                        cot_false_distr=(None, None),
                        cot_true_none=(None, None),
                        cot_true_distr=(None, None),
                        diff_reasoning=(None, None),
                    )
                    continue

                def parse_mean_std(df, comp):
                    mean = df["CoT"][comp].split("\n±")[0]
                    std = df["CoT"][comp].split("\n±")[1]
                    return float(mean), float(std)

                results[dataset][model][metric] = dict(
                    gt=parse_mean_std(df, "Ground Truth"),
                    icl=parse_mean_std(df, "ICL-prior"),
                    cot_false_distr=parse_mean_std(
                        df, "CoT-False-Distribution"
                    ),
                    cot_true_none=parse_mean_std(df, "CoT-True-None"),
                    cot_true_distr=parse_mean_std(df, "CoT-True-Distribution"),
                    diff_chain=parse_mean_std(df, "Diff-Chain"),
                )

                for quant in results[dataset][model][metric]:
                    height_range[i][0] = min(
                        height_range[i][0],
                        results[dataset][model][metric][quant][0],
                    )
                    height_range[i][1] = max(
                        height_range[i][1],
                        results[dataset][model][metric][quant][0],
                    )

    n_results = len(results[dataset][model][metric])

    # bar plot comparing the strength of the prior to the ground truth
    # compare for each dataset, all the models for every metric at different shots
    fig, ax = plt.subplots(
        len(DATASET_NAMES),
        len(MODELS),
        figsize=(17, 5),
        sharey=True,
        sharex=True,
    )

    barwidth = 0.09
    colors = [
        "skyblue",
        "limegreen",
        "orange",
        "violet",
        "red",
        "peru",
        "dimgray",
    ]

    for j, dataset in enumerate(DATASET_NAMES):
        for k, model in enumerate(MODELS):
            for i, metric in enumerate(METRICS):
                # nested_x_value is y_yhat and yhat_yprior
                # x_value is SHOT

                for l, quant in enumerate(results[dataset][model][metric]):

                    if i == j == k == 0:
                        ax[j, k].bar(
                            [i * barwidth * (n_results + 1) + l * barwidth],
                            results[dataset][model][metric][quant][0] or 0,
                            barwidth,
                            yerr=results[dataset][model][metric][quant][1] or 0,
                            label=labels[l],
                            color=colors[l],
                            edgecolor="black",
                        )
                    else:
                        ax[j, k].bar(
                            [i * barwidth * (n_results + 1) + l * barwidth],
                            results[dataset][model][metric][quant][0] or 0,
                            barwidth,
                            yerr=results[dataset][model][metric][quant][1] or 0,
                            color=colors[l],
                            edgecolor="black",
                        )

                    if l == n_results - 1 and j == len(DATASET_NAMES) - 1:
                        ax[j, k].text(
                            # move as bars as there are nested x values + gap between nested x values
                            i * barwidth * (n_results + 1)
                            # move to the middle of current bars
                            # (we dont consider gap part of bars for text purposes)
                            + n_results * barwidth / 2
                            # adjust because text is not a point (assume about one bar)
                            # + starting point is after first bar
                            - 0.8 * barwidth,  #  - offset_y_axis_labels[i],
                            -0.205
                            * (height_range[j][1] - min(height_range[j][0], 0)),
                            metric_names[i],
                            size=10,
                            fontweight="bold",
                            rotation=60,
                            ha="left",
                            rotation_mode="anchor",
                        )
                if j == 0:
                    model_name = (
                        model.split("--")[1] if "--" in model else model
                    )
                    if "gpt-4o-mini" in model_name:
                        model_name = "gpt-4o-mini"
                    ax[j][k].set_title(model_name)

                if k == 0:
                    ax[j, k].set_ylabel(DATASET_NAMES[j], fontsize=14)

                ax[j, k].grid(axis="y", linestyle="dashed")

                ax[j, k].set_xticklabels([])
                ax[j, k].set_xticks([])
                # ax[j][k].set_xticks(
                #     # move to middle of whole current section of nested_x_values + gap for each
                #     # y_axis value, adjust one bar back because starting after the first bar
                #     barwidth * (len(METRICS) * (n_results + 1) / 2 - 1)
                #     # - offset_x_axis_labels
                # )
                for label in ax[j, k].get_xticklabels():
                    label.set_y(
                        label.get_position()[1] - 0.08
                    )  # Adjust the offset value as needed

                ax[j, k].set_ylim(
                    top=min(np.nan_to_num(height_range[j][1]) * 1.1, 1.1),
                    bottom=0,
                )

    fig.suptitle("")
    # fig.legend(
    #     loc="center left",
    #     fontsize=12,
    #     bbox_to_anchor=(-0.01, 0.52),
    #     # ncol=2,
    #     frameon=False,
    # )

    handles, labels = ax[0, 0].get_legend_handles_labels()
    order = [0, 1, 2, 3, 4, 5]
    plt.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc="upper center",
        fontsize=12,
        bbox_to_anchor=(-1.95, 2.74),
        ncol=(n_results + 1) // 2,
        frameon=False,
    )

    # set x label for entire figure
    fig.text(
        0.27,
        0.985,
        "Similarity of CoT to",
        ha="center",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            analyses_folder,
            basename or f"cot-priors.pdf",
        ),
        bbox_inches="tight",
    )
    plt.clf()


def cot_consistency(analyses_folder, log_folder, basename=None):
    """Plots how consistent the CoT is with different reasoning chains."""


if __name__ == "__main__":
    script = sys.argv[1]
    globals()[script](*sys.argv[2:])
