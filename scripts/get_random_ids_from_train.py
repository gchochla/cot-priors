import warnings

import gridparse
import pandas as pd
from legm import splitify_namespace
from legm.argparse_utils import add_arguments

from cot_priors import (
    PromptTextDataset,
    SemEval2018Task1EcDataset,
    MFRCDataset,
    GoEmotionsDataset,
    CONSTANT_ARGS,
)

DATASET = dict(
    SemEval=SemEval2018Task1EcDataset,
    GoEmotions=GoEmotionsDataset,
    MFRC=MFRCDataset,
)


def parse_args():
    parser = gridparse.GridArgumentParser()
    sp = parser.add_subparsers(dest="task")

    for task in DATASET:
        sp_task = sp.add_parser(task)

        argparse_args = {}
        for module in [DATASET[task], PromptTextDataset]:
            argparse_args.update(module.argparse_args())

        add_arguments(sp_task, argparse_args, replace_underscores=True)
        add_arguments(
            sp_task,
            CONSTANT_ARGS,
            replace_underscores=True,
        )
        sp_task.add_argument(
            "--output-filename",
            type=str,
            required=True,
            help="Path to the directory where the output files will be saved.",
        )

    return parser.parse_args()


def main():
    warnings.warn(
        "This script only works for the train split and varies the seed and the shot."
    )
    grid_args = parse_args()

    train_ids = set()
    train_dataset = DATASET[grid_args[0].task](
        splits="train",
        init__namespace=splitify_namespace(grid_args[0], "train"),
    )
    dataset = PromptTextDataset(
        train_dataset=train_dataset,
        test_dataset=train_dataset,
        init__namespace=grid_args[0],
    )

    for args in grid_args:
        dataset.shot = args.shot
        dataset._example_sampler_mixin_data["seed"] = args.seed
        [None for _ in dataset]  # to sample the examples
        ids = [
            (
                _id.split(train_dataset.id_separator)[0]
                if train_dataset.annotation_mode == "aggregate"
                else _id
            )
            for _ids in dataset.ids_per_query.values()
            for _id in _ids
        ]
        train_ids.update(ids)

    full_examples = []
    for _id in sorted(train_ids):
        original_id, annotator = _id.split(train_dataset.id_separator)
        example = train_dataset.examples[original_id]
        label = train_dataset.annotations[annotator][original_id]
        full_examples.append(
            {
                "id": _id,
                "text": example["original_text"],
                "label": train_dataset.index_label_set(label),
            }
        )

    with open(grid_args[0].output_filename, "w") as fp:
        fp.write("\n".join(train_ids) + "\n")

    pd.DataFrame(full_examples).to_csv(
        grid_args[0].output_filename.replace(".txt", ".csv"), index=False
    )


if __name__ == "__main__":
    main()
