import os
import langcodes
import json
from typing import Any

import torch
import yaml
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer

from cot_priors.base_datasets import TextDataset


class GoEmotions(TextDataset):
    """Plain text dataset for `GoEmotions`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance.

    Attributes:
        Check `TextDataset` for attributes.
    """

    multilabel = True
    annotator_labels = True
    name = "GoEmotions"
    source_domain = "Reddit"

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = TextDataset.argparse_args() | dict(
            emotion_clustering_json=dict(
                type=str,
                help="JSON file with clustering of emotions",
            )
        )
        return args

    def __init__(self, emotion_clustering_json, *args, **kwargs):
        """Initializes dataset.

        Args:
            emotion_clustering_json: JSON file with clustering of emotions.
            Check `TextDataset` for other arguments.
        """
        self.emotion_clustering_json = emotion_clustering_json
        super().__init__(*args, **kwargs)

    def _multilabel_one_hot(
        self, labels: "np.ndarray", n_classes: int = 27
    ) -> torch.Tensor:
        """GoEmotions-specific label transformer to multilable one-hot,
        neutral emotion is discarded (represented as 0s)."""

        labels = [
            list(filter(lambda x: x < n_classes, map(int, lbl.split(","))))
            for lbl in labels
        ]
        new_labels = [
            torch.nn.functional.one_hot(
                torch.tensor(lbl, dtype=int), n_classes
            ).sum(0)
            for lbl in labels
        ]
        return torch.stack(new_labels)

    def _subset_emotions(
        self,
        annotations: dict[Any, dict[str, str | torch.Tensor]],
        emotions: list[str],
    ) -> list[str]:
        """Transforms emotions to a subset of emotions based on clustering
        in `emotion_clustering_json`. Each new label is union of old labels."""

        if not self.emotion_clustering_json:
            return emotions

        with open(self.emotion_clustering_json) as fp:
            clustering = json.load(fp)

        new_emotions = list(clustering)

        for annotation in annotations.values():
            for worker_id, label in annotation["label"].items():
                new_label = torch.zeros(len(new_emotions))

                for i, emotion in enumerate(new_emotions):
                    for old_emotion in clustering[emotion]:
                        new_label[i] += label[emotions.index(old_emotion)]

                annotation["label"][worker_id] = new_label.clamp(0, 1)

        return new_emotions

    def _load_data(
        self, split: str
    ) -> tuple[dict[Any, dict[str, str | torch.Tensor]], list[str]]:
        ## read emotions from file
        emotion_fn = os.path.join(self.root_dir, "emotions.txt")
        emotions = pd.read_csv(emotion_fn, header=None)[0].values.tolist()[
            :-1
        ]  # gets rid of neutral emotion

        ## read aggregated labels from file
        filename = os.path.join(self.root_dir, f"{split}.tsv")
        df = pd.read_csv(filename, sep="\t", header=None)

        ids = df.iloc[:, -1].values.tolist()
        aggr_labels = {
            _id: y
            for _id, y in zip(
                ids,
                self._multilabel_one_hot(
                    df.iloc[:, 1].values, len(emotions)
                ).float(),
            )
        }

        if self.annotation_mode == "aggregate":
            annotations = {
                _id: dict(
                    text=self.preprocessor(text),
                    original_text=text,
                    label={"aggregate": aggr_labels[_id]},
                )
                for _id, text in zip(ids, df.iloc[:, 0].values)
            }
            self.annotators = set()

        else:
            ## read annotator labels from file
            filenames = [
                os.path.join(self.root_dir, f"goemotions_{i}.csv")
                for i in range(1, 4)
            ]
            df = pd.concat([pd.read_csv(fn) for fn in filenames])
            df = df[df["id"].isin(set(ids))]
            df["labels"] = [
                [row[lbl] for lbl in emotions] for _, row in df.iterrows()
            ]

            groupby = df[["text", "rater_id", "id", "labels"]].groupby("id")
            annotations = groupby.agg(
                {
                    "text": lambda x: x.iloc[0],
                    "rater_id": lambda x: x.tolist(),
                    "labels": lambda x: x.tolist(),
                }
            )

            annotations = {
                _id: dict(
                    text=self.preprocessor(text),
                    original_text=text,
                    label={
                        worker_id: torch.tensor(labels).float()
                        for worker_id, labels in zip(rater_ids, label_list)
                    }
                    | {"aggregate": aggr_labels[_id]},
                )
                for _id, text, rater_ids, label_list in annotations.itertuples()
            }

            self.annotators = set(df["rater_id"].unique())

        emotions = self._subset_emotions(annotations, emotions)

        return annotations, emotions


class MFRC(TextDataset):
    """Plain text dataset for `MFRC`. Class doesn't
    use from_namespace decorator, so it can be used for inheritance.

    Attributes:
        Check `TextDataset` for attributes.
    """

    multilabel = True
    annotator_labels = True
    name = "MFRC"
    source_domain = "Reddit"

    def _load_data(self, split: str) -> tuple[
        dict[Any, dict[str, str | torch.Tensor | dict[str, torch.Tensor]]],
        list[str],
    ]:
        # { id1: { "text": "lorem ipsum", "label": {
        #   "ann1": torch.tensor([4]), "ann2": torch.tensor([3, 4]), "aggregate": torch.tensor([4]),
        # }, ... }, ... }

        # only train available in the dataset, contains entire dataset
        dataset = load_dataset("USC-MOLA-Lab/MFRC", split="train")

        with open(os.path.join(self.root_dir, "splits.yaml"), "r") as fp:
            text2id = yaml.safe_load(fp)[split]

        label_set = set()
        annotations = {}
        for e in dataset:
            id = text2id.get(e["text"], None)
            if id is None:
                # from another split
                continue

            labels = e["annotation"].split(",")
            if len(labels) > 1 and (
                "Non-Moral" in labels or "Thin Morality" in labels
            ):
                # https://arxiv.org/pdf/2208.05545v2 Appendix A.2.1:
                # Thin Morality only if no other label,
                # Non-Moral if no other label and not Thin Morality
                # so it cannot be that either is present and more than one modality
                continue
            elif labels[0] == "Non-Moral" or labels[0] == "Thin Morality":
                labels = []

            label_set.update(labels)

            if id not in annotations:
                annotations[id] = {
                    "text": self.preprocessor(e["text"]),
                    "original_text": e["text"],
                    "label": {e["annotator"]: labels},
                }
            else:
                annotations[id]["label"][e["annotator"]] = labels

        label_set = sorted(label_set)
        mlb = MultiLabelBinarizer().fit([label_set])
        for id in annotations:
            for annotator, label in annotations[id]["label"].items():
                annotations[id]["label"][annotator] = torch.tensor(
                    mlb.transform([label])[0]
                ).float()

            annotations[id]["label"]["aggregate"] = (
                (
                    sum(annotations[id]["label"].values())
                    / len(annotations[id]["label"])
                )
                >= 0.5
            ).float()

        return annotations, label_set
