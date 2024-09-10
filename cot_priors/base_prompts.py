import json
import warnings
import random
from string import Template
from typing import Any
from copy import deepcopy

import torch
import langcodes
import pandas as pd
import gensim.downloader as api
from ember.dataset import BaseDataset
from sentence_transformers import SentenceTransformer

from cot_priors.base_datasets import TextDataset


class ExampleSamplerMixin:
    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return dict(
            retries=dict(
                type=int,
                default=3,
                help="number of retries for random conditional sampling",
            ),
            sentence_model_name_or_path=dict(
                type=str,
                # default="all-mpnet-base-v2",
                help="sentence embedding model name or path",
            ),
            sampling_strategy=dict(
                type=str,
                choices=["complete", "similarity"],
                help="how to sample examples",
                searchable=True,
                metadata=dict(name=True, name_priority=1),
            ),
            label_mode=dict(
                type=str,
                help="how to handle labels",
                choices=[
                    "random",
                    "distribution",
                    "preds",
                ],
                searchable=True,
                metadata=dict(name=True, name_priority=1),
            ),
            query_label_mode=dict(
                type=str,
                help="how to handle query labels in prompt (ablation)",
                choices=["random", "distribution", "preds"],
                searchable=True,
                metadata=dict(name=True, name_priority=1),
            ),
            cot_randomize=dict(
                type=bool,
                help="whether to sample chains of thought randomly",
                searchable=True,
            ),
            sentence_device=dict(
                type=str,
                default="args.device",
                help="device to use for sentence embeddings",
                metadata=dict(disable_comparison=True),
            ),
            label_randomization_seed=dict(
                type=int,
                searchable=True,
                help="random seed for label randomization",
                metadata=dict(disable_comparison=True),
            ),
        )

    def __init__(
        self,
        sentence_model_name_or_path: str | None = None,
        seed: int | None = None,
        label_randomization_seed: int | None = None,
        retries: int = 3,
        sentence_device: str = "cpu",
        sampling_strategy: str | None = None,
        label_mode: str | None = None,
        query_label_mode: str | None = None,
        cot_randomize: bool = False,
        *args,
        **kwargs,
    ):
        assert retries > 0, "retries must be positive"
        assert (
            sampling_strategy != "similarity"
            or sentence_model_name_or_path is not None
        ), "sentence_model_name_or_path must be provided for similarity sampling"
        super().__init__(*args, **kwargs)
        self.sampling_strategy = sampling_strategy or "random"
        self._example_sampler_mixin_data = dict(
            seed=seed,
            retries=retries,
            label_randomization_seed=label_randomization_seed,
        )

        self._example_sampler_mixin_data["label_mode"] = (
            label_mode or "none"
        ).lower()
        self._example_sampler_mixin_data["query_label_mode"] = (
            query_label_mode or "none"
        ).lower()
        self._example_sampler_mixin_data["cot_randomize"] = cot_randomize

        if self.sampling_strategy == "similarity":
            self._example_sampler_mixin_data["sentence_embedding_model"] = (
                SentenceTransformer(
                    sentence_model_name_or_path, device=sentence_device
                )
            )

    def _handle_labels(
        self,
        samples: list[dict[str, str | torch.Tensor]],
        dataset: TextDataset,
        label_mode: str = "none",
    ):
        """Handles label transformation of `samples` in-place
        according to `label_mode`.

        The basis for randomization is the paper
        `Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?`
        """

        label_randomization_seed = self._example_sampler_mixin_data[
            "label_randomization_seed"
        ]

        if (
            label_randomization_seed != self._example_sampler_mixin_data["seed"]
            and label_randomization_seed is not None
        ):
            # setting the same seed can result in the same samples twice
            # for distribution sampling
            # also don't set when None to avoid same seeds when user doesn't specify this one
            random.seed(
                self._example_sampler_mixin_data["label_randomization_seed"]
            )

        for sample in samples:

            if label_mode == "random":
                if dataset.multilabel:
                    new_labels = random.sample(
                        dataset.label_set,
                        random.randint(0, len(dataset.label_set)),
                    )
                else:
                    new_labels = random.choice(dataset.label_set)

                new_labels = dataset.get_label_from_str(new_labels)

            elif label_mode == "distribution":
                new_labels = random.choice(dataset)["label"]

            elif label_mode == "preds":
                if "pred_label" not in sample:
                    raise ValueError(
                        "label_mode is 'preds' but 'pred_label'"
                        f" not in sample {sample['id']}"
                    )
                new_labels = sample["pred_label"]

            elif label_mode == "contrastive":
                raise NotImplementedError(
                    "contrastive label mode not implemented"
                )

            else:
                new_labels = sample["label"]

            sample["label"] = new_labels

    def handle_query_label(
        self,
        query: dict[str, str | torch.Tensor],
        is_demo: bool = False,
        dataset: TextDataset = None,
    ) -> str:
        """Returns the label of `query` based on label mode.

        Args:
            query: query dictionary.
            is_demo: whether the query is a demonstration as well (ablation).
            dataset: dataset to use for label handling, e.g. sampling from.
        """

        if not is_demo:
            if (
                self._example_sampler_mixin_data["label_mode"] == "preds"
                and query["pred_label"] is not None
            ):
                return query["pred_label"]
            return query["label"]
        else:
            if (
                self._example_sampler_mixin_data["query_label_mode"] == "preds"
                and query["pred_label"] is not None
            ):
                return query["pred_label"]
            elif self._example_sampler_mixin_data["query_label_mode"] != "none":

                assert (
                    self._example_sampler_mixin_data["query_label_mode"]
                    != "distribution"
                    or dataset is not None
                ), "query_label_mode is 'distribution' but dataset is None"

                query = deepcopy(query)
                self._handle_labels(
                    [query],
                    dataset,
                    self._example_sampler_mixin_data["query_label_mode"],
                )
                return query["label"]
            return query["label"]

    def sample_cot(self, cots: dict[str, str], query_id: str) -> str | None:
        """Samples a chain of thought for `query_id` based on `cot_mode`."""
        if self._example_sampler_mixin_data["cot_randomize"] and cots:
            return random.choice(list(cots.values()))

        return cots.get(query_id, None)

    def sample_with_strategy(
        self,
        query: dict[str, str | torch.Tensor],
        dataset: TextDataset,
        shot: int,
    ) -> list[dict[str, str | torch.Tensor]]:
        if shot == 0 or dataset is None:
            return []

        if self.sampling_strategy == "random" or self.sampling_strategy is None:
            samples = self._random_sample(
                query=query, dataset=dataset, shot=shot
            )
        elif self.sampling_strategy == "complete":
            samples = self._complete_sample(
                query=query, dataset=dataset, shot=shot
            )
        elif self.sampling_strategy == "similarity":
            samples = self._similarity_sample(
                query=query, dataset=dataset, shot=shot
            )

        if query["id"].split(dataset.id_separator)[1] == "aggregate":
            self._handle_labels(
                samples, dataset, self._example_sampler_mixin_data["label_mode"]
            )

        return samples

    def _random_sample(
        self,
        query: dict[str, str | torch.Tensor],
        dataset: TextDataset,
        shot: int,
        **kwargs,
    ) -> list[dict[str, str | torch.Tensor]]:
        """Randomly samples `shot` examples from `dataset` to use in the prompt
        of `query`.

        Args:
            query: query dictionary.
            dataset: dataset to sample from.
            shot: number of examples to sample.

        Returns:
            List of sampled examples.
        """

        random.seed(self._example_sampler_mixin_data["seed"])

        annotator = query["id"].split(dataset.id_separator)[1]
        inds = dataset.annotator2inds[annotator]
        inds = random.sample(inds, shot)
        return [self.train_dataset[i] for i in inds]

    def _complete_sample(
        self,
        query: dict[str, str | torch.Tensor],
        dataset: TextDataset,
        shot: int,
    ) -> list[dict[str, str | torch.Tensor]]:
        """Samples `shot` examples from `dataset` to use in the prompt
        of `query` that contain all labels.

        This is not implemented optimally, only heuristically, because of
        the complexity of that for multilabel datasets.

        Args:
            query: query dictionary.
            shot: number of examples to sample.
            dataset: dataset to sample from.
            seed: random seed.

        Returns:
            List of sampled examples.
        """

        def sample(shot, seed):
            random.seed(seed)

            annotator = query["id"].split(dataset.id_separator)[1]
            label_inds = dataset.annotator2label_inds[annotator]
            labels = set(dataset.label_set)

            samples = []
            inds = set()

            while labels and shot > 0:
                label = random.choice(list(labels))
                if shot == 1:
                    # make an effort to sample all remaining labels
                    for idx in label_inds[label]:
                        if (
                            set(dataset.index_label_set(dataset[idx]["label"]))
                            == labels
                        ):
                            break
                        # if not broken, we are going to get an assertion anyway
                        # so no need to handle it here
                else:
                    idx = random.choice(label_inds[label])

                # index cannot overlap because it comes from label
                # that has not been sampled yet
                inds.add(idx)
                sample = dataset[idx]
                labels.difference_update(
                    dataset.index_label_set(sample["label"])
                )
                samples.append(sample)
                shot -= 1

            assert shot > 0 or not labels

            extra_inds = random.sample(
                set(dataset.annotator2inds[annotator]).difference(inds), shot
            )
            samples.extend([dataset[i] for i in extra_inds])

            return samples

        seed = self._example_sampler_mixin_data["seed"]

        for _ in range(self._example_sampler_mixin_data["retries"]):
            try:
                return sample(shot, seed)
            except AssertionError:
                random.seed(seed)
                seed = random.randint(0, 2**32 - 1)

        raise RuntimeError(
            f"{shot}-shot not enough to sample all labels: "
            + ",".join(dataset.label_set)
        )

    def _similarity_sample(
        self,
        query: dict[str, str | torch.Tensor],
        dataset: TextDataset,
        shot: int,
    ) -> list[dict[str, str | torch.Tensor]]:
        """Samples `shot` examples from `dataset` to use in the prompt
        of `query` that are most similar to it w.r.t. sentence embeddings."""

        annotator = query["id"].split(dataset.id_separator)[1]
        inds = dataset.annotator2inds[annotator]

        if "sentence_embeddings" not in self._example_sampler_mixin_data:
            self._example_sampler_mixin_data[
                "sentence_embeddings"
            ] = self._example_sampler_mixin_data[
                "sentence_embedding_model"
            ].encode(
                [dataset[i]["text"] for i in inds],
                convert_to_tensor=True,
            )

        query_embedding = (
            self._example_sampler_mixin_data["sentence_embedding_model"]
            .encode([query["text"]], convert_to_tensor=True)
            .squeeze()
        )

        cossims = torch.cosine_similarity(
            query_embedding,
            self._example_sampler_mixin_data["sentence_embeddings"],
        )

        _, local_inds = torch.topk(cossims, k=shot)

        return [dataset[inds[i]] for i in local_inds]


class LabelSimilarityMixin:
    """Mixin for computing closest label from model prediction based
    on word similarity between label set and predictions of model."""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return dict(
            word_embeddings_model=dict(
                type=str,
                help="word embeddings model to use for similarity",
                metadata=dict(name=True),
            )
        )

    def __init__(
        self,
        word_embeddings_model: str | None = None,
        similarity_threshold: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Init.

        Args:
            word_embeddings_model: word embeddings model to use for similarity.
            threshold: similarity threshold for considering a label.
        """
        super().__init__(*args, **kwargs)

        self._label_similarity_mixin_data = dict(word_embeddings_model=None)

        if word_embeddings_model is not None:
            self._label_similarity_mixin_data["word_embeddings_model"] = (
                api.load(word_embeddings_model)
            )
            self._label_similarity_mixin_data["similarity_threshold"] = (
                similarity_threshold
            )

            if "conceptnet" in word_embeddings_model:
                lang = getattr(
                    self, "language", getattr(self, "lang", "english")
                ).lower()
                lang = langcodes.find(lang).language
                self._label_similarity_mixin_data["word_transformation"] = (
                    lambda x: f"/c/{lang}/{x.lower()}"
                )
            else:
                self._label_similarity_mixin_data["word_transformation"] = (
                    lambda x: x
                )

    def get_closest_label(self, prediction: str, label_set: list[str]) -> str:
        """Finds the closest label to `prediction` in `label_set` based on
        word similarity.

        Args:
            label: label to find the closest label to.
            label_set: set of labels to find the closest label in.

        Returns:
            Closest label to `label` in `label_set`.
        """

        wv = self._label_similarity_mixin_data["word_embeddings_model"]
        prediction = self._label_similarity_mixin_data["word_transformation"](
            prediction
        )

        if wv is None:
            return None

        if prediction not in wv.key_to_index:
            return None

        similarities = [
            wv.similarity(
                self._label_similarity_mixin_data["word_transformation"](label),
                prediction,
            )
            for label in label_set
        ]

        max_sim = max(similarities)

        if (
            self._label_similarity_mixin_data["similarity_threshold"]
            is not None
            and max_sim
            < self._label_similarity_mixin_data["similarity_threshold"]
        ):
            return None

        return label_set[similarities.index(max_sim)]


DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


class PromptBaseDataset(ExampleSamplerMixin, BaseDataset):
    """Prompt dataset for text classification, based on other TextDataset.
    Base class that doesn't use tokenization or `from_namespace`, so it can
    be inherited from."""

    name = "Prompt dataset"

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = dict(
            system_prompt=dict(
                type=str,
                help="prompt for system (use {labels} to insert labels)",
                searchable=True,
            ),
            instruction_prompt=dict(
                type=str,
                help="prompt for instructions (use {labels} to insert labels)",
                searchable=True,
            ),
            incontext_prompt=dict(
                type=str,
                help="prompt for in-context learning (use {text} and {label} "
                "to insert text and label of example)",
                searchable=True,
            ),
            query_prompt=dict(
                type=str,
                help="prompt for in-context query, if the default extension "
                "of the `incontext_prompt` is not desired (use {text} "
                "to insert text of example)",
                searchable=True,
            ),
            user_prompt=dict(
                type=str,
                help="prompt for user (use {text} to insert text of example)",
                searchable=True,
            ),
            assistant_prompt=dict(
                type=str,
                help="prompt for assistant (use {label} to insert label of example)",
                searchable=True,
            ),
            label_format=dict(
                type=str,
                help="formatting function for labels in the in-context prompt"
                " (default: join with commas)",
                searchable=True,
                metadata=dict(name=True),
            ),
            label_parser=dict(
                type=str,
                help="parsing function for labels in the response assuming "
                "`label_format` has been applied. Should return a list of strings.",
            ),
            shot=dict(
                type=int,
                required=True,
                help="number of examples to sample",
                metadata=dict(name=True, name_priority=1),
                searchable=True,
            ),
            include_query_in_demos=dict(
                type=bool,
                default=False,
                help="whether to include label of query in prompt",
                searchable=True,
            ),
            cot_csv=dict(
                type=str,
                help="path to csv file with chains of thought",
            ),
        )
        return args | ExampleSamplerMixin.argparse_args()

    def __init__(
        self,
        train_dataset: TextDataset,
        test_dataset: TextDataset,
        shot: int,
        system_prompt: str | None = None,
        instruction_prompt: str | None = None,
        incontext_prompt: str | None = None,
        query_prompt: str | None = None,
        user_prompt: str | None = None,
        assistant_prompt: str | None = None,
        label_format: str | None = None,
        label_parser: str | None = None,
        include_query_in_demos: bool = False,
        cot_csv: str | None = None,
        *args,
        **kwargs,
    ):
        """Init.

        Args:
            train_dataset: train dataset.
            test_dataset: test dataset.
            shot: number of examples to sample.
            system_prompt: prompt for system.x
            instruction_prompt: prompt for instructions.
            incontext_prompt: prompt for in-context learning.
            query_prompt: prompt for in-context query.
            user_prompt: prompt for user. Provide if
                incontent_prompt is not used.
            assistant_prompt: prompt for assistant. Provide if
                incontent_prompt is not used.
            label_format: formatting function for labels in the
                in-context prompt.
            label_parser: parsing function for labels in the response
                assuming `label_format` has been applied. Should return
                a list of strings.
            include_query_in_demos: whether to include label of query in prompt.
            args, kwargs: additional arguments, mostly for logging.
        """

        super().__init__(*args, **kwargs)
        self.include_query_in_demos = include_query_in_demos
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.any_dataset = self.train_dataset or self.test_dataset

        assert self.test_dataset is not None

        self.shot = shot
        self.label_set = self.any_dataset.label_set

        assert self.label_set == self.test_dataset.label_set

        ### Setting up system prompt
        # "is None" instead of "or" allows the user to specify an empty string
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        elif system_prompt == " ":
            system_prompt = ""
        self.system_prompt = self._format_instruction_prompt(system_prompt)

        ### Setting up instruction prompt
        if instruction_prompt is None:
            instruction_prompt = (
                instruction_prompt
                or "Classify the following examples as {} of the following: ".format(
                    "none, one, or many"
                    if self.any_dataset.multilabel
                    else "one"
                )
                + "{labels}\n"
            )
        elif instruction_prompt == " ":
            instruction_prompt = ""
        self.instruction_prompt = self._format_instruction_prompt(
            instruction_prompt
        )

        ### Setting up in-context prompts

        if user_prompt is None and assistant_prompt is not None:
            user_prompt = "{text}"
        elif user_prompt is not None and assistant_prompt is None:
            assistant_prompt = "{label}"

        if incontext_prompt is None:
            incontext_prompt = (
                (user_prompt or "{text}")
                + "\n"
                + (assistant_prompt or "{label}")
                + "\n"
            )

        self.incontext_prompt = incontext_prompt
        self.user_prompt = user_prompt
        self.assistant_prompt = assistant_prompt

        self.query_prompt = query_prompt or self.user_prompt
        if not self.query_prompt:
            idx = self.incontext_prompt.find("{cot}")
            if idx == -1:
                idx = self.incontext_prompt.find("{label}")
            self.query_prompt = self.incontext_prompt[:idx]

        if label_format and "json" in label_format:
            try:
                key = label_format.split("-")[1]
            except IndexError:
                key = "label"
            self.label_formatter = lambda x: json.dumps({key: x})
            self.label_parser = lambda x: json.loads(
                # avoids instances where JSON is followed by explanation
                x[x.find("{") : x.rfind("}") + 1]
            )[key]

        elif not label_format:
            self.label_formatter = lambda x: ", ".join(x)
            self.label_parser = lambda x: [e.strip() for e in x.split(",")]
        else:
            self.label_formatter = eval(label_format)
            if label_parser:
                try:
                    self.label_parser = eval(label_parser)
                except Exception:
                    warnings.warn(
                        "Could not parse label_parser. Using default."
                    )
                    self.label_parser = None
            else:
                self.label_parser = None

        self.ids_per_query = {}

        if cot_csv:
            cots = pd.read_csv(cot_csv, index_col=0)
            self.cots = {
                k: v["cot"] for k, v in cots.to_dict(orient="index").items()
            }
        else:
            self.cots = {}

    def debug_message(self):
        s = ["Label set: " + ", ".join(self.label_set)]
        s.append("Example prompt: " + self[0]["text"])
        return "\n".join(s)

    def __len__(self):
        return len(self.test_dataset)

    def sample(
        self, query: dict[str, str | torch.Tensor]
    ) -> list[dict[str, str | torch.Tensor]]:
        """Samples `self.shot` examples from `self.train_dataset`
        to use in the prompt of `query`."""
        samples = self.sample_with_strategy(
            query, self.train_dataset, self.shot
        )
        self.ids_per_query[query["id"]] = [sample["id"] for sample in samples]

        return samples

    def _format_instruction_prompt(
        self, instruction_prompt_template: str
    ) -> str:
        """Formats the instruction prompt."""
        labels = ", ".join(self.label_set[:-1]) + " and " + self.label_set[-1]
        return Template(
            instruction_prompt_template.replace("{labels}", "$labels")
        ).safe_substitute(labels=labels)

    def _format_user_prompt(self, user_prompt_template: str, text: str) -> str:
        """Formats the user prompt."""
        return Template(
            user_prompt_template.replace("{text}", "$text")
        ).safe_substitute(text=text)

    def _format_assistant_prompt(
        self, assistant_prompt_template: str, label: str
    ) -> str:
        """Formats the assistant prompt."""
        label = self.label_formatter(self.any_dataset.index_label_set(label))
        # if string is empty, use "none" as label
        # not if label is empty, because formatter could add other stuff
        if not label:
            label = "none"

        return Template(
            assistant_prompt_template.replace("{label}", "$label")
        ).safe_substitute(label=label)

    def _format_cot(self, template: str, cot: str | None) -> str:
        """Formats the chain of thought."""
        if not cot:
            return template
        return Template(template.replace("{cot}", "$cot")).safe_substitute(
            cot=cot
        )

    def _format_incontext_prompt(
        self,
        incontext_prompt_template: str,
        text: str,
        label: str,
        cot: str | None = None,
    ) -> str:
        """Formats the in-context prompt."""
        return self._format_cot(
            self._format_user_prompt(
                self._format_assistant_prompt(incontext_prompt_template, label),
                text,
            ),
            cot,
        )

    def __getitem__(self, index) -> dict[str, str | torch.Tensor]:
        """Returns prompt dictionary for `index`-th example in
        `test_dataset`. The return dictionary contains the following keys:
            - `id`: ID of example;
            - `query`: query text;
            - `text`: prompt text;
            - `label`: tensor label of example.

        The prompt is constructed as follows:
            1. The system and the instruction prompt are added.
            2. For each support example, the in-context prompt is added.
            3. The query prompt is added.
        """
        query = self.test_dataset[index]
        support = self.sample(query)

        if self.include_query_in_demos:
            demo_query = deepcopy(query)
            demo_query_label = self.handle_query_label(
                query,
                is_demo=True,
                dataset=self.train_dataset,
            )
            demo_query["label"] = demo_query_label
            support = [demo_query] + support
        else:
            demo_query_label = None

        prompt = [self.system_prompt or "", self.instruction_prompt or ""]
        prompt.extend(
            [
                self._format_incontext_prompt(
                    self.incontext_prompt,
                    sample["text"],
                    sample["label"],
                    self.sample_cot(self.cots, sample["id"]),
                )
                for sample in support
            ]
        )
        prompt.append(self.query_prompt.format(text=query["text"]))

        prompt = "".join(prompt)

        odict = dict(
            id=query["id"],
            query=query["text"],
            text=prompt,
            label=self.handle_query_label(query),
        )

        if demo_query_label is not None:
            odict["demo_label"] = demo_query_label

        return odict
