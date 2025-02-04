import warnings

from typing import Any

import torch
from legm import from_namespace

from cot_priors.base_datasets import TokenizationMixin
from cot_priors.base_prompts import PromptBaseDataset


class OpenAIPromptTextDataset(PromptBaseDataset):
    """Prompt dataset for text classification, based on other TextDatasetWithPriors.
    Base class that doesn't use `from_namespace`, so it can be inherited from.
    """

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = PromptBaseDataset.argparse_args()
        args.update(
            dict(
                use_system_prompt=dict(
                    type=bool,
                    default=False,
                    help="whether to use role in prompt",
                    searchable=True,
                ),
            )
        )
        return args

    @from_namespace
    def __init__(self, use_system_prompt: bool, *args, **kwargs):
        self.use_system_prompt = use_system_prompt
        if self.use_system_prompt:
            kwargs["system_prompt"] = kwargs["instruction_prompt"]
            kwargs["instruction_prompt"] = ""
            self.log("Using instruction as system prompt", "debug")
        else:
            kwargs["system_prompt"] = ""
        super().__init__(*args, **kwargs)


class PromptTextDataset(PromptBaseDataset):
    """Prompt dataset for text classification, based on other TextDatasetWithPriors.
    Uses `from_namespace`, so it shouldn't be inherited from."""

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PromptDataset(TokenizationMixin, PromptBaseDataset):
    """Prompt dataset for text classification, based on other TextDatasetWithPriors."""

    name = "Prompt tokenized dataset"

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return (
            PromptBaseDataset.argparse_args()
            | TokenizationMixin.argparse_args()
        )

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        example_prompt = self[0]
        self.log(
            "Example tokenization: "
            + self.decode(example_prompt["encoding"]["input_ids"]),
            "debug",
        )

    def debug_message(self):
        return "\n".join(
            [
                super().debug_message(),
                "Example tokenization: "
                + self.decode(self[0]["encoding"]["input_ids"]),
            ]
        )

    def __getitem__(self, index) -> dict[str, str | torch.Tensor]:
        """Returns prompt dictionary for `index`-th example in
        `test_dataset`. The return dictionary contains the following keys:
            - `id`: ID of example;
            - `query`: query text;
            - `text`: prompt text;
            - `encoding`: tokenized prompt;
            - `label`: tensor label of example.

        The prompt is constructed as follows:
            1. The instruction prompt is added.
            2. For each support example, the in-context prompt is added.
            3. The query prompt is added.
        """

        if self.user_prompt is None or self.assistant_prompt is None:
            warnings.warn(
                "Not using conversation template because user_prompt "
                "and/or assistant_prompt is not set"
            )
            item = super().__getitem__(index)
            item["encoding"] = self.tokenize(item["text"])
            return item

        query = self.test_dataset[index]
        support = self.sample(query)

        prompt = []

        if self.system_prompt:
            prompt.append(dict(role="system", content=self.system_prompt))

        for i, sample in enumerate(support):
            prompt.extend(
                [
                    dict(
                        role="user",
                        content=(self.instruction_prompt if i == 0 else "")
                        + self._format_user_prompt(
                            self.user_prompt, sample["text"]
                        ),
                    ),
                    dict(
                        role="assistant",
                        content=self._format_assistant_prompt(
                            self.assistant_prompt, sample["label"]
                        ),
                    ),
                ]
            )

        query_text = self._format_user_prompt(self.query_prompt, query["text"])
        if (
            not prompt or prompt[-1]["role"] == "system"
        ) and self.instruction_prompt:
            query_text = self.instruction_prompt + query_text

        prompt.append(dict(role="user", content=query_text))

        encoding = self.tokenize_conversation(prompt)

        return dict(
            id=query["id"],
            query=query["text"],
            text=self.decode(encoding[0]),
            encoding=dict(input_ids=encoding),
            label=query["label"],
        )
