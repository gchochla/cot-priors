from typing import Any, Literal, Callable
import os
import warnings

import torch
import torch.nn as nn
from legm import from_namespace
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
)
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from dotenv import load_dotenv

from cot_priors.base_prompts import LabelSimilarityMixin


class LMForGeneration(nn.Module):
    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return dict(
            model_name_or_path=dict(
                type=str,
                required=True,
                help="model name or path to load tokenizer and model from",
                metadata=dict(name=True, name_priority=2),
                searchable=True,
            ),
            max_new_tokens=dict(
                type=int,
                help="maximum number of new tokens to generate",
            ),
            generation_max_length=dict(
                type=int,
                help="maximum length of generation (including prompt)",
            ),
            model_dtype=dict(
                type=str,
                default="float",
                help="dtype of model",
            ),
            load_in_8bit=dict(
                action="store_true",
                help="whether to load model in 8bit",
            ),
            load_in_4bit=dict(
                action="store_true",
                help="whether to load model in 4bit",
            ),
            cache_dir=dict(
                type=str,
                help="directory where models are cached, if any",
            ),
            trust_remote_code=dict(
                action="store_true",
                help="whether to trust remote code for model",
            ),
            device=dict(
                type=str,
                help="device to load model on",
            ),
        )

    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int | None = None,
        generation_max_length: int | None = None,
        model_dtype: str = "float",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        device: str | None = None,
        cache_dir: str | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        trust_remote_code: bool = False,
        *args,
        **kwargs,
    ):
        """Init.

        Args:
            model_name_or_path: model name or path to load tokenizer and model from.
            max_new_tokens: maximum number of new tokens to generate.
            generation_max_length: maximum length of generation (including prompt).
            model_dtype: dtype of model.
            load_in_8bit: whether to load model in 8bit.
            load_in_4bit: whether to load model in 4bit.
            device: device to load model on.
            cache_dir: path to `transformers` cache directory.
            tokenizer: tokenizer to use.
            trust_remote_code: whether to trust remote code for model.
        """

        assert (
            generation_max_length is not None or max_new_tokens is not None
        ), "Either generation_max_length or max_new_tokens must be provided"

        assert not (
            load_in_8bit and load_in_4bit
        ), "Only one of load_in_8bit and load_in_4bit can be provided"

        super().__init__()

        load_kwargs = dict(
            torch_dtype=(
                getattr(torch, model_dtype)
                if isinstance(model_dtype, str)
                else model_dtype
            ),
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            cache_dir=cache_dir,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )

        try:
            self.lm = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **load_kwargs
            ).eval()
            self.causal = True
        except ValueError:  # not a causal LM but a seq2seq LM
            self.lm = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path, **load_kwargs
            ).eval()
            self.causal = False

        if self.lm.generation_config.pad_token_id is None:
            if not hasattr(self.lm.generation_config.eos_token_id, "__len__"):
                self.lm.generation_config.pad_token_id = (
                    self.lm.generation_config.eos_token_id
                )
            else:
                self.lm.generation_config.pad_token_id = (
                    self.lm.generation_config.eos_token_id[0]
                )
        if max_new_tokens is not None:
            self.lm.generation_config.max_new_tokens = max_new_tokens
        else:
            self.lm.generation_config.max_length = generation_max_length

        self.tokenizer = tokenizer

    def _process_cutoff_args(
        self,
        cutoff_ids: torch.Tensor | None,
        cutoff_str: str | None,
    ) -> tuple[torch.Tensor | None, str | None]:
        """Check that the cutoff arguments are valid,
        and computes both IDs and str."""

        if not cutoff_str:
            cutoff_str = None
        if cutoff_ids is None or not cutoff_ids.tolist():
            cutoff_ids = None

        if cutoff_str is not None:
            if cutoff_ids is None:
                assert (
                    self.tokenizer is not None
                ), f"Tokenizer required to encode new example string \"{cutoff_str}\""

                # shape is 1 x tokenization_length
                cutoff_ids = self.tokenizer(
                    cutoff_str,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"]
            else:
                if self.tokenizer is not None:
                    assert torch.all(
                        cutoff_ids
                        == self.tokenizer(
                            cutoff_str,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["input_ids"]
                    ), (
                        f"Provided cutoff string `{cutoff_str}` "
                        f"and ids `{cutoff_ids.tolist()}` do not match"
                    )
        elif cutoff_ids is not None and self.tokenizer is not None:
            cutoff_str = self.tokenizer.decode(cutoff_ids)

        return cutoff_ids, cutoff_str

    @staticmethod
    def _tensor_overlap(t1: torch.Tensor, t2: torch.Tensor) -> int:
        """Computes whether smaller tensor `t2` is a subsequence
        of tensor `t1`. Both are 1D.

        Returns the index in `t1` where `t2` starts if it is a subsequence,
        otherwise the length of `t1`.
        """

        if len(t2) > len(t1):
            t1, t2 = t2, t1

        for i in range(len(t1) - len(t2) + 1):
            if (t1[i : i + len(t2)] == t2).all():
                return i

        return len(t1)

    @torch.no_grad()
    def forward(
        self,
        cutoff_ids: torch.Tensor | None = None,
        cutoff_str: str | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | list[str]]:
        """Generates text from the model.

        Args:
            cutoff_ids: ids to stop generation at.
            cutoff_str: string to stop generation at.
            **kwargs: keyword arguments to pass to the model.

        Returns:
            A dictionary if the form:
                {
                    "ids": list of generated ids before cutoff,
                    "text": list of generated text before cutoff,
                    "residual_ids": list of residual ids after cutoff,
                    "residual_text": list of residual text after cutoff,
                }
        """

        assert (
            not self.causal or "input_ids" in kwargs
        ), "input_ids required for causal LM to decode predictions"

        out = self.lm.generate(**kwargs)

        # causal models "generate" the input as well
        if self.causal:
            out = out[:, kwargs["input_ids"].shape[-1] :]

        out = {"ids": out}

        if self.tokenizer is not None:
            out["text"] = [
                self.tokenizer.decode(o, skip_special_tokens=True).strip()
                for o in out["ids"]
            ]

        cutoff_ids, cutoff_str = self._process_cutoff_args(
            cutoff_ids, cutoff_str
        )

        # remove hallucinated examples
        if cutoff_str is not None and self.tokenizer is not None:
            cutoff_inds = []
            for o in out["text"]:
                i = o.find(cutoff_str)
                # need the loop because of this
                # ow i = -1 will keep the last character in o[:i]
                if i == -1:
                    cutoff_inds.append(len(o))
                else:
                    cutoff_inds.append(i)
            out["residual_text"] = [
                o[i:] for o, i in zip(out["text"], cutoff_inds)
            ]
            out["text"] = [
                o[:i].strip() for o, i in zip(out["text"], cutoff_inds)
            ]
            out["ids"] = [
                self.tokenizer(o, return_tensors="pt")["input_ids"][0]
                for o in out["text"]
            ]
            out["residual_ids"] = [
                self.tokenizer(o, return_tensors="pt")["input_ids"][0]
                for o in out["residual_text"]
            ]

        elif cutoff_ids is not None:
            warnings.warn(
                "Provided `cutoff_ids` and no tokenizer, there's a "
                "chance tokenization doesn't yield identical IDs."
            )
            cutoff_ids = cutoff_ids.squeeze()
            cutoff_inds = [
                self._tensor_overlap(o, cutoff_ids.to(o.device))
                for o in out["ids"]
            ]
            out["residual_ids"] = [
                o[i:] for o, i in zip(out["ids"], cutoff_inds)
            ]
            out["ids"] = [o[:i] for o, i in zip(out["ids"], cutoff_inds)]

        return out


class LMForClassification(LabelSimilarityMixin, LMForGeneration):

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return (
            LMForGeneration.argparse_args()
            | LabelSimilarityMixin.argparse_args()
        )

    @from_namespace
    def __init__(
        self, labels: list[str] | dict[str, torch.Tensor], *args, **kwargs
    ):
        """Init.

        Args:
            labels: labels to use for predictions, either a list of string
                labels, or a dictionary where string labels are keys and
                their tokenization the values.
            args, kwargs: arguments to pass to LMForGeneration.
        """
        super().__init__(*args, **kwargs)
        if isinstance(labels, list):
            labels = [label.lower() for label in labels]
        else:
            labels = {label.lower(): value for label, value in labels.items()}
        self.labels = labels

    @torch.no_grad()
    def forward(
        self,
        label_parser: (
            Callable[
                [
                    str,
                ],
                list[str],
            ]
            | None
        ) = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | list[str]]:
        """Generates predictions from the model.

        Args:
            label_parser: function to parser response into a list of strings.
            **kwargs: keyword arguments to pass to the model and `LMForGeneration`.

        Returns:
            A dictionary from `LMForGeneration.forward` with an additional
            "preds" key containing the predictions as a list of strings.
        """

        out = super().forward(**kwargs)

        if "text" in out:  # almost means tokenizer
            labels = list(self.labels)  # list or keys to list

            # good if label_parser is not provided or if it fails
            out["preds"] = [
                [label for label in labels if label in o.lower()]
                for o in out["text"]
            ]

            if label_parser is not None:
                try:
                    preds = [label_parser(o) for o in out["text"]]
                    preds = [
                        [pred.lower() for pred in example_preds]
                        for example_preds in preds
                    ]
                    preds = [
                        [
                            (
                                pred
                                if pred in labels
                                else self.get_closest_label(pred, labels)
                            )
                            for pred in example_preds
                        ]
                        for example_preds in preds
                    ]
                    out["preds"] = [
                        list(
                            # in case some preds are the same
                            # because of similarity matching
                            set(
                                [
                                    pred
                                    for pred in example_pred
                                    if pred is not None
                                ]
                            )
                        )
                        for example_pred in preds
                    ]
                except:
                    pass

        else:
            assert isinstance(
                self.labels, dict
            ), "Labels must be a dict if no tokenizer is provided"
            out["preds"] = [
                [
                    label
                    for label, label_ids in self.labels.items()
                    if self._tensor_overlap(o, label_ids) < len(o)
                ]
                for o in out["ids"]
            ]

        return out


# inherit from nn.Module because of the way the trainer works
# e.g. it calls model.to(device) and model.train()
class OpenAIModel(nn.Module):
    """Wrapper for OpenAI API.

    Attributes:
        model_name: OpenAI model name to use.
        max_tokens: maximum number of tokens to generate.
        temperature: temperature to use for sampling.
        mode: mode to use for generation, can be "chat" or None.
        completion_tokens: number of tokens generated.
        prompt_tokens: number of prompt tokens.
    """

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return dict(
            model_name=dict(
                type=str,
                required=True,
                help="OpenAI model to use",
                metadata=dict(name=True, name_priority=2),
                searchable=True,
            ),
            max_new_tokens=dict(
                type=int,
                default=128,
                help="maximum number of tokens to generate",
            ),
            temperature=dict(
                type=float,
                default=0.0,
                help="temperature to use for sampling",
                searchable=True,
            ),
            mode=dict(
                type=str,
                default="chat",
                help="mode to use for generation",
            ),
        )

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        mode: Literal["chat"] | None = "chat",
    ):
        """Init.

        Args:
            model_name: OpenAI model name to use.
            max_new_tokens: maximum number of tokens to generate.
            temperature: temperature to use for sampling.
            mode: mode to use for generation, can be "chat" or None.
        """

        super().__init__()

        load_dotenv()

        self.model_name = model_name
        self.max_tokens = max_new_tokens
        self.temperature = temperature
        self.mode = mode
        self.completion_tokens = 0
        self.prompt_tokens = 0

        self.client = OpenAI()

    @retry(
        wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6)
    )
    def completion_with_backoff(self, **kwargs):
        """Retries completion with exponential backoff."""
        kwargs |= dict(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if self.mode == "chat":
            # input has to be messages=[dict(role="...", content="..."), ...]
            # use models like gpt-3.5-turbo
            resp = self.client.chat.completions.create(**kwargs)
            self.completion_tokens += resp.usage.completion_tokens
            self.prompt_tokens += resp.usage.prompt_tokens
            return resp

        # input has to be prompt="..."
        # use models like gtp-3.5-turbo-instruct
        resp = self.client.completions.create(**kwargs)
        self.completion_tokens += resp.usage.completion_tokens
        self.prompt_tokens += resp.usage.prompt_tokens
        return resp

    def __call__(self, user_prompt: str, system_prompt: str | None = None):
        """Generates completion for prompt.

        Args:
            prompt: prompt to generate completion for.
            role: role to use for chat mode.

        Returns:
            Generated completion.

        Raises:
            AssertionError: if role is None in chat mode.
        """
        if self.mode == "chat":
            messages = [
                dict(role="user", content=user_prompt),
            ]
            if system_prompt is not None:
                messages.append(dict(role="system", content=system_prompt))
            resp = self.completion_with_backoff(messages=messages)
            return resp.choices[0].message.content

        assert (
            system_prompt is None
        ), "system_prompt must be None in NON-chat mode"
        resp = self.completion_with_backoff(prompt=user_prompt)
        return resp.choices[0].text


class OpenAIClassifier(LabelSimilarityMixin, OpenAIModel):
    """Wrapper for OpenAI API for classification."""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        args = (
            OpenAIModel.argparse_args() | LabelSimilarityMixin.argparse_args()
        )
        args.pop("temperature")
        return args

    @from_namespace
    def __init__(self, labels: list[str], **kwargs):
        """Init.

        Args:
            labels: labels to use for predictions.
            **kwargs: keyword arguments to pass to OpenAIModel.
        """
        self.labels = labels
        kwargs["temperature"] = 0
        super().__init__(**kwargs)

    def __call__(
        self,
        label_parser: (
            Callable[
                [
                    str,
                ],
                list[str],
            ]
            | None
        ) = None,
        **kwargs,
    ):
        """Generates completion for prompt.

        Args:
            **kwargs: keyword arguments to pass to OpenAIModel.

        Returns:
            Generated completion and predictions based on labels.
        """
        out = {}
        # make 2d for compatibility with other models
        out["text"] = [super().__call__(**kwargs)]
        # good if label_parser is not provided or if it fails
        out["preds"] = [
            [label for label in self.labels if label in o.lower()]
            for o in out["text"]
        ]
        if label_parser is not None:
            try:
                preds = [label_parser(o) for o in out["text"]]
                preds = [
                    [pred.lower() for pred in example_preds]
                    for example_preds in preds
                ]
                preds = [
                    [
                        (
                            pred
                            if pred in self.labels
                            else self.get_closest_label(pred, self.labels)
                        )
                        for pred in example_preds
                    ]
                    for example_preds in preds
                ]
                out["preds"] = [
                    list(
                        # in case some preds are the same
                        # because of similarity matching
                        set([pred for pred in example_pred if pred is not None])
                    )
                    for example_pred in preds
                ]
            except:
                pass

        return out
