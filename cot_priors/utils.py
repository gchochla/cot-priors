import os
import sys
import re
import warnings
import gc
from typing import Callable

import torch
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor


def clean_cuda(*args):
    """Deletes CUDA cache."""
    for arg in args:
        del arg
    gc.collect()
    torch.cuda.empty_cache()


def twitter_preprocessor(
    normalized_tags: list | None = None, extra_tags: list | None = None
) -> Callable[
    [
        str,
    ],
    str,
]:
    """Creates a Twitter specific text preprocessor.

    Args:
        normalized_tags: `ekphrasis` tags to anonymize,
            e.g. "user" for @userNamE -> user.
        extra_tags: other `ekphrasis` normalizations,
            e.g. "repeated" for Helloooooo -> hello.

    Returns:
        The processing function.
    """

    normalized_tags = normalized_tags or ["url", "email", "phone", "user"]

    extra_tags = extra_tags or [
        "hashtag",
        "elongated",
        "allcaps",
        "repeated",
        "emphasis",
        "censored",
    ]

    def intersect_delimiters(l: list[str], demiliter: str) -> list[str]:
        new_l = []
        for elem in l:
            new_l.extend([elem, demiliter])
        return new_l

    def tag_handler_and_joiner(tokens: list[str]) -> str:
        new_tokens = []
        for token in tokens:
            for tag in normalized_tags:
                if token == f"<{tag}>":
                    token = tag
            for tag in set(extra_tags).difference(["hashtag"]):
                if token in (f"<{tag}>", f"</{tag}>"):
                    token = None
            if token:
                new_tokens.append(token)

        full_str = []
        end_pos = -1

        if "hashtag" in extra_tags:
            start_pos = -1
            while True:
                try:
                    start_pos = new_tokens.index("<hashtag>", start_pos + 1)
                    full_str.extend(
                        intersect_delimiters(
                            new_tokens[end_pos + 1 : start_pos], " "
                        )
                    )
                    end_pos = new_tokens.index("</hashtag>", start_pos + 1)
                    full_str.extend(
                        ["# "]
                        + intersect_delimiters(
                            new_tokens[start_pos + 1 : end_pos], "-"
                        )[:-1]
                        + [" "]
                    )
                except:
                    break

        full_str.extend(intersect_delimiters(new_tokens[end_pos + 1 :], " "))
        return "".join(full_str).strip()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # stop ekphrasis prints
        sys.stdout = open(os.devnull, "w")

        preprocessor = TextPreProcessor(
            normalize=normalized_tags,
            annotate=extra_tags,
            all_caps_tag="wrap",
            fix_text=False,
            segmenter="twitter_2018",
            corrector="twitter_2018",
            unpack_hashtags=True,
            unpack_contractions=True,
            spell_correct_elong=False,
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
        ).pre_process_doc

        # re-enable prints
        sys.stdout = sys.__stdout__

    fun = lambda x: tag_handler_and_joiner(preprocessor(x))
    fun.log = f"ekphrasis: {normalized_tags}, {extra_tags} | tag handler"
    return fun


def reddit_preprocessor(
    normalized_tags: list | None = None, extra_tags: list | None = None
) -> Callable[
    [
        str,
    ],
    str,
]:
    """Creates a Reddit specific text preprocessor.

    Args:
        normalized_tags: `ekphrasis` tags to anonymize,
            e.g. "user" for  /u/userNamE -> user.
        extra_tags: other `ekphrasis` normalizations,
            e.g. "repeated" for Helloooooo -> hello.

    Returns:
        The processing function.
    """

    def prepreprocessor(text):
        text = re.sub("\[NAME\]", "@name", text)
        text = re.sub("\[RELIGION\]", "religion", text)
        text = re.sub("/r/", "", text)
        text = re.sub("/u/[A-Za-z0-9_-]*", "@user", text)
        return text

    preprocessor = twitter_preprocessor(
        normalized_tags=normalized_tags, extra_tags=extra_tags
    )

    return lambda x: preprocessor(prepreprocessor(x))
