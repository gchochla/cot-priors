from typing import Any

from legm import from_namespace

from cot_priors.base_datasets import TokenizationMixin
from cot_priors.benchmarks import GoEmotions, MFRC


class GoEmotionsDataset(GoEmotions):
    """Plain text dataset for `GoEmotions`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `GoEmotions`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GoEmotionsDatasetForTransformers(TokenizationMixin, GoEmotions):
    """Dataset with encodings for `transformers`
    for `GoEmotions`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return GoEmotions.argparse_args() | TokenizationMixin.argparse_args()

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_tokenize(self.data, self.preprocessor)


class MFRCDataset(MFRC):
    """Plain text dataset for `MFRC`. Class uses
    from_namespace decorator, so it should not be used for inheritance.

    Attributes: Check `MFRC`.
    """

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MFRCDatasetForTransformers(TokenizationMixin, MFRC):
    """Dataset with encodings for `transformers`
    for `MFRC`"""

    @staticmethod
    def argparse_args() -> dict[str, dict[str, Any]]:
        return MFRC.argparse_args() | TokenizationMixin.argparse_args()

    @from_namespace
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_tokenize(self.data, self.preprocessor)
