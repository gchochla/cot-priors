from .datasets import (
    GoEmotionsDataset,
    GoEmotionsDatasetForTransformers,
    MFRCDataset,
    MFRCDatasetForTransformers,
)
from .models import LMForClassification, OpenAIClassifier
from .prompt_dataset import (
    PromptDataset,
    PromptTextDataset,
    OpenAIPromptTextDataset,
)
from .trainers import PromptEvaluator, APIPromptEvaluator
from .utils import twitter_preprocessor, reddit_preprocessor

text_preprocessor = dict(
    Twitter=twitter_preprocessor,
    Reddit=reddit_preprocessor,
    Plain=lambda *a, **k: lambda x: x,
)

CONSTANT_ARGS = dict(
    seed=dict(
        type=int,
        help="random seed",
        metadata=dict(disable_comparison=True),
        searchable=True,
    ),
)
