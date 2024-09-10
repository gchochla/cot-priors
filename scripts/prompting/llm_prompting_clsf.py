import traceback
import gridparse
from legm import splitify_namespace, ExperimentManager
from legm.argparse_utils import add_arguments, add_metadata
from accelerate import Accelerator

accelerator = Accelerator()

from cot_priors import (
    PromptDataset,
    SemEval2018Task1EcDataset,
    GoEmotionsDataset,
    MFRCDataset,
    LMForClassification,
    PromptEvaluator,
    text_preprocessor,
    CONSTANT_ARGS,
)
from cot_priors.utils import clean_cuda


DATASET = dict(
    SemEval=SemEval2018Task1EcDataset,
    GoEmotions=GoEmotionsDataset,
    MFRC=MFRCDataset,
)


def parse_args_and_metadata():
    parser = gridparse.GridArgumentParser()
    metadata = {}

    sp = parser.add_subparsers(dest="task")

    for task in DATASET:
        sp_task = sp.add_parser(task)
        metadata[task] = {}

        argparse_args = {}
        for module in [
            DATASET[task],
            PromptDataset,
            LMForClassification,
            PromptEvaluator,
            ExperimentManager,
        ]:
            argparse_args.update(module.argparse_args())

        add_arguments(sp_task, argparse_args, replace_underscores=True)
        add_metadata(metadata[task], argparse_args)

        add_arguments(sp_task, CONSTANT_ARGS, replace_underscores=True)
        add_metadata(metadata[task], CONSTANT_ARGS)

    return parser.parse_args(), metadata


# make its own function to avoid memory leaks
def loop(args, metadata):
    accelerator.print("\nCurrent setting: ", args, "\n")

    exp_manager = ExperimentManager(
        "./logs",
        args.task,
        logging_level=args.logging_level,
        description=args.description,
        alternative_experiment_name=args.alternative_experiment_name,
    )
    exp_manager.set_namespace_params(args)
    exp_manager.set_param_metadata(metadata[args.task], args)
    exp_manager.start()

    # this is done after exp_manager.set_namespace_params
    # so as not to log the actual preprocessing function
    if args.text_preprocessor:
        args.text_preprocessor = text_preprocessor[
            DATASET[args.task].source_domain
        ]()
    else:
        args.text_preprocessor = None

    train_dataset = DATASET[args.task](
        init__namespace=splitify_namespace(args, "train")
    )
    test_dataset = DATASET[args.task](
        init__namespace=splitify_namespace(args, "test"),
        annotator_ids=train_dataset.annotators,
    )
    dataset = PromptDataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        init__namespace=args,
    )

    model = LMForClassification(
        init__namespace=args,
        labels=dataset.label_set,
        tokenizer=dataset.get_tokenizer(),
    )

    evaluator = PromptEvaluator(
        model=model, test_dataset=dataset, experiment_manager=exp_manager
    )

    evaluator.train()

    clean_cuda(model)


def main():
    grid_args, metadata = parse_args_and_metadata()
    for args in grid_args:
        try:
            loop(args, metadata)
        except Exception as e:
            print("\n\n\nError:", traceback.format_exc())
            print("\n\n\nContinuing...\n\n\n")
            clean_cuda()


if __name__ == "__main__":
    main()
