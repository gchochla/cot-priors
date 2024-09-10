import gridparse
from legm import splitify_namespace, ExperimentManager
from legm.argparse_utils import add_arguments, add_metadata

from cot_priors import (
    OpenAIPromptTextDataset,
    SemEval2018Task1EcDataset,
    GoEmotionsDataset,
    MFRCDataset,
    OpenAIClassifier,
    APIPromptEvaluator,
    text_preprocessor,
    CONSTANT_ARGS,
)


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
            OpenAIPromptTextDataset,
            OpenAIClassifier,
            APIPromptEvaluator,
            ExperimentManager,
        ]:
            argparse_args.update(module.argparse_args())

        add_arguments(sp_task, argparse_args, replace_underscores=True)
        add_metadata(metadata[task], argparse_args)

        add_arguments(sp_task, CONSTANT_ARGS, replace_underscores=True)
        add_metadata(metadata[task], CONSTANT_ARGS)

    return parser.parse_args(), metadata


def main():
    grid_args, metadata = parse_args_and_metadata()

    for _, args in enumerate(grid_args):
        print("\nCurrent setting: ", args, "\n")

        exp_manager = ExperimentManager(
            "./logs",
            args.task + "OpenAI",
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
        dataset = OpenAIPromptTextDataset(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            init__namespace=args,
            logging_file=exp_manager.logging_file,
            logging_level=exp_manager.logging_level,
        )

        model = OpenAIClassifier(
            init__namespace=args,
            labels=dataset.label_set,
        )

        evaluator = APIPromptEvaluator(
            model=model, test_dataset=dataset, experiment_manager=exp_manager
        )

        evaluator.train()


if __name__ == "__main__":
    main()
