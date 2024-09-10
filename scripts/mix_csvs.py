import pandas
import gridparse
import os


def parse_args():
    parser = gridparse.GridArgumentParser()
    parser.add_argument(
        "--input-csvs",
        type=str,
        required=True,
        nargs="+",
        help="Paths to the CSV files to be mixed.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to the output CSV file.",
    )
    return parser.parse_args()[0]


def main():
    args = parse_args()
    dataframes = [
        pandas.read_csv(csv, index_col="id") for csv in args.input_csvs
    ]
    mixed = pandas.concat(dataframes)
    # sort by index
    mixed.sort_index(inplace=True)
    mixed.to_csv(args.output_csv)


if __name__ == "__main__":
    main()
