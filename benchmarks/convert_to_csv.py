from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser()
parser.add_argument("--input-path", type=str, required=True)
parser.add_argument("--output-path", type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.read_json(args.input_path, lines=True)
    df.to_csv(args.output_path)
