import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    default="tqa",
    help="dataset [tqa, toxigen, bold, harmful]",
)

parser.add_argument(
    "--model",
    type=str,
    default="llama",
    help="model [llama, vicuna]",
)

parser.add_argument(
    "--prompt_type",
    type=str,
    default="freeform",
    help="prompt type [freeform, choice]",
)

parser.add_argument(
    "--max_token", type=int, default=150, help="number of max new token"
)

parser.add_argument("--rp", type=float, default=1.5, help="repetition penalty")

parser.add_argument("--verbose", help="verbose mode", action="store_true")

args = parser.parse_args()
print(args)
