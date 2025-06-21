import argparse
from src.utils.Enums import MODEL_MAP, quantisation_map


def parse_arguments_answer_generation():
    """Configure and parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate perturbed outputs from language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Start index of the dataset to process"
    )
    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="End index (exclusive) of the dataset to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_MAP.keys()),
        help="Model identifier"
    )
    parser.add_argument(
        "--quantisation",
        type=str,
        required=False,
        choices=list(quantisation_map.keys()),
        help="Quantisation parameter"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="SVAMP",
        help="Task name to process"
    )

    return parser.parse_args()

def parse_arguments_evaluation():
    parser = argparse.ArgumentParser(description='Process some task results.')

    parser.add_argument('--executor', type=str, default="habrok",
                        help='Executor environment ("habrok" or local)')
    parser.add_argument('--task', type=str, default="SVAMP",
                        help='Task name (e.g., "SVAMP")')
    parser.add_argument("--index", type=int, default=None,
        help="Start index of the part of the dataset to process")
    parser.add_argument('--model', type=str, default="gemma9b",
                        help='Model name (e.g., "gemma9b")')
    parser.add_argument('--method', type=str, default="cosine",)

    args = parser.parse_args()

    return args