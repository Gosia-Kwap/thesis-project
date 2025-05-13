import argparse
import load_dotenv
import os
from model_handlers.ModelPipeline import ModelPipeline
from utils.log_functions import log_message
from utils.parsers import parse_arguments_answer_generation


def main():
    """Main execution flow"""
    load_dotenv()
    args = parse_arguments_answer_generation()

    log_message(
        f"Starting execution with parameters:\n"
        f"  Model: {args.model}\n"
        f"  Quantization: {args.quantisation}\n"
        f"  Task: {args.task}\n"
        f"  Range: {args.start}-{args.end}"
    )

    try:
        pipeline = ModelPipeline(args)
        pipeline.run()
        log_message("Processing completed successfully")
    except Exception as e:
        log_message(f"Error during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()