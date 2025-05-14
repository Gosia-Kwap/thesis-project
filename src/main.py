from dotenv import load_dotenv
from src.model_handlers.ModelPipeline import ModelPipeline
from src.utils.log_functions import log_message
from src.utils.parsers import parse_arguments_answer_generation


def main():
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