import sys

from sygra.core.base_task_executor import DefaultTaskExecutor
from sygra.logger.logger_config import configure_logger
from sygra.core.models.custom_models import ModelParams
from sygra.core.models.model_factory import ModelFactory
from sygra.utils import utils
import argparse
import time
import ast
import json
from pathvalidate import is_valid_filename
import os
from sygra.utils.dotenv import load_dotenv

# Sometimes there is SSL retry error; to fix it: https://github.com/huggingface/transformers/issues/17611
CURL_CA_BUNDLE = os.environ.get("CURL_CA_BUNDLE", "")
REQUESTS_CA_BUNDLE = os.environ.get("REQUESTS_CA_BUNDLE", "")
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

load_dotenv(dotenv_path=".env", override=True)


def check_model_availability(task_name):
    # get all the models used in this task
    model_config_this_task = utils.get_models_used(task_name)
    # test if all the models are active, else abort the process
    for mn, mc in model_config_this_task.items():
        if mc is None:
            logger.error(f"Model {mn} has no model configuration. Exiting the process.")
            sys.exit(1)
        mc["name"] = mn

        # create model object for inference
        mod = ModelFactory.create_model(mc)
        model_param = ModelParams(url=mc.get("url"), auth_token=mc.get("auth_token"))
        # inference
        status = mod.ping()
        print("*"*40)
        print(status)
        print("*"*40)
        if status != 200:
            logger.error(f"Model({mn}) is down. Aborting the process.")
            sys.exit(1)
    logger.info("Required models are up and running.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        help="Task name to synthesize data for",
        required=True,
    )
    parser.add_argument(
        "--start_index",
        "-si",
        type=int,
        default=0,
        help="Start index of the input dataset to pick records from",
    )
    parser.add_argument(
        "--num_records",
        "-n",
        type=int,
        default=10,
        help="Num of records to pick from the input dataset",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=25,
        help="Num of records to process in a batch",
    )
    parser.add_argument(
        "--checkpoint_interval",
        "-ci",
        type=int,
        default=100,
        help="Num of records after which an output file checkpoint is saved",
    )
    parser.add_argument(
        "--debug",
        "-d",
        type=ast.literal_eval,
        default=False,
        help="Enable debug mode",
    )
    parser.add_argument(
        "--clear_logs",
        "-cl",
        type=ast.literal_eval,
        default=False,
        help="Clear output logs before running the script",
    )
    parser.add_argument(
        "--output_with_ts",
        "-owt",
        type=ast.literal_eval,
        default=True,
        help="Generate output file with timestamp suffix",
    )
    parser.add_argument(
        "--run_name",
        "-rn",
        type=str,
        default="",
        help="Name of the run to be used in output file name and logs",
    )
    parser.add_argument(
        "--judge_stages",
        "-js",
        type=int,
        default=None,
        help="Number of sequential judge stages to run",
    )
    parser.add_argument(
        "--judge_confidence_threshold",
        "-jct",
        type=float,
        default=None,
        help="Confidence threshold (0-1 or 0-100) above which the pipeline will stop after judge stage 1",
    )

    parser.add_argument(
        "--max_attempts",
        "--max_retries",
        "-k",
        dest="k",
        type=int,
        default=None,
        help="Max attempts for retry/refinement loops (alias: --max_retries).",
    )
    parser.add_argument(
        "--max_tries",
        type=int,
        default=None,
        help="Max tries for feedback/retry loops (alias of --max_attempts).",
    )
    parser.add_argument(
        "--judge_confidence_increment",
        "-jci",
        type=float,
        default=None,
        help="Per-attempt confidence increment (0-1 or 0-100).",
    )
    parser.add_argument(
        "--judge_confidence_thresholds",
        "-jcts",
        type=str,
        default=None,
        help="Comma-separated per-attempt confidence thresholds, e.g. '65,70,75' (0-1 or 0-100).",
    )

    parser.add_argument(
        "--run_args",
        "-ra",
        type=json.loads,
        default={},
        help='Custom args for the run as a JSON string, e.g. \'{"key1": "value1", "key2": "value1"}\'',
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=ast.literal_eval,
        default=None,
        help="Override resumable config to force resume (True) or disable resume (False)",
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        type=str,
        help="Output directory for the synthesized data, if not specified, defaults to the current task directory",
        default=None,
    )

    parser.add_argument(
        "--oasst",
        "-ost",
        type=bool,
        help="Boolean flag to run the OASST mapper",
        default=False,
    )

    parser.add_argument(
        "--quality",
        "-q",
        type=bool,
        help="Boolean flag to enable quality metrics",
        default=False,
    )

    parser.add_argument(
        "--disable_metadata",
        "-dm",
        type=ast.literal_eval,
        default=False,
        help="Disable metadata collection (default: False)",
    )

    args = parser.parse_args()

    # Merge dedicated CLI args into run_args for backward compatibility.
    if args.run_args is None or not isinstance(args.run_args, dict):
        args.run_args = {}

    if getattr(args, "k", None) is not None and "max_attempts" not in args.run_args:
        args.run_args["max_attempts"] = args.k

    if getattr(args, "max_tries", None) is not None:
        if "max_tries" not in args.run_args:
            args.run_args["max_tries"] = args.max_tries
        if getattr(args, "k", None) is None:
            args.k = args.max_tries
        if "max_attempts" not in args.run_args:
            args.run_args["max_attempts"] = args.max_tries

    if getattr(args, "judge_confidence_increment", None) is not None and "judge_confidence_increment" not in args.run_args:
        args.run_args["judge_confidence_increment"] = args.judge_confidence_increment

    if getattr(args, "judge_confidence_thresholds", None) is not None and "judge_confidence_thresholds" not in args.run_args:
        args.run_args["judge_confidence_thresholds"] = args.judge_confidence_thresholds

    start = time.time()
    task_name = args.task

    # initialize logger with/without debug mode
    configure_logger(args.debug, args.clear_logs, args.run_name)
    # this import cannot be moved to the top because logger is not yet initialized
    from sygra.logger.logger_config import logger

    logger.info("------------------------------------")
    logger.info(f"STARTING SYNTHESIS FOR TASK: {task_name}")
    logger.info("------------------------------------")
    logger.info(f"SCRIPT ARGS: {args}")

    if args.run_name:
        assert is_valid_filename(args.run_name), f"Invalid run name: {args.run_name}"

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger.info(f"Output directory set to: {args.output_dir}")

    # check models are available and normalize task name
    if not task_name.startswith("tasks.") and not '/' in task_name:
        full_task_name = f"tasks.{task_name}"
        check_model_availability(full_task_name)
        args.task = full_task_name
        utils.current_task = (
            full_task_name  # Set current_task to the full task name with prefix
        )
    else:
        check_model_availability(task_name)
        utils.current_task = task_name

    specialized_executor_cls = None
    module_task_name = (
        utils.get_dot_walk_path(args.task) if "/" in args.task else args.task
    )

    # Optional override: allow callers to specify an explicit executor path.
    executor_override = None
    try:
        if isinstance(args.run_args, dict):
            executor_override = args.run_args.get("executor_path") or args.run_args.get("task_executor")
    except Exception:
        executor_override = None

    if isinstance(executor_override, str) and executor_override.strip():
        try:
            specialized_executor_cls = utils.get_func_from_str(executor_override.strip())
            logger.info(f"Using TaskExecutor override at {executor_override.strip()}")
        except Exception as e:
            logger.error(f"Failed to load TaskExecutor override {executor_override!r}: {e}")
            sys.exit(1)

    executor_paths = [
        f"{module_task_name}.task_executor_llm_as_judge_feedback_loop_v2.TaskExecutor",
        f"{module_task_name}.task_executor_feedback.TaskExecutor",
        # f"{module_task_name}.task_executor_llm_as_judge.TaskExecutor",
        # f"{module_task_name}.task_executor.TaskExecutor",
    ]
    for executor_path in executor_paths:
        if specialized_executor_cls is not None:
            break
        try:
            specialized_executor_cls = utils.get_func_from_str(executor_path)
            logger.info(f"Found specialized TaskExecutor at {executor_path}")
            break
        except (ModuleNotFoundError, AttributeError):
            continue
    if specialized_executor_cls is None:
        logger.info(
            f"No specialized TaskExecutor found for task: {task_name}. Using DefaultTaskExecutor."
        )

    task_executor = specialized_executor_cls or DefaultTaskExecutor

    logger.info(f"Running {task_executor} for task {args.task}")
    task_executor(args).execute()

    logger.info("------------------------------------")
    logger.info(
        f"SYNTHESIS COMPLETE FOR TASK: {args.task} IN {(time.time() - start):0.2f} secs"
    )
    logger.info("------------------------------------")
    os.environ["CURL_CA_BUNDLE"] = CURL_CA_BUNDLE
    os.environ["REQUESTS_CA_BUNDLE"] = REQUESTS_CA_BUNDLE
