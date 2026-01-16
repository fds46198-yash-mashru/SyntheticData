import json
import os
from typing import Any, Optional

from sygra.core.base_task_executor import BaseTaskExecutor
from sygra.core.dataset.dataset_processor import DatasetProcessor
from sygra.core.graph.functions.edge_condition import EdgeCondition
from sygra.core.graph.functions.lambda_function import LambdaFunction
from sygra.core.graph.functions.node_processor import NodePostProcessorWithState
from sygra.core.graph.sygra_message import SygraMessage
from sygra.core.graph.sygra_state import SygraState
from sygra.logger.logger_config import logger
from sygra.utils import constants, utils


def _get_max_tries_from_args(args: Any) -> int:
    raw = None
    if hasattr(args, "k"):
        raw = getattr(args, "k")
    if raw is None and hasattr(args, "run_args") and isinstance(getattr(args, "run_args"), dict):
        ra = getattr(args, "run_args")
        raw = ra.get("max_tries", None)
        if raw is None:
            raw = ra.get("max_attempts", None)
        if raw is None:
            raw = ra.get("max_retries", None)

    try:
        v = int(raw) if raw is not None else 1
    except Exception:
        v = 1
    return max(1, v)


def _approx_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _truncate_to_token_budget(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    if _approx_token_count(text) <= token_budget:
        return text
    max_chars = max(0, token_budget * 4)
    if max_chars <= 0:
        return ""
    return text[-max_chars:]


def _append_feedback_history(state: SygraState, attempt: int, reasoning: str) -> None:
    history = state.get("judge_reasoning_history", None)
    if not isinstance(history, list):
        history = []
    reasoning_str = str(reasoning or "").strip()
    if reasoning_str:
        history.append({"attempt": attempt, "reasoning": reasoning_str})
    state["judge_reasoning_history"] = history


def _build_feedback_summary(state: SygraState, token_budget: int) -> str:
    history = state.get("judge_reasoning_history", [])
    if not isinstance(history, list) or not history:
        return ""

    parts = []
    for item in history:
        try:
            a = item.get("attempt")
            r = str(item.get("reasoning", "")).strip()
        except Exception:
            continue
        if not r:
            continue
        parts.append(f"ATTEMPT {a} FEEDBACK:\n{r}")

    combined = "\n\n".join(parts)
    return _truncate_to_token_budget(combined, token_budget)


class LoadSchema(LambdaFunction):
    @staticmethod
    def apply(lambda_node_dict: dict, state: SygraState):
        file_path = lambda_node_dict.get("file_path")
        fallback_text = lambda_node_dict.get("schema_text", "")

        original_csv = fallback_text

        try:
            if file_path and os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    original_csv = f.read()
        except Exception as e:
            logger.warning(f"Failed to load CSV from {file_path}: {e}")

        state["original_csv"] = original_csv
        return state


class PrepareGenerationInput(LambdaFunction):
    @staticmethod
    def apply(lambda_node_dict: dict, state: SygraState):
        attempt = state.get("attempt", 1)
        try:
            attempt = int(attempt)
        except Exception:
            attempt = 1

        max_attempts = state.get("max_attempts", 1)
        try:
            max_attempts = int(max_attempts)
        except Exception:
            max_attempts = 1

        original_csv = state.get("original_csv", "")

        prev_output = state.get("synthetic_table_csv", "")
        prev_output = str(prev_output or "").strip()

        generation_input_csv = original_csv

        judge_reasoning = state.get("judge_reasoning", "")
        judge_reasoning = str(judge_reasoning or "").strip()

        if attempt > 1 and judge_reasoning:
            _append_feedback_history(state, attempt - 1, judge_reasoning)

        feedback_token_budget = int(state.get("feedback_token_budget", 1200) or 1200)
        feedback = _build_feedback_summary(state, token_budget=feedback_token_budget)

        state["attempt"] = attempt
        state["max_attempts"] = max_attempts
        state["generation_input_csv"] = generation_input_csv
        state["prior_judge_reasoning"] = feedback
        state["prior_candidate_csv"] = prev_output
        return state


class SyntheticTablePostProcessor(NodePostProcessorWithState):
    def apply(self, response: SygraMessage, state: SygraState) -> SygraState:
        state["synthetic_table_csv"] = response.message.content
        return state


class JudgeReasoningPostProcessor(NodePostProcessorWithState):
    def apply(self, response: SygraMessage, state: SygraState) -> SygraState:
        reasoning = response.message.content
        state["judge_reasoning"] = reasoning
        state["judge_stop"] = "no more feedback" in str(reasoning).lower()
        return state


class WriteAttemptOutput(LambdaFunction):
    @staticmethod
    def apply(lambda_node_dict: dict, state: SygraState):
        output_subdir = lambda_node_dict.get("output_subdir", "feedback_outputs")

        attempt = state.get("attempt", 1)
        try:
            attempt = int(attempt)
        except Exception:
            attempt = 1

        record_index = state.get("record_index", 0)
        try:
            record_index = int(record_index)
        except Exception:
            record_index = 0

        total_records = state.get("total_records", 1)
        try:
            total_records = int(total_records)
        except Exception:
            total_records = 1

        output_dir = utils.get_file_in_task_dir(utils.current_task, output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        prefix = "" if total_records <= 1 else f"record_{record_index:03d}_"

        csv_path = os.path.join(output_dir, f"{prefix}output_{attempt}.csv")
        csv_text = state.get("synthetic_table_csv", "")

        try:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(str(csv_text))
        except Exception as e:
            logger.error(f"Failed to write output CSV to {csv_path}: {e}")

        state["output_csv_path"] = csv_path
        return state


class WriteAttemptReasoning(LambdaFunction):
    @staticmethod
    def apply(lambda_node_dict: dict, state: SygraState):
        output_subdir = lambda_node_dict.get("output_subdir", "feedback_outputs")

        attempt = state.get("attempt", 1)
        try:
            attempt = int(attempt)
        except Exception:
            attempt = 1

        record_index = state.get("record_index", 0)
        try:
            record_index = int(record_index)
        except Exception:
            record_index = 0

        total_records = state.get("total_records", 1)
        try:
            total_records = int(total_records)
        except Exception:
            total_records = 1

        output_dir = utils.get_file_in_task_dir(utils.current_task, output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        prefix = "" if total_records <= 1 else f"record_{record_index:03d}_"

        reasoning_path = os.path.join(output_dir, f"{prefix}reasoning_{attempt}.txt")
        reasoning = state.get("judge_reasoning", "")

        try:
            with open(reasoning_path, "w", encoding="utf-8") as f:
                f.write(str(reasoning))
        except Exception as e:
            logger.error(f"Failed to write reasoning to {reasoning_path}: {e}")

        state["reasoning_path"] = reasoning_path
        return state


class IncrementAttempt(LambdaFunction):
    @staticmethod
    def apply(lambda_node_dict: dict, state: SygraState):
        attempt = state.get("attempt", 1)
        try:
            attempt = int(attempt)
        except Exception:
            attempt = 1
        state["attempt"] = attempt + 1
        return state


class ShouldContinueFeedbackCondition(EdgeCondition):
    @staticmethod
    def apply(state: SygraState) -> str:
        attempt = state.get("attempt", 1)
        try:
            attempt = int(attempt)
        except Exception:
            attempt = 1

        max_attempts = state.get("max_attempts", 1)
        try:
            max_attempts = int(max_attempts)
        except Exception:
            max_attempts = 1

        judge_stop = state.get("judge_stop", False)
        if isinstance(judge_stop, str):
            judge_stop = judge_stop.strip().lower() in {"true", "yes", "1"}
        judge_stop = bool(judge_stop)

        if judge_stop or attempt >= max(1, max_attempts):
            return constants.SYGRA_END
        return "increment_attempt"


class TaskExecutor(BaseTaskExecutor):
    def __init__(self, args: Any, graph_config_dict: Optional[dict] = None):
        cfg = graph_config_dict
        if cfg is None:
            config_file_path = utils.get_file_in_task_dir(args.task, "graph_config.yaml")
            cfg = utils.load_yaml_file(filepath=config_file_path)
        super().__init__(args=args, graph_config_dict=cfg)

    def init_dataset(self):
        data_config = self.config.get("data_config", {})
        self._configure_sink(data_config)

        num_records = self.args.num_records
        max_attempts = _get_max_tries_from_args(self.args)

        dataset = [
            {
                "record_index": i,
                "total_records": num_records,
                "attempt": 1,
                "max_attempts": max_attempts,
                "judge_reasoning": "",
                "judge_reasoning_history": [],
                "feedback_token_budget": getattr(self.args, "feedback_token_budget", 1200),
                "judge_stop": False,
            }
            for i in range(num_records)
        ]
        return self.assign_ids(dataset)

    def execute(self):
        graph = self.init_graph()
        compiled_graph = graph.compile()
        logger.info("Graph compiled successfully")
        logger.info("\n" + compiled_graph.get_graph().draw_ascii())

        if not isinstance(self.dataset, list):
            raise ValueError("Feedback TaskExecutor expects in-memory list dataset")

        num_records_total = len(self.dataset)

        output_dir = utils.get_file_in_task_dir(self.args.task, "feedback_outputs")
        os.makedirs(output_dir, exist_ok=True)

        out_file = os.path.join(output_dir, "intermediate_output.json")
        if os.path.exists(out_file):
            utils.delete_file(out_file)

        if self.args.start_index != 0:
            self.dataset = self.dataset[self.args.start_index :]

        dataset_processor = DatasetProcessor(
            self.dataset,
            compiled_graph,
            self.graph_config,
            out_file,
            num_records_total=num_records_total,
            start_index=self.args.start_index,
            batch_size=self.args.batch_size,
            checkpoint_interval=self.args.checkpoint_interval,
            debug=self.args.debug,
            input_record_generator=self.input_record_generator,
            output_record_generator=self.output_record_generator,
            resumable=False,
            task_name=self.task_name,
        )
        dataset_processor.process_and_store_results()

        try:
            with open(out_file, "r", encoding="utf-8") as f:
                _ = json.load(f)
        except Exception:
            return
