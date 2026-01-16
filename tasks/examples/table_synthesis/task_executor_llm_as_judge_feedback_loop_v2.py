import json
import os
import re
from typing import Any, Dict, Optional

from regex import regex

from sygra.core.base_task_executor import BaseTaskExecutor
from sygra.core.dataset.dataset_processor import DatasetProcessor
from sygra.core.graph.functions.edge_condition import EdgeCondition
from sygra.core.graph.functions.lambda_function import LambdaFunction
from sygra.core.graph.functions.node_processor import NodePostProcessorWithState
from sygra.core.graph.sygra_message import SygraMessage
from sygra.core.graph.sygra_state import SygraState
from sygra.logger.logger_config import logger
from sygra.utils import constants, utils


def _parse_response_as_json(s: str) -> Optional[Dict[str, Any]]:
    JSON_REGEX_PATTERN = regex.compile(r"\{(?:[^{}]|(?R))*\}")
    if isinstance(s, str):
        s = s.strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
            s = re.sub(r"\n```$", "", s).strip()
    try:
        return json.loads(s)
    except json.decoder.JSONDecodeError as e:
        p = JSON_REGEX_PATTERN.search(s)
        if not p:
            logger.error("No json string found: " + e.msg)
            logger.error(s)
            return None
        try:
            return json.loads(p[0])
        except json.decoder.JSONDecodeError as e:
            logger.error("Unable to parse json string: " + e.msg)
            logger.error(s)
            return None


def _normalize_confidence(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if x > 1.0:
        x = x / 100.0
    return min(1.0, max(0.0, x))


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


def _get_max_attempts_from_args(args: Any) -> int:
    raw = None
    if hasattr(args, "k"):
        raw = getattr(args, "k")
    if raw is None and hasattr(args, "run_args") and isinstance(getattr(args, "run_args"), dict):
        ra = getattr(args, "run_args")
        raw = ra.get("k", None)
        if raw is None:
            raw = ra.get("max_attempts", None)
        if raw is None:
            raw = ra.get("max_retries", None)

    try:
        v = int(raw) if raw is not None else 1
    except Exception:
        v = 1
    return max(1, v)


def _get_judge_confidence_threshold_from_args(args: Any) -> float:
    raw = None
    if hasattr(args, "judge_confidence_threshold"):
        raw = getattr(args, "judge_confidence_threshold")
    if raw is None and hasattr(args, "run_args") and isinstance(getattr(args, "run_args"), dict):
        ra = getattr(args, "run_args")
        raw = ra.get("judge_confidence_threshold", None)
        if raw is None:
            raw = ra.get("judge_score_threshold", None)

    try:
        v = float(raw) if raw is not None else 0.9
    except Exception:
        v = 0.9
    if v > 1.0:
        v = v / 100.0
    return min(1.0, max(0.0, v))


def _get_judge_confidence_increment_from_args(args: Any) -> float:
    raw = None
    if hasattr(args, "judge_confidence_increment"):
        raw = getattr(args, "judge_confidence_increment")
    if raw is None and hasattr(args, "run_args") and isinstance(getattr(args, "run_args"), dict):
        ra = getattr(args, "run_args")
        raw = ra.get("judge_confidence_increment", None)
        if raw is None:
            raw = ra.get("judge_confidence_step", None)

    try:
        v = float(raw) if raw is not None else 0.0
    except Exception:
        v = 0.0
    if v > 1.0:
        v = v / 100.0
    return min(1.0, max(0.0, v))


def _get_judge_confidence_thresholds_from_args(args: Any) -> Optional[list[float]]:
    raw = None
    if hasattr(args, "judge_confidence_thresholds"):
        raw = getattr(args, "judge_confidence_thresholds")
    if raw is None and hasattr(args, "run_args") and isinstance(getattr(args, "run_args"), dict):
        ra = getattr(args, "run_args")
        raw = ra.get("judge_confidence_thresholds", None)
        if raw is None:
            raw = ra.get("judge_thresholds", None)

    if raw is None:
        return None

    thresholds: list[float] = []
    if isinstance(raw, (list, tuple)):
        parts = list(raw)
    else:
        parts = str(raw).split(",")

    for p in parts:
        v = _normalize_confidence(str(p).strip())
        if v is None:
            continue
        thresholds.append(v)

    return thresholds or None


def _get_target_confidence_for_stage(state: SygraState, stage: int) -> float:
    thresholds = state.get("judge_confidence_thresholds", None)
    if isinstance(thresholds, str):
        thresholds = [t.strip() for t in thresholds.split(",") if t.strip()]
    if isinstance(thresholds, list):
        parsed: list[float] = []
        for t in thresholds:
            v = _normalize_confidence(t)
            if v is not None:
                parsed.append(v)
        if parsed and stage >= 1:
            if stage <= len(parsed):
                return parsed[stage - 1]
            return parsed[-1]

    base = _normalize_confidence(state.get("judge_confidence_threshold", 0.9))
    if base is None:
        base = 0.9
    inc = _normalize_confidence(state.get("judge_confidence_increment", 0.0))
    if inc is None:
        inc = 0.0
    return min(1.0, max(0.0, base + max(0, stage - 1) * inc))


def _format_judge_reasoning(judge_obj: dict) -> str:
    if not isinstance(judge_obj, dict):
        return ""

    lines: list[str] = []

    p = judge_obj.get("pass", None)
    c = judge_obj.get("confidence", judge_obj.get("llm_confidence", None))
    s = judge_obj.get("overall_score", None)
    summary = judge_obj.get("summary", "")
    regen = judge_obj.get("regeneration_instructions", "")

    conf = _normalize_confidence(c)

    header = []
    if p is not None:
        header.append(f"pass={p}")
    if conf is not None:
        header.append(f"confidence={round(conf, 4)}")
    if s is not None:
        header.append(f"overall_score={s}")
    if header:
        lines.append("JUDGE: " + ", ".join(header))

    if isinstance(summary, str) and summary.strip():
        lines.append("SUMMARY: " + summary.strip())

    issues = judge_obj.get("issues", [])
    if isinstance(issues, list) and issues:
        lines.append("ISSUES:")
        for it in issues:
            if not isinstance(it, dict):
                continue
            cat = str(it.get("category", "")).strip()
            sev = str(it.get("severity", "")).strip()
            desc = str(it.get("description", "")).strip()
            ev = str(it.get("evidence", "")).strip()
            tag = "/".join([x for x in [cat, sev] if x])
            if tag:
                lines.append(f"- {tag}: {desc}")
            else:
                lines.append(f"- {desc}")
            if ev:
                lines.append(f"  evidence: {ev}")

    if isinstance(regen, str) and regen.strip():
        lines.append("CONSOLIDATED FIX:")
        lines.append(regen.strip())

    return "\n".join(lines).strip()


def _append_reasoning_history(state: SygraState, attempt: int, reasoning_text: str) -> None:
    history = state.get("judge_reasoning_history", None)
    if not isinstance(history, list):
        history = []
    r = str(reasoning_text or "").strip()
    if r:
        history.append({"attempt": attempt, "reasoning": r})
    state["judge_reasoning_history"] = history


def _build_reasoning_summary(state: SygraState, token_budget: int) -> str:
    history = state.get("judge_reasoning_history", [])
    if not isinstance(history, list) or not history:
        return ""

    parts: list[str] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        a = item.get("attempt", None)
        r = str(item.get("reasoning", "")).strip()
        if not r:
            continue
        parts.append(f"ATTEMPT {a} JUDGE FEEDBACK:\n{r}")

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

        original_csv = state.get("original_csv", "")
        generation_input_csv = original_csv

        state["attempt"] = attempt
        state["generation_input_csv"] = generation_input_csv

        target = _get_target_confidence_for_stage(state, attempt)
        state["judge_target_confidence"] = target
        state["judge_target_confidence_pct"] = round(target * 100.0, 2)

        state.setdefault("judge_feedback", "")
        state.setdefault("judge_reasoning_history", [])
        state.setdefault("judge_feedback_cumulative", "")
        return state


class PrepareIterativeFeedback(LambdaFunction):
    @staticmethod
    def apply(lambda_node_dict: dict, state: SygraState):
        attempt = state.get("attempt", 1)
        try:
            attempt = int(attempt)
        except Exception:
            attempt = 1

        feedback_token_budget = state.get("feedback_token_budget", 1200)
        try:
            feedback_token_budget = int(feedback_token_budget)
        except Exception:
            feedback_token_budget = 1200

        if attempt > 1:
            prev_attempt = attempt - 1
            judge_obj = state.get("judge_result", None)
            reasoning_text = _format_judge_reasoning(judge_obj) if isinstance(judge_obj, dict) else ""
            if not reasoning_text:
                reasoning_text = str(state.get("judge_feedback", "") or "").strip()
            if reasoning_text:
                _append_reasoning_history(state, prev_attempt, reasoning_text)

        state["judge_feedback_cumulative"] = _build_reasoning_summary(state, token_budget=feedback_token_budget)
        return state


class SyntheticTablePostProcessor(NodePostProcessorWithState):
    def apply(self, response: SygraMessage, state: SygraState) -> SygraState:
        state["synthetic_table_csv"] = response.message.content
        return state


def _apply_judge_result_to_state(data: Dict[str, Any], state: SygraState, stage: int) -> SygraState:
    judge_pass = data.get("pass", False)
    if isinstance(judge_pass, str):
        judge_pass = judge_pass.strip().lower() in {"true", "yes", "1"}
    judge_pass = bool(judge_pass)

    score = data.get("overall_score", None)
    try:
        score = float(score) if score is not None else None
    except Exception:
        score = None

    feedback = data.get("regeneration_instructions", "")
    if not isinstance(feedback, str):
        feedback = str(feedback)

    corrected_csv = data.get("corrected_csv", "")
    if corrected_csv is None:
        corrected_csv = ""
    if not isinstance(corrected_csv, str):
        corrected_csv = str(corrected_csv)

    if corrected_csv.strip():
        state["synthetic_table_csv"] = corrected_csv

    confidence = data.get("confidence", None)
    if confidence is None:
        confidence = data.get("llm_confidence", None)

    try:
        if confidence is None and score is not None:
            if 0.0 <= score <= 1.0:
                confidence = float(score)
            elif 0.0 <= score <= 5.0:
                confidence = float(score) / 5.0
            elif 0.0 <= score <= 100.0:
                confidence = float(score) / 100.0
        elif confidence is not None:
            confidence = float(confidence)
            if confidence > 1.0:
                confidence = confidence / 100.0
    except Exception:
        confidence = None

    state[f"judge{stage}_result"] = data
    state[f"judge{stage}_pass"] = judge_pass
    state[f"judge{stage}_score"] = score
    state[f"judge{stage}_feedback"] = feedback
    state[f"judge{stage}_confidence"] = confidence

    state["judge_result"] = data
    state["judge_pass"] = judge_pass
    state["judge_score"] = score
    state["judge_feedback"] = feedback
    state["judge_confidence"] = confidence
    state.setdefault("judge_history", []).append(
        {"stage": stage, "pass": judge_pass, "overall_score": score, "confidence": confidence, "raw": data}
    )
    return state


class JudgeTablePostProcessor(NodePostProcessorWithState):
    def apply(self, response: SygraMessage, state: SygraState) -> SygraState:
        content = response.message.content
        data = _parse_response_as_json(content) or {}
        if not isinstance(data, dict):
            data = {}

        stage = state.get("attempt", 1)
        try:
            stage = int(stage)
        except Exception:
            stage = 1
        stage = max(1, stage)

        return _apply_judge_result_to_state(data, state, stage=stage)


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


class GenerateVsRefineCondition(EdgeCondition):
    @staticmethod
    def apply(state: SygraState) -> str:
        attempt = state.get("attempt", 1)
        try:
            attempt = int(attempt)
        except Exception:
            attempt = 1

        if attempt <= 1:
            return "generate"

        candidate = state.get("synthetic_table_csv", "")
        if isinstance(candidate, str) and candidate.strip():
            return "refine"
        return "generate"


class WriteAttemptOutput(LambdaFunction):
    @staticmethod
    def apply(lambda_node_dict: dict, state: SygraState):
        output_subdir = lambda_node_dict.get("output_subdir", "output_feedback_loop")

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
                f.write(str(csv_text or ""))
        except Exception as e:
            logger.error(f"Failed to write attempt CSV to {csv_path}: {e}")

        state["attempt_output_csv_path"] = csv_path
        return state


class WriteAttemptReasoning(LambdaFunction):
    @staticmethod
    def apply(lambda_node_dict: dict, state: SygraState):
        output_subdir = lambda_node_dict.get("output_subdir", "output_feedback_loop")

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

        judge_obj = state.get("judge_result", None)
        reasoning_text = _format_judge_reasoning(judge_obj) if isinstance(judge_obj, dict) else ""
        if not reasoning_text:
            reasoning_text = str(state.get("judge_feedback", "") or "").strip()

        try:
            with open(reasoning_path, "w", encoding="utf-8") as f:
                f.write(str(reasoning_text or ""))
        except Exception as e:
            logger.error(f"Failed to write attempt reasoning to {reasoning_path}: {e}")

        state["attempt_reasoning_path"] = reasoning_path
        return state


class ShouldContinueFeedbackLoopCondition(EdgeCondition):
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

        judge_pass = state.get("judge_pass", False)
        if isinstance(judge_pass, str):
            judge_pass = judge_pass.strip().lower() in {"true", "yes", "1"}
        judge_pass = bool(judge_pass)

        stage = max(1, attempt)
        threshold = state.get("judge_target_confidence", None)
        if threshold is None:
            threshold = _get_target_confidence_for_stage(state, stage)
        try:
            threshold = float(threshold)
        except Exception:
            threshold = 0.9
        if threshold > 1.0:
            threshold = threshold / 100.0
        threshold = min(1.0, max(0.0, threshold))

        confidence = state.get(f"judge{stage}_confidence", state.get("judge_confidence", None))
        confidence = _normalize_confidence(confidence)

        success = judge_pass and (confidence is None or confidence >= threshold)

        if success or attempt >= max(1, max_attempts):
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
        max_attempts = _get_max_attempts_from_args(self.args)

        judge_confidence_threshold = _get_judge_confidence_threshold_from_args(self.args)
        judge_confidence_increment = _get_judge_confidence_increment_from_args(self.args)
        judge_confidence_thresholds = _get_judge_confidence_thresholds_from_args(self.args)

        feedback_token_budget = getattr(self.args, "feedback_token_budget", 1200)
        try:
            feedback_token_budget = int(feedback_token_budget)
        except Exception:
            feedback_token_budget = 1200

        dataset = [
            {
                "record_index": i,
                "total_records": num_records,
                "max_attempts": max_attempts,
                "attempt": 1,
                "judge_feedback": "",
                "judge_reasoning_history": [],
                "judge_feedback_cumulative": "",
                "feedback_token_budget": feedback_token_budget,
                "judge_confidence_threshold": judge_confidence_threshold,
                "judge_confidence_increment": judge_confidence_increment,
                "judge_confidence_thresholds": judge_confidence_thresholds,
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
            raise ValueError("Table synthesis TaskExecutor expects in-memory list dataset")

        num_records_total = len(self.dataset)

        output_dir = utils.get_file_in_task_dir(self.args.task, "output")
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
