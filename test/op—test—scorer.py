from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd

class _ResponseQualityPrompt:
    """Internal prompt builder for ResponseQualityScorer."""
    def __init__(self, criteria: str = "overall quality"):
        self.criteria = criteria

    def build_system_prompt(self, instruction: str, user_input: str, response: str):
        return (
            f"You are an impartial expert evaluator. Evaluate the assistant\'s response according to the following criteria: {self.criteria}.\n"
            "Rate the response on a scale from 0 (worst) to 5 (best). Provide your output in two lines:\n"
            "Line 1: only the numeric score.\n"
            "Line 2: a short explanation (within 50 words).\n"
            "---\n"
            f"Instruction: {instruction}\n"
            f"User Input: {user_input}\n"
            f"Assistant Response: {response}\n"
        )

    def build_user_prompt(self):
        return "Please provide your evaluation now."

@OPERATOR_REGISTRY.register()
class ResponseQualityScorer(OperatorABC):
    """Operator that scores AI assistant responses from 0-5 and returns an explanation."""

    def __init__(self, llm_serving: LLMServingABC = None, criteria: str = "overall quality"):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} ...")
        self.llm_serving = llm_serving
        self.score_name = "ResponseQualityScore"
        self.expl_name = "ResponseQualityExplanation"
        self.prompt = _ResponseQualityPrompt(criteria)
        self.logger.info(f"{self.__class__.__name__} initialized.")

    @staticmethod
    def get_desc(lang: str = "en"):
        if lang == "zh":
            return (
                "使用LLM评估助手响应质量，输出0-5分评分及评估解释。输入字段：instruction, input, output。"
            )
        return (
            "Evaluate assistant responses with an LLM, returning a 0-5 score and a brief explanation."
        )

    def _get_scores_and_explanations(self, samples, instr_key, input_key, output_key):
        system_prompts, user_prompts = [], []
        for s in samples:
            instr = s.get(instr_key, "")
            user_in = s.get(input_key, "")
            resp = s.get(output_key, "")
            system_prompts.append(self.prompt.build_system_prompt(instr, user_in, resp))
            user_prompts.append(self.prompt.build_user_prompt())
        full_inputs = [sp + "\n" + up for sp, up in zip(system_prompts, user_prompts)]
        llm_outputs = self.llm_serving.generate_from_input(user_inputs=full_inputs)
        scores, expls = [], []
        for out in llm_outputs:
            lines = [l for l in out.strip().split("\n") if l.strip()]
            score_val = float(lines[0].strip().split()[0]) if lines else float('nan')
            explanation = lines[1].strip() if len(lines) > 1 else ""
            scores.append(score_val)
            expls.append(explanation)
        return scores, expls

    def eval(self, df: pd.DataFrame, instruction_key: str, input_key: str, output_key: str):
        self.logger.info(f"Evaluating {self.score_name} ...")
        samples = df.to_dict(orient="records")
        scores, expls = self._get_scores_and_explanations(samples, instruction_key, input_key, output_key)
        self.logger.info("Evaluation complete!")
        return scores, expls

    def run(
        self,
        storage: DataFlowStorage,
        input_instruction_key: str = "instruction",
        input_input_key: str = "input",
        input_output_key: str = "output",
        output_score_key: str = "ResponseQualityScore",
        output_expl_key: str = "ResponseQualityExplanation",
    ):
        df = storage.read("dataframe")
        scores, expls = self.eval(df, input_instruction_key, input_input_key, input_output_key)
        df[output_score_key] = scores
        df[output_expl_key] = expls
        storage.write(df)
