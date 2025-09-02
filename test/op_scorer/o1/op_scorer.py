import re
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow import get_logger

@OPERATOR_REGISTRY.register()
class AssistantResponseScorer(OperatorABC):
    """
    Uses an LLM to score AI-assistant responses and provide an explanation.
    The model should output text containing an integer score (0–5) and an
    explanation, e.g.:
    "Score: 4\nExplanation: The answer is mostly correct but misses details."
    """

    def __init__(self, llm_serving: LLMServingABC,
                 system_prompt: str = "You are an evaluator. Given the user question and the assistant response, rate the response from 0 to 5 (5 is best) and provide a brief explanation. Reply in the following format:\nScore: <number>\nExplanation: <text>"):
        self.llm_serving = llm_serving
        self.system_prompt = system_prompt
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "AssistantResponseScorer：使用 LLM 对 AI 助手回答进行 0–5 分评分，并给出解释。\n"
                "输入列：question，assistant_response（可自定义）\n"
                "输出列：score，score_explanation（可自定义）"
            )
        else:
            return (
                "AssistantResponseScorer: uses an LLM to rate an assistant response (0–5) and provide an explanation.\n"
                "Input columns: question, assistant_response (customisable)\n"
                "Output columns: score, score_explanation (customisable)"
            )

    def _build_prompts(self, dataframe: pd.DataFrame, question_key: str, answer_key: str):
        inputs = []
        for _, row in dataframe.iterrows():
            q = str(row.get(question_key, "")).strip()
            a = str(row.get(answer_key, "")).strip()
            prompt = f"User Question: {q}\nAssistant Response: {a}\n" + self.system_prompt
            inputs.append(prompt)
        return inputs

    def _parse(self, outputs: list[str]):
        scores = []
        explanations = []
        for out in outputs:
            score_val = 0
            explain_val = ""
            if out:
                match_score = re.search(r"[0-5]", out)
                if match_score:
                    score_val = int(match_score.group())
                match_exp = re.search(r"Explanation[:：]?\s*(.+)", out, re.DOTALL)
                if match_exp:
                    explain_val = match_exp.group(1).strip()
            scores.append(score_val)
            explanations.append(explain_val)
        return scores, explanations

    def run(self,
            storage: DataFlowStorage,
            question_key: str = "question",
            answer_key: str = "assistant_response",
            output_score_key: str = "score",
            output_explanation_key: str = "score_explanation"):
        dataframe = storage.read("dataframe")
        if question_key not in dataframe.columns or answer_key not in dataframe.columns:
            raise ValueError("Input columns not found in dataframe")

        prompts = self._build_prompts(dataframe, question_key, answer_key)
        self.logger.info("Requesting LLM for scoring...")
        responses = self.llm_serving.generate_from_input(prompts)
        scores, explanations = self._parse(responses)

        dataframe[output_score_key] = scores
        dataframe[output_explanation_key] = explanations
        storage.write(dataframe)

        return [output_score_key, output_explanation_key]