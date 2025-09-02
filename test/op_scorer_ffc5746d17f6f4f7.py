import re
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC

@OPERATOR_REGISTRY.register()
class PromptedScoreExplainer(OperatorABC):
    """PromptedScoreExplainer 通过 LLM 按用户自定义 prompt 对 AI 助手回答进行评分并给出解释。"""
    def __init__(self, llm_serving: LLMServingABC, system_prompt: str = "Please score the assistant response from 0 to 5 and explain your reasoning."):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.system_prompt = system_prompt

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "PromptedScoreExplainer：接入 LLM，根据用户提供的 prompt 对 AI 助手回答进行 0–5 分打分，并给出详细解释。\n"
                "输入参数：\n"
                "- llm_serving：LLM 服务对象\n"
                "- system_prompt：评分提示词，可包含评估维度\n"
                "- answer_key：AI 回答所在列名（默认 'assistant_answer'）\n"
                "- score_key：分数写入列名（默认 'score'）\n"
                "- explain_key：解释写入列名（默认 'explanation'）\n"
                "输出：返回 [score_key, explain_key]。"
            )
        return (
            "PromptedScoreExplainer: uses an LLM to rate an assistant's answer (0–5) with explanation using a custom prompt.\n"
            "Params: llm_serving, system_prompt, answer_key, score_key, explain_key. Returns [score_key, explain_key]."
        )

    def _parse_output(self, outputs: list[str]):
        scores, explanations = [], []
        for out in outputs:
            score, explanation = 0, ""
            if out is None:
                scores.append(0)
                explanations.append("")
                continue
            text = str(out).strip()
            match = re.search(r"[0-5]", text)
            if match:
                score = int(match.group())
            explanation = re.sub(r"^[^\\n]*\\n?", "", text, count=1).strip()
            scores.append(score)
            explanations.append(explanation)
        return scores, explanations

    def eval(self, dataframe: pd.DataFrame, answer_key: str):
        prompts = [f"{self.system_prompt}\nAssistant response: {str(row.get(answer_key, ''))}\nPlease output in the format:\nScore:<number>\nExplanation:<text>" for _, row in dataframe.iterrows()]
        try:
            self.logger.info("Requesting LLM for evaluation…")
            outputs = self.llm_serving.generate_from_input(prompts)
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            outputs = [None] * len(dataframe)
        return self._parse_output(outputs)

    def run(self, storage: DataFlowStorage, answer_key: str = "assistant_answer", score_key: str = "score", explain_key: str = "explanation"):
        df = storage.read("dataframe")
        self.logger.info(f"Loaded dataframe rows: {len(df)}")
        scores, explanations = self.eval(df, answer_key)
        df[score_key] = scores
        df[explain_key] = explanations
        storage.write(df)
        return [score_key, explain_key]


# ======== Auto-generated runner ========
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm, LocalModelLLMServing_sglang
from dataflow.core import LLMServingABC

if __name__ == "__main__":
    # 1. FileStorage
    storage = FileStorage(
        first_entry_file_name="/mnt/public/data/lh/ygc/dataflow-agent/DataFlow/dataflow/example/DataflowAgent/test.jsonl",
        cache_path="./cache_local",
        file_name_prefix="dataflow_cache_step",
        cache_type="jsonl",
    )

    # 2. LLM-Serving
    # -------- LLM Serving (Remote) --------
    llm_serving = APILLMServing_request(
        api_url="http://123.129.219.111:3000/v1/chat/completions",
        key_name_of_api_key = 'DF_API_KEY',
        model_name="gpt-4.1",
        max_workers=100,
    )
    # 若需本地模型，请改用 LocalModelLLMServing 并设置 local=True

# 3. Instantiate operator
operator = PromptedScoreExplainer(llm_serving=llm_serving, system_prompt='Please score the assistant response from 0 to 5 and explain your reasoning.')

# 4. Run
operator.run(storage=storage.step())
