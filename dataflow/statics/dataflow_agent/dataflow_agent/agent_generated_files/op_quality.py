from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import LLMServingABC
import pandas as pd
import numpy as np


@OPERATOR_REGISTRY.register()
class TrainingSampleQualityScorer(OperatorABC):
    def __init__(self,
                 llm_serving: LLMServingABC = None,
                 threshold: float | None = None,
                 batch_size: int = 16):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.llm_serving = llm_serving

        # --- fix: robustly handle empty-string threshold --------------------
        if threshold in (None, ""):
            self.threshold = None
        else:
            self.threshold = float(threshold)
        # --------------------------------------------------------------------

        self.batch_size = batch_size
        self.score_name = 'TrainingQualScore'
        self.logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def get_desc(lang: str = 'zh'):
        if lang == 'zh':
            return (
                '使用LLMServing对文本进行质量评估并输出0-1之间的连续分数，可选阈值过滤低质量样本。\n'
                '输入参数:\n'
                '- llm_serving: 实现LLMServingABC接口的LLM服务\n'
                '- threshold: 过滤阈值，可选，若提供则只保留得分≥threshold的样本\n'
                '- batch_size: 批量调用LLM的大小，默认16\n'
                '- input_key: 文本字段名，默认"question"\n'
                '- output_key: 输出得分字段名，默认"TrainingQualScore"\n'
                '输出: 添加质量得分列，并在指定阈值情况下过滤低质量行的DataFrame')
        else:
            return (
                'Evaluate text quality with LLMServing and return a continuous score between 0 and 1; can optionally filter low-quality samples.\n'
                'Input parameters:\n'
                '- llm_serving: LLM service implementing LLMServingABC\n'
                '- threshold: Optional filtering threshold; rows with score < threshold will be removed\n'
                '- batch_size: Batch size when calling the LLM, default 16\n'
                '- input_key: Field name for text, default "question"\n'
                '- output_key: Field name for score, default "TrainingQualScore"\n'
                'Output: DataFrame with an added score column and, if threshold is set, only high-quality rows retained')

    def _build_prompts(self, texts: list[str]):
        prompts = []
        for text in texts:
            prompt = (
                "You are an expert data curator. Read the following training sample and rate its overall quality on a continuous scale between 0 and 1, where 1 means excellent quality and 0 means unusable. Output ONLY the numeric score.\n\n"
                f"TEXT:\n{text}\n\nSCORE:")
            prompts.append(prompt)
        return prompts

    def _parse_scores(self, responses: list[str]):
        scores = []
        for resp in responses:
            try:
                first_line = resp.strip().split("\n")[0]
                score = float(first_line.strip().split()[0])
                scores.append(score)
            except Exception:
                scores.append(np.nan)
        return scores

    def eval(self, dataframe: pd.DataFrame, input_key: str):
        self.logger.info(f"Evaluating {self.score_name}...")
        if input_key not in dataframe.columns:
            raise KeyError(f'Input key "{input_key}" not found in DataFrame columns: {list(dataframe.columns)}')
        texts = dataframe[input_key].astype(str).tolist()
        all_scores = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            prompts = self._build_prompts(batch_texts)
            responses = self.llm_serving.generate_from_input(user_inputs=prompts)
            batch_scores = self._parse_scores(responses)
            all_scores.extend(batch_scores)
        self.logger.info("Evaluation complete!")
        return np.array(all_scores)

    def run(self,
            storage: DataFlowStorage,
            input_key: str = 'question',
            output_key: str = 'TrainingQualScore'):
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_key)
        dataframe[output_key] = scores
        if self.threshold is not None:
            dataframe = dataframe[dataframe[output_key] >= self.threshold].reset_index(drop=True)
        storage.write(dataframe)


# ======== Auto-generated runner ========
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm, LocalModelLLMServing_sglang
from dataflow.core import LLMServingABC

if __name__ == "__main__":
    # 1. FileStorage
    storage = FileStorage(
        first_entry_file_name="/mnt/h_h_public/lh/lz/DataFlow/dataflow/example/DataflowAgent/mq_test_data.jsonl",
        cache_path="./cache_local",
        file_name_prefix="dataflow_cache_step",
        cache_type="jsonl",
    )

    # 2. LLM-Serving
    # -------- LLM Serving (Remote) --------
    llm_serving = APILLMServing_request(
        api_url="http://123.129.219.111:3000/v1/chat/completions",
        key_name_of_api_key = 'DF_API_KEY',
        model_name="gpt-4o",
        max_workers=100,
    )
    # 若需本地模型，请改用 LocalModelLLMServing 并设置 local=True

# 3. Instantiate operator
operator = TrainingSampleQualityScorer(llm_serving=llm_serving, threshold="", batch_size=16)

# 4. Run
operator.run(storage=storage.step())
