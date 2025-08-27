from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd

@OPERATOR_REGISTRY.register()
class CriteriaScorer(OperatorABC):
    def __init__(self,
                 llm_serving: LLMServingABC = None,
                 criteria: str = "overall quality of the response",
                 score_range: tuple = (1, 10)):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.llm_serving = llm_serving
        self.criteria = criteria
        self.score_range = score_range
        self.score_name = 'CriteriaScore'
        self.logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "使用LLM根据给定评估标准为助手回复生成分数与评价。\n"
                "输入参数：\n"
                "- llm_serving：实现LLMServingABC接口的LLM服务对象\n"
                "- criteria：评价标准描述字符串\n"
                "- score_range：评分范围元组，默认为(1,10)\n"
                "- input_instruction_key：指令字段名\n"
                "- input_response_key：回复字段名\n"
                "- output_score_key：输出分数字段名，默认'CriteriaScore'\n"
                "- output_evaluation_key：输出评价字段名，默认'CriteriaEvaluation'\n"
                "输出：包含分数与文字评价两列的DataFrame"
            )
        else:
            return (
                "Use an LLM to generate a score and written evaluation for each assistant response according to supplied criteria.\n"
                "Inputs:\n"
                "- llm_serving: LLM service implementing LLMServingABC\n"
                "- criteria: description of evaluation criteria\n"
                "- score_range: tuple indicating score range, default (1,10)\n"
                "- input_instruction_key: column name for instruction text\n"
                "- input_response_key: column name for response text\n"
                "- output_score_key: column name for numeric score, default 'CriteriaScore'\n"
                "- output_evaluation_key: column name for evaluation text, default 'CriteriaEvaluation'\n"
                "Output: DataFrame with numeric score and evaluation text columns"
            )

    def _build_prompt(self, instruction, response):
        min_s, max_s = self.score_range
        system_prompt = (
            f"You are an expert evaluator. Evaluate the assistant's response according to the following criteria: {self.criteria}. "
            f"Provide your answer strictly in the following format:\n"
            f"score: <integer between {min_s} and {max_s}>\n"
            f"evaluation: <concise explanation>"
        )
        user_prompt = (
            f"# Instruction:\n{instruction}\n\n# Response:\n{response}"
        )
        return system_prompt + "\n\n" + user_prompt

    def _parse_response(self, text):
        lines = text.strip().split("\n")
        score_line = next((l for l in lines if l.lower().startswith('score')), '')
        eval_lines = [l for l in lines if l.lower().startswith('evaluation')]
        try:
            score = float(score_line.split(":")[1].strip())
        except Exception:
            score = float('nan')
        evaluation = eval_lines[0].split(":",1)[1].strip() if eval_lines else ''
        return score, evaluation

    def eval(self, dataframe: pd.DataFrame, input_instruction_key: str, input_response_key: str):
        samples = dataframe.to_dict(orient='records')
        prompts = [self._build_prompt(s.get(input_instruction_key, ''), s.get(input_response_key, '')) for s in samples]
        self.logger.info(f"Evaluating {self.score_name}...")
        responses = self.llm_serving.generate_from_input(user_inputs=prompts)
        scores, evaluations = [], []
        for resp in responses:
            score, evaluation = self._parse_response(resp)
            scores.append(score)
            evaluations.append(evaluation)
        self.logger.info("Evaluation complete!")
        return scores, evaluations

    def run(self,
            storage: DataFlowStorage,
            input_instruction_key: str = 'instruction',
            input_response_key: str = 'output',
            output_score_key: str = 'CriteriaScore',
            output_evaluation_key: str = 'CriteriaEvaluation'):
        dataframe = storage.read("dataframe")
        scores, evaluations = self.eval(dataframe, input_instruction_key, input_response_key)
        dataframe[output_score_key] = scores
        dataframe[output_evaluation_key] = evaluations
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
        api_url=http://123.129.219.111:3000/v1/chat/completions,
        key_name_of_api_key = 'DF_API_KEY',
        model_name="gpt-4o",
        max_workers=100,
    )
    # 若需本地模型，请改用 LocalModelLLMServing 并设置 local=True

# 3. Instantiate operator
operator = CriteriaScorer(llm_serving=llm_serving, criteria='overall quality of the response', score_range=(1, 10))

# 4. Run
operator.run(storage=storage.step())
