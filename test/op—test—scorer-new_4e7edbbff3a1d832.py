from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
import re

@OPERATOR_REGISTRY.register()
class PromptScorer(OperatorABC):
    """
    Generic scorer that queries an LLM with a user-provided evaluation prompt and
    extracts a numeric score together with an explanation.
    The system_prompt string can contain the placeholders {instruction}, {input} and {output}
    which will be replaced by corresponding fields of the dataframe row.
    """

    def __init__(
        self,
        llm_serving: LLMServingABC = None,
        system_prompt: str = "",
        score_regex: str = r"[-+]?[0-9]*\.?[0-9]+",
        score_column: str = "PromptScore",
        explanation_column: str = "PromptExplanation",
    ):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__}…")
        self.llm_serving = llm_serving
        self.system_prompt = system_prompt
        self.score_regex = re.compile(score_regex)
        self.score_column = score_column
        self.explanation_column = explanation_column
        self.logger.info(f"{self.__class__.__name__} initialized.")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "调用LLM，根据自定义系统提示词对 AI 助手回复进行自动打分并给出解释。\n"
                "系统提示词可包含 {instruction}、{input}、{output} 占位符，分别替换为数据中的指令、输入与回复文本。\n"
                "输入参数：\n"
                "- llm_serving：实现 LLMServingABC 的 LLM 服务对象\n"
                "- system_prompt：包含占位符的评估系统提示词\n"
                "- score_regex：抽取分数的正则表达式，默认为匹配数字\n"
                "- score_column：输出分数字段名，默认 PromptScore\n"
                "- explanation_column：输出解释字段名，默认 PromptExplanation\n"
                "run 方法参数：\n"
                "- input_instruction_key：指令列名\n"
                "- input_input_key：输入列名\n"
                "- input_output_key：回复列名\n"
                "输出：在 DataFrame 新增分数和解释两列后写回存储。"
            )
        else:
            return (
                "Invoke an LLM with a custom system prompt to score assistant responses and provide explanations.\n"
                "The system prompt may include {instruction}, {input} and {output} placeholders replaced with row data.\n"
                "Input parameters:\n"
                "- llm_serving: object implementing LLMServingABC\n"
                "- system_prompt: evaluation prompt containing placeholders\n"
                "- score_regex: regex pattern to extract numeric score (default matches any number)\n"
                "- score_column: name for score output column (default PromptScore)\n"
                "- explanation_column: name for explanation output column (default PromptExplanation)\n"
                "run method params:\n"
                "- input_instruction_key: column containing instruction text\n"
                "- input_input_key: column containing input text\n"
                "- input_output_key: column containing assistant response\n"
                "Output: DataFrame augmented with score and explanation columns."
            )

    def _build_prompts(self, dataframe: pd.DataFrame, ins_key: str, in_key: str, out_key: str):
        prompts = []
        for _, row in dataframe.iterrows():
            prompt = self.system_prompt.format(
                instruction=row.get(ins_key, ""),
                input=row.get(in_key, ""),
                output=row.get(out_key, ""),
            )
            prompts.append(prompt)
        return prompts

    def _parse_response(self, response: str):
        score_match = self.score_regex.search(response)
        score = float(score_match.group()) if score_match else float('nan')
        explanation = response.strip()
        return score, explanation

    def run(
        self,
        storage: DataFlowStorage,
        input_instruction_key: str = "instruction",
        input_input_key: str = "input",
        input_output_key: str = "output",
    ):
        dataframe = storage.read("dataframe")
        prompts = self._build_prompts(dataframe, input_instruction_key, input_input_key, input_output_key)
        self.logger.info("Sending prompts to LLM…")
        responses = self.llm_serving.generate_from_input(user_inputs=prompts)
        scores, explanations = zip(*[self._parse_response(r) for r in responses])
        dataframe[self.score_column] = scores
        dataframe[self.explanation_column] = explanations
        storage.write(dataframe)


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
        api_url='http://123.129.219.111:3000/v1/chat/completions',
        key_name_of_api_key = 'DF_API_KEY',
        model_name="gpt-4o",
        max_workers=100,
    )
    # 若需本地模型，请改用 LocalModelLLMServing 并设置 local=True

# 3. Instantiate operator
operator = PromptScorer(llm_serving=llm_serving, system_prompt='', score_regex='[-+]?[0-9]*\\.?[0-9]+', score_column='PromptScore', explanation_column='PromptExplanation')

# 4. Run
operator.run(storage=storage.step())
