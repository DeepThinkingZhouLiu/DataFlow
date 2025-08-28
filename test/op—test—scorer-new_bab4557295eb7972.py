from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import LLMServingABC
import pandas as pd
import re

@OPERATOR_REGISTRY.register()
class AccuracyScorer(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC = None):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.llm_serving = llm_serving
        self.score_name = 'AccuracyScore'
        self.logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def get_desc(lang: str = 'zh'):
        if lang == 'zh':
            return (
                '使用LLM 对助手回复的准确性进行自动化评分，并返回分数及原因说明。\n'
                '输入参数：\n'
                '- llm_serving: 实现LLMServingABC接口的服务对象\n'
                '- input_instruction_key: 指令字段名\n'
                '- input_input_key: 额外输入内容字段名\n'
                '- input_output_key: 助手回复字段名\n'
                '- output_score_key: 评分输出字段名(默认AccuracyScore)\n'
                '- output_reason_key: 原因说明字段名(默认AccuracyReason)\n'
                '输出为在原DataFrame追加两列后的结果。'
            )
        else:
            return (
                'Automatically evaluate the accuracy of an assistant\'s response with an LLM and return a numeric score and explanation.\n'
                'Input Parameters:\n'
                '- llm_serving: LLM service implementing LLMServingABC\n'
                '- input_instruction_key: field name for instruction\n'
                '- input_input_key: field name for extra input\n'
                '- input_output_key: field name for assistant response\n'
                '- output_score_key: field name for score (default AccuracyScore)\n'
                '- output_reason_key: field name for explanation (default AccuracyReason)\n'
                'The operator appends two columns to the original DataFrame.'
            )

    def _build_prompts(self, samples, instruction_key, input_key, output_key):
        prompts = []
        system_template = (
            'You are an expert evaluator. You will be given an instruction, optional input, and the assistant\'s response. '
            'Evaluate ONLY the factual and semantic accuracy of the assistant\'s response with respect to the instruction and input. '
            'Return your evaluation in the following strict JSON format: {"score": <float 0-10>, "reason": "<concise reason>"}. '
            'Higher score means higher accuracy.'
        )
        for s in samples:
            inst = s.get(instruction_key, '')
            inp = s.get(input_key, '')
            out = s.get(output_key, '')
            user_prompt = f"Instruction:\n{inst}\n\nInput:\n{inp}\n\nAssistant Response:\n{out}\n"
            full_prompt = system_template + "\n" + user_prompt
            prompts.append(full_prompt)
        return prompts

    def _parse_response(self, resp: str):
        try:
            match_score = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', resp)
            match_reason = re.search(r'"reason"\s*:\s*"([\s\S]*?)"', resp)
            score = float(match_score.group(1)) if match_score else float('nan')
            reason = match_reason.group(1).strip() if match_reason else ''
        except Exception:
            score, reason = float('nan'), ''
        return score, reason

    def eval(self, dataframe: pd.DataFrame, instruction_key: str, input_key: str, output_key: str):
        samples = dataframe.to_dict(orient='records')
        prompts = self._build_prompts(samples, instruction_key, input_key, output_key)
        self.logger.info(f"Evaluating {self.score_name} ...")
        responses = self.llm_serving.generate_from_input(user_inputs=prompts)
        scores, reasons = zip(*[self._parse_response(r) for r in responses])
        self.logger.info("Evaluation complete!")
        return list(scores), list(reasons)

    def run(
        self,
        storage: DataFlowStorage,
        input_instruction_key: str = 'instruction',
        input_input_key: str = 'input',
        input_output_key: str = 'output',
        output_score_key: str = 'AccuracyScore',
        output_reason_key: str = 'AccuracyReason'
    ):
        dataframe = storage.read('dataframe')
        scores, reasons = self.eval(dataframe, input_instruction_key, input_input_key, input_output_key)
        dataframe[output_score_key] = scores
        dataframe[output_reason_key] = reasons
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
        api_url='http://123.129.219.111:3000/v1/chat/completions',
        key_name_of_api_key = 'DF_API_KEY',
        model_name="gpt-4o",
        max_workers=100,
    )
    # 若需本地模型，请改用 LocalModelLLMServing 并设置 local=True

# 3. Instantiate operator
operator = AccuracyScorer(llm_serving=llm_serving)

# 4. Run
operator.run(storage=storage.step())
