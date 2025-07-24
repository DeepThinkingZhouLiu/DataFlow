from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd


@OPERATOR_REGISTRY.register()
class TopicClassifier(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC = None, categories: list[str] | None = None):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.llm_serving = llm_serving
        self.categories = categories or [
            "Technology", "Health", "Finance", "Sports", "Entertainment", "Politics", "Science", "Education"
        ]
        self.system_prompt = (
            "You are a helpful assistant. Classify the given text into one of the following categories: "
            + ", ".join(self.categories)
            + ". Respond with only the category name."
        )
        self.output_key = "Topic"
        self.logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "使用LLM将文本分类到预定义主题类别中，并将主题信息追加到原始数据。输入：llm_serving, categories, input_key。输出：新增列Topic的DataFrame。"
        else:
            return "Classify text into predefined topic categories with an LLM and append the topic to the data. Inputs: llm_serving, categories, input_key. Output: DataFrame with new column Topic."

    def _classify(self, texts: list[str]):
        prompts = [self.system_prompt + "\n" + text for text in texts]
        responses = self.llm_serving.generate_from_input(user_inputs=prompts)
        results = []
        for resp in responses:
            topic = resp.strip().split("\n")[0]
            if topic not in self.categories:
                topic = "Unknown"
            results.append(topic)
        return results

    def eval(self, dataframe: pd.DataFrame, input_key: str):
        self.logger.info("Classifying topics...")
        topics = self._classify(dataframe[input_key].tolist())
        self.logger.info("Classification complete!")
        return topics

    # Added a default value for `input_key` to avoid missing argument error
    def run(self, storage: DataFlowStorage, input_key: str = "question", output_key: str = "Topic"):
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        topics = self.eval(dataframe, input_key)
        dataframe[self.output_key] = topics
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
operator = TopicClassifier(llm_serving=llm_serving, categories="")

# 4. Run
operator.run(storage=storage.step())
