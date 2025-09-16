import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class PromptedSentimentClassifier(OperatorABC):
    
    def __init__(self, llm_serving: LLMServingABC, system_prompt: str = "You are a sentiment analyzer. Classify the sentiment of the given text as either 'positive' or 'negative'."):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.system_prompt = system_prompt
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "PromptedSentimentClassifier：使用 LLM 根据系统提示词对输入文本进行情感分类，输出 'positive' 或 'negative'。\n"
                "功能：对每行输入文本进行情感判断并写入结果列。\n"
                "输入参数：\n"
                "- llm_serving：LLM 服务对象，需实现 LLMServingABC 接口。\n"
                "- system_prompt：系统提示词（默认：'You are a sentiment analyzer. Classify the sentiment of the given text as either 'positive' or 'negative'.'）。\n"
                "- input_key：输入文本所在列名（默认：'raw_content'）。\n"
                "- output_key：分类结果写入的列名（默认：'sentiment'）。\n"
                "输出：\n"
                "- 返回输出列名（用于后续算子引用），分类结果已写回并保存。"
            )
        elif lang == "en":
            return (
                "PromptedSentimentClassifier: uses an LLM to classify the sentiment of input text as 'positive' or 'negative' based on the system prompt.\n"
                "Purpose: perform sentiment classification on each input row and store the result.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC.\n"
                "- system_prompt: system prompt (default: 'You are a sentiment analyzer. Classify the sentiment of the given text as either "positive" or "negative".')\n"
                "- input_key: column name containing input text (default: 'raw_content').\n"
                "- output_key: column name to store sentiment classification (default: 'sentiment').\n"
                "Output:\n"
                "- Returns the output column name for downstream operators; the classified DataFrame is saved."
            )
        else:
            return "PromptedSentimentClassifier classifies text sentiment into 'positive' or 'negative' using LLM." 

    def _parse_sentiment(self, outputs: list[str]) -> list[str]:
        results = []
        for out in outputs:
            sentiment = 'negative'
            try:
                if out is None:
                    results.append(sentiment)
                    continue
                text = str(out).strip().lower()
                if 'positive' in text:
                    sentiment = 'positive'
                elif 'negative' in text:
                    sentiment = 'negative'
                else:
                    sentiment = 'negative'  # default
            except Exception:
                sentiment = 'negative'
            results.append(sentiment)
        return results

    def classify(self, dataframe, input_key):
        llm_inputs = []
        for index, row in dataframe.iterrows():
            raw_content = row.get(input_key, '')
            if raw_content:
                llm_input = self.system_prompt + '\nText: ' + str(raw_content) + '\nClassify the sentiment as either positive or negative. Only output the label.'
                llm_inputs.append(llm_input)
        
        try:
            self.logger.info("Generating sentiment classification using the model...")
            generated_outputs = self.llm_serving.generate_from_input(llm_inputs)
            sentiments = self._parse_sentiment(generated_outputs)
            self.logger.info("Sentiment classification completed.")
        except Exception as e:
            self.logger.error(f"Error during sentiment classification: {e}")
            return
        return sentiments

    def run(self, storage: DataFlowStorage, input_key: str = "raw_content", output_key: str = "sentiment"):
        self.logger.info("Running PromptedSentimentClassifier...")

        # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")

        # Perform sentiment classification
        sentiments = self.classify(dataframe, input_key)

        # Add the classification results back to the dataframe
        dataframe[output_key] = sentiments

        # Save the updated dataframe to the output file
        output_file = storage.write(dataframe)
        return output_key