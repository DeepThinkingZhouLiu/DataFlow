import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
from typing import List, Dict, Any, Tuple
import requests
from datasets import load_dataset
from huggingface_hub import HfApi
from openai import OpenAI
import concurrent.futures
from collections import Counter
import re

from dataflow import get_logger
from dataflow.agent.toolkits import (
    ChatResponse,
    ChatAgentRequest
)
from ..promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow.agent.servicemanager import Memory, MemoryClient
from dataflow.cli_funcs.paths import DataFlowPath

logger = get_logger()

STATIC_DIR = DataFlowPath.get_dataflow_statics_dir()
output_dir = os.path.join(STATIC_DIR, "download_datasets")

class WebCollectionAgent:
    def __init__(
        self,
        request: ChatAgentRequest,
        memory_entity: Memory,
        prompt_template: PromptsTemplateGenerator,
    ):
        self.api_key = request.api_key
        self.api_url = request.chat_api_url
        self.model = request.model
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self.request = request
        self.memory_entity = memory_entity
        self.session_id = self.memory_entity.get_session_id(request.sessionKEY)
        self.prompt_template = prompt_template
        self.output_dir = request.download_output_dir or output_dir
        self.dataset_size_category = request.dataset_size_category or '1K<n<10K'
        self.dataset_num_limit = request.dataset_num_limit or 5  # Max number of datasets to download
        
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize HuggingFace API client
        self.hf_endpoint = 'https://hf-mirror.com'
        self.hf_api = HfApi(endpoint=self.hf_endpoint)
    
    def recognize_intent(self) -> List[str]:
        """
        Use the language model to extract search keywords from the user query.
            
        Returns:
            A list of keywords to search for
        """
        try:
            # Using the model API for intent recognition
            prompt = self.prompt_template.render(
                "task_prompt_for_recognize_intent",
                user_query=self.request.target
            )

            messages = [{"role": "user", "content": prompt}]
            chat_completion = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=128)
        
            extracted_text = chat_completion.choices[0].message.content.strip()

            if extracted_text == 'No valid keyword':
                raise Exception("No valid keyword")
            # Parse comma-separated keywords
            keywords = [k.strip() for k in extracted_text.split(",")]
            logger.info(f"Extracted keywords: {keywords}")
            return keywords

        
        except Exception as e:
            logger.error(f"Error in intent recognition: {str(e)}")
            return []
    
    def search_datasets(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Search for datasets on Hugging Face based on keywords.
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            List of dictionaries
        """
        results = {}
        for keyword in keywords:
            try:
                results[keyword] = []
                logger.info(f"Searching for datasets with query: '{keyword}'")
                
                # Use the Hugging Face API to search for datasets
                datasets = self.hf_api.list_datasets(search=keyword, limit=self.dataset_num_limit, size_categories=self.dataset_size_category)
                
                
                for dataset in datasets:
                    results[keyword].append({
                        "id": dataset.id
                    })
                        
            except Exception as e:
                logger.error(f"Error searching for datasets: {str(e)}")
        
        return results

    
    def download_dataset(self, download_dir: str, dataset_id: str) -> Tuple[bool, str]:
        """
        Download a dataset from Hugging Face.
        
        Args:
            dataset_dir: Parent directory to save the dataset
            dataset_id: The ID of the dataset to download
            
        Returns:
            Tuple of (success, message)
        """
        try:
            logger.info(f"Downloading dataset: {dataset_id}")
    
            dataset_dir = os.path.join(download_dir, dataset_id.replace("/", "_"))
            
            os.makedirs(dataset_dir, exist_ok=True)
            
            dataset = load_dataset(dataset_id, cache_dir=dataset_dir)
            
            return True, f"Successfully downloaded {dataset_id} to {dataset_dir}"
            
        except Exception as e:
            error_msg = f"Error downloading dataset {dataset_id}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def process_dataset_request(self) -> Dict[str, Any]:
        """
        Process a user's dataset request from start to finish.
            
        Returns:
            Results of the operation
        """
        results = {
            "keywords": [],
            "datasets": {},
            "downloads": {}
        }
        
        # Step 1: Extract keywords from the query
        keywords = self.recognize_intent()
        if not keywords:
            logger.warning("No keywords extracted from the query")
            return results
        results["keywords"] = keywords
        logger.info(f"Extracted keywords: {', '.join(keywords)}")
        
        # Step 2: Search for datasets
        datasets = self.search_datasets(keywords)
        results["datasets"] = datasets
        
        if all(len(lst) == 0 for lst in datasets.values()):
            logger.warning("No datasets found matching the keywords")
            return results
        
        # Step 3: Download each dataset
        for keyword, dataset_infos in datasets.items():
            if not dataset_infos:
                logger.info(f"No datasets found for keyword: {keyword}")
                continue
            
            logger.info(f"Found {len(dataset_infos)} datasets for keyword: {keyword}")
            download_dir = os.path.join(self.output_dir, keyword.replace(" ", "_"), 'tmp')
            os.makedirs(download_dir, exist_ok=True)
            download_result = []

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_dataset = {
                    executor.submit(self.download_dataset, download_dir, dataset["id"]): dataset["id"]
                    for dataset in dataset_infos
                }

                for future in concurrent.futures.as_completed(future_to_dataset):
                    dataset_id = future_to_dataset[future]
                    success, message = future.result()
                    download_result.append({
                        "dataset_id": dataset_id,
                        "success": success,
                        "message": message
                    })
            
            results["downloads"][keyword] = download_result
            logger.info(f"Completed downloads for keyword: {keyword}, {Counter([res['success'] for res in download_result])[True]} datasets succeeded, {Counter([res['success'] for res in download_result])[False]} failed.")
        
        return results

    def post_process_datastes(self, download_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process the download results to dataflow-read format.
        Args:
            download_results: The raw results from the download process
        
        Returns:
            Results of download and post processing summary
        """
        results = download_results.copy()
        results['sources'] = {}

        for keyword in results['keywords']:
            if keyword not in results['downloads'].keys() or not Counter([res['success'] for res in results['downloads'][keyword]])[True]:
                results['sources'][keyword] = {'PT': [], 'SFT': []}
                continue
            
            data_sources = {'PT': [], 'SFT': []}

            data_dir = os.path.join(self.output_dir, keyword.replace(" ", "_"))
            for dataset in results['downloads'][keyword]:
                if not dataset['success']:
                    continue
                dataset_id = dataset['dataset_id']
                try:
                    data = load_dataset(os.path.join(data_dir, 'tmp', dataset_id.replace("/", "_")))
                    for split, data_content in data.items():
                        classification_result = self.classify_dataset(data_content.column_names, data_content[0])
                        category = classification_result.get('category', 'Unknown')
                        if category == 'PT':
                            data_file = os.path.join(data_dir, 'PT.jsonl')
                            with open(data_file, 'a') as f:
                                for row in data_content:
                                    text = ' '.join([str(row[field]) for field in classification_result['text']])
                                    json_obj = {'text': text}
                                    f.write(json.dumps(json_obj) + '\n')
                            data_sources['PT'].append((f'{dataset_id}_({split})', len(data_content)))
                            logger.info(f"Post-processed dataset {dataset_id} split {split} as PT, {len(data_content)} records.")

                        elif category == 'SFT':
                            data_file = os.path.join(data_dir, 'SFT.jsonl')
                            with open(data_file, 'a') as f:
                                for row in data_content:
                                    question = ' '.join([str(row[field]) for field in classification_result['question']])
                                    output = ' '.join([str(row[field]) for field in classification_result['output']])
                                    answer = ' '.join([str(row[field]) for field in classification_result['answer']])
                                    json_obj = {
                                        'question': question,
                                        'output': output,
                                        'answer': answer
                                    }
                                    f.write(json.dumps(json_obj) + '\n')
                            data_sources['SFT'].append((f'{dataset_id}_({split})', len(data_content)))
                            logger.info(f"Post-processed dataset {dataset_id} split {split} as SFT, {len(data_content)} records.")

                        else:
                            logger.warning(f"Invalid category '{category}' for dataset {dataset_id}, skipping post-processing.")
                            raise ValueError(f"Invalid category '{category}'")
                            
                except Exception as e:
                    logger.error(f"Error post-processing dataset {dataset_id}: {str(e)}")
                    continue

            results['sources'][keyword] = data_sources
        return results
            
    
    def classify_dataset(self, column_names, first_row) -> Dict[str, Any]:
        """
        Classify datasets into categories like PT/SFT etc.
        
        Args:
            column_names: List of column names in the dataset
            first_row: The first row of data to infer types
            
        Returns:
            A dictionary including category and column type mapping
        """
        try:
            prompt = self.prompt_template.render(
                "task_prompt_for_hf_dataset_classification",
                column_names=column_names,
                first_row=first_row
            )
            print(prompt)

            messages = [{"role": "user", "content": prompt}]
            chat_completion = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=512)
        
            extracted_text = chat_completion.choices[0].message.content.strip()
            pattern = r'```json([\s\S]*?)```'
            match = re.search(pattern, extracted_text).group(1).strip() if re.search(pattern, extracted_text) else extracted_text
            print(match)

            try:
                classification_result = json.loads(match)
                return classification_result
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse GPT response as JSON: {e}")
        
        except Exception as e:
            raise Exception(f"Error during dataset classification: {e}")


    
    def create_response(self, results: Dict[str, Any]) -> ChatResponse:
        """
        Create a response to the user based on the download and post-processing results.
        
        Args:
            results: The results from processing the user's request
            
        Returns:
            A formatted response for the user
        """
        response = ChatResponse(id='0', name='test', info=self._format_response_content(results))
        return response
    
    def _format_response_content(self, results: Dict[str, Any]) -> str:
        """Format the download results into a human-readable response."""
        info = ""

        if not results['keywords']:
            info += "Sorry, I couldn't extract any valid keywords from your request."
            return info
        info += f"Extracted keywords: {', '.join(results['keywords'])}\n\n"

        if all(len(lst) == 0 for lst in results['datasets'].values()):
            info += "No datasets were found matching your keywords."
            return info
        
        info += "Datasets found:\n"
        for keyword, dataset_infos in results['datasets'].items():
            if not dataset_infos:
                info += f"- No datasets found for keyword: {keyword}\n"
                continue
            download_infos = results['downloads'][keyword]
            info += f"- {len(download_infos)} datasets found for keyword: {keyword}\n"
            for download_info in download_infos:
                status = "Download succeeded" if download_info['success'] else "Download failed"
                info += f"  - {download_info['dataset_id']}: {status}\n"
        info += "\n"

        info += "Post-processing summary:\n"
        for keyword, sources in results['sources'].items():
            info += f"- Keyword: {keyword}\n"
            info += f"-- Total count:\tPT {sum(item[1] for item in sources['PT'])} records, SFT {sum(item[1] for item in sources['SFT'])} records\n"
            if not sources['PT'] and not sources['SFT']:
                info += "  No datasets were successfully post-processed.\n"
                continue
            info += "-- Source details:\n"
            for category, datasets in sources.items():
                if datasets:
                    info += f"-- {category}:\t"
                    for dataset_id, record_count in datasets:
                        info += f"{dataset_id}: {record_count} records\t"
                    info += "\n"
                else:
                    info += f"-- {category}: No datasets processed\n"
        
        return info.strip()





    
    def run(self) -> ChatResponse:
        download_results = self.process_dataset_request()
        results = self.post_process_datastes(download_results)
        return self.create_response(results)