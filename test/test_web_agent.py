import os, asyncio
import sys
sys.path.append('/mnt/public/code/zks/DataFlow')
from dataflow.cli_funcs.paths import DataFlowPath
from dataflow import get_logger
from dataflow.agent.toolkits import (
    ChatResponse,
    ChatAgentRequest
)
from dataflow.agent.servicemanager import AnalysisService, Memory
from dataflow.agent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow.agent.agentrole.web_collector import WebCollectionAgent

logger = get_logger()
memorys = {
    "web_collector": Memory()
}
BASE_DIR = DataFlowPath.get_dataflow_dir()
DATAFLOW_DIR = BASE_DIR.parent
api_key = os.environ.get("DF_API_KEY", "")
chat_api_url = os.environ.get("DF_API_URL", "")

if __name__ == "__main__":
    req = ChatAgentRequest(
            language="en",
            target="搜集一些金融和法律的数据",
            model="gpt-4o",
            sessionKEY="dataflow_demo",
            api_key=api_key,
            chat_api_url=chat_api_url
        )
    tmpl = PromptsTemplateGenerator(req.language)
    web_agent = WebCollectionAgent(
        request         = req,
        memory_entity   = memorys["web_collector"],
        prompt_template = tmpl
        )
    print(web_agent.run().info)
    # resp = web_agent.run()
    # print(resp)
