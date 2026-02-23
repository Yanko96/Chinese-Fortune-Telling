"""Debug RAGAS v2 - check singleton metric attributes."""
import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
load_dotenv()

from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
print("faithfulness type:", type(faithfulness))
print("has llm:", hasattr(faithfulness, "llm"))
print("has embeddings:", hasattr(faithfulness, "embeddings"))
attrs = [a for a in dir(faithfulness) if not a.startswith("_")]
print("attrs:", attrs[:30])

from ragas.llms import llm_factory
from openai import OpenAI
client = OpenAI(api_key=os.environ["KIMI_API_KEY"], base_url="https://api.moonshot.cn/v1")
eval_llm = llm_factory("moonshot-v1-8k", client=client)
print("\neval_llm type:", type(eval_llm))
from ragas.llms.base import BaseRagasLLM
print("eval_llm is BaseRagasLLM:", isinstance(eval_llm, BaseRagasLLM))

# Try setting llm
faithfulness.llm = eval_llm
print("\nFaithfulness llm set successfully")

# Try running evaluate with small dataset
from datasets import Dataset
from ragas import evaluate

data = {
    "question": ["什么是五行?"],
    "answer": ["五行是金木水火土"],
    "contexts": [["五行包括金木水火土，是中国传统哲学的基础。"]],
    "ground_truth": ["五行指金木水火土"],
}
ds = Dataset.from_dict(data)

try:
    result = evaluate(ds, metrics=[faithfulness], llm=eval_llm)
    print("evaluate result:", result)
except Exception as e:
    print("ERROR:", type(e).__name__, str(e)[:500])
