"""Debug RAGAS metric initialization."""
import os
from dotenv import load_dotenv
load_dotenv()

from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from ragas.llms import llm_factory
from openai import OpenAI

key = os.environ.get("KIMI_API_KEY")
client = OpenAI(api_key=key, base_url="https://api.moonshot.cn/v1")
eval_llm = llm_factory("moonshot-v1-8k", client=client)
print("eval_llm type:", type(eval_llm))
print("eval_llm:", eval_llm)

f = Faithfulness(llm=eval_llm)
print("\nFaithfulness type:", type(f))
print("Faithfulness bases:", type(f).__mro__)

# Check what evaluate() expects
from ragas.evaluation import aevaluate
import inspect
src = inspect.getsource(aevaluate)
# Find the validation logic
idx = src.find("initialised metric")
if idx > 0:
    print("\n--- Validation context ---")
    print(src[max(0,idx-500):idx+300])
