"""Debug RAGAS v3 - check evaluate() LLM propagation."""
import warnings
warnings.filterwarnings("ignore")

import os, sys
from dotenv import load_dotenv
load_dotenv()

import inspect
import ragas.evaluation as re_mod
src = inspect.getsource(re_mod.aevaluate)
# Print the part about llm propagation
idx = src.find("llm")
lines = src.split("\n")
# Print first 100 lines
for i, l in enumerate(lines[:120]):
    print(f"{i:3}: {l}")
