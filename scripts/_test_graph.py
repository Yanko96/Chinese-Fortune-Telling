"""Quick isolation test for GraphRetriever."""
import sys, os, json, traceback, time
sys.path.insert(0, 'api'); sys.path.insert(0, 'scripts')
from dotenv import load_dotenv; load_dotenv(override=True)
os.environ['CHROMA_DIR'] = './chroma_db_bge'
os.environ['EMBEDDING_MODEL'] = 'BAAI/bge-small-zh-v1.5'

print('Step 1: load vectorstore', flush=True)
import chroma_utils as _cu
_cu._vectorstore = None; _cu._embedding_function = None
from chroma_utils import get_vectorstore
vs = get_vectorstore()
print(f'  OK  count={vs._collection.count()}', flush=True)

print('Step 2: load graph + chunk_index', flush=True)
import pickle
t0 = time.perf_counter()
with open('data/knowledge_graph.pkl', 'rb') as f:
    G = pickle.load(f)
print(f'  graph OK  nodes={G.number_of_nodes()} edges={G.number_of_edges()}  ({time.perf_counter()-t0:.1f}s)', flush=True)
t0 = time.perf_counter()
with open('data/chunk_index.json', encoding='utf-8') as f:
    ci = json.load(f)
print(f'  chunk_index OK  entries={len(ci)}  ({time.perf_counter()-t0:.1f}s)', flush=True)

print('Step 3: instantiate GraphRetriever', flush=True)
try:
    from graph_retriever import GraphRetriever
    r = GraphRetriever(vectorstore=vs, graph=G, chunk_index=ci, k=10, hop=1, top_n=7)
    print(f'  OK  type={type(r).__name__}', flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)

print('Step 4: warm-up invoke (lazy-loads BGE reranker)', flush=True)
try:
    t0 = time.perf_counter()
    docs = r.invoke('如何判断正财格')
    print(f'  first invoke OK  docs={len(docs)}  ({time.perf_counter()-t0:.1f}s)', flush=True)
    t0 = time.perf_counter()
    docs2 = r.invoke('日主旺衰如何判断')
    print(f'  second invoke OK  docs={len(docs2)}  ({time.perf_counter()-t0:.1f}s)', flush=True)
    for d in docs[:3]:
        print(f'  [{d.metadata.get("book")}] {d.page_content[:60]}', flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)

print('\nALL OK', flush=True)
