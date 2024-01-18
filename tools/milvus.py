from openai import OpenAI
from langchain.tools import Tool
from pymilvus import Collection, connections
from langchain.tools.base import StructuredTool

def get_embedding(text, model="text-embedding-ada-002"):
   client = OpenAI()
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

#Does a similarity search on milvus on a milvus collection.
def milvus_search(collection_name, filter=""):
    search_params = {"metric_type": "L2", "offset": 0, "ignore_growing": False, "params": {"nprobe": 10}}
    og_query_embedding = get_embedding("what projects have we worked on in uganda the past 5 years?")
    search_results = collection_name.search(
            data=[og_query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=50,
            expr=f"{filter}",
            output_fields=['content'],
            consistency_level="Strong"
        )
    return search_results


def run_milvus_filter(filter):
    print(f"GOT>>> {filter} LOWER {filter.lower()}")
    # Connect to Milvus db
    
    connections.connect(host='192.168.1.34', port='19530')
    collection_name = Collection('mcf_dev1')
    search_results = ""
    
    try:
        # Search with filter
        search_results = milvus_search(collection_name, filter)
    except Exception:
        #If some exception occurs, likely cause
        #could be a malfromed filter returned 
        #by GPT, so for once, try without a filter.
        search_results = milvus_search(collection_name)
    
    return search_results

#Just a wrapper if LLM decides that is needs more
#data but doens't have a suitable filter to apply.
def search_milvus_without_filter(**kwargs):
    return run_milvus_filter("")

#Tools
run_filter_tool = Tool.from_function(
    name="run_milvus_filter",
    description = (
        """
        Searches Mastercard's Milvus database. "
        This tool will simply search the Milvus database with a **SYNTACTICALLY VALID** boolean milvus filter expression.
        """
    ),
    func=run_milvus_filter
)

search_milvus_without_filter_tool = StructuredTool.from_function(
    name="search_milvus_without_filter",
    description=(
        "Can search a Mastercard's Milvus database and provide more information for generic question"
        "Expects no arguments"
    ),
    func=search_milvus_without_filter
)