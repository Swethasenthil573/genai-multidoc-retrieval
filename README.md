## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
To develop an intelligent multidocument retrieval system that can access multiple research papers, retrieve relevant information based on user queries, and generate meaningful responses using a Large Language Model (LLM).
### DESIGN STEPS:
### STEP 1:
Import the required libraries, configure the OpenAI API key, and initialize the asynchronous execution environment using nest_asyncio.

### STEP 2:
Load three research papers and create vector search tools and summary tools for each document using LlamaIndex. Combine all the generated tools into a single list for multidocument retrieval.

### STEP 3:
Configure the GPT-4o-mini language model and text embedding model. Create a FunctionAgent using the collected document tools and submit a query about chatbot testing and evaluation results to retrieve and generate relevant information.

### PROGRAM:
```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio
nest_asyncio.apply()

urls = [
    "https://openreview.net/attachment?id=4Hq8kgrNLJ&name=pdf",
    "https://openreview.net/attachment?id=whsMqf4egO&name=pdf",
    "https://openreview.net/attachment?id=khrwY2SUDh&name=pdf",
]

papers = [
    "12_Optimizing_the_Landscape_of.pdf",
    "34_The_Use_of_AI_Tools_to_Deve.pdf",
    "18_Automated_Item_Generation_f.pdf",
]

from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
len(initial_tools)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)

response = agent.query(
    "Tell me about the evaluation dataset used in Optimizing_the_Landscape_of, "
    "and then tell me about the evaluation results"
)

response = agent.query("Give me a summary of both Automated_Item_Generation_f and The_Use_of_AI_Tools_to_Deve")
print(str(response))

```
### OUTPUT:
<img width="1500" height="1308" alt="WhatsApp Image 2026-11-24 at 5 47 38 PM" src="https://github.com/user-attachments/assets/505ac47f-d700-428b-aa2f-ae834c43146e" />

### RESULT:
Thus, the multidocument retrieval agent was successfully designed and implemented using LlamaIndex. The agent retrieved relevant information from the loaded research papers and generated meaningful responses to user queries using GPT-4o-mini.
