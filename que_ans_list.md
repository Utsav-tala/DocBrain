# question - Answer

## que - 1
### question : What is LCEL in LangChain?
### ans : 

**LCEL** stands for **LangChain Expression Language**.

It's a declarative way to build and compose LangChain applications by connecting components (prompts, models, parsers, retrievers, tools, etc.) into a pipeline using operators like `|`.

### Why LCEL was introduced

Before LCEL, LangChain chains were often built using classes like `LLMChain`, `SequentialChain`, and custom code. LCEL provides:

* Simpler syntax
* Better composability
* Built-in streaming support
* Async execution
* Parallel execution
* Easier debugging and tracing

---

## Basic Example

Without LCEL:

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Tell me a joke about {topic}"
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.invoke({"topic": "cats"})
```

With LCEL:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Tell me a joke about {topic}"
)

chain = prompt | llm

result = chain.invoke({"topic": "cats"})
```

The `|` operator creates a runnable pipeline.

---

## LCEL Components

Everything in LCEL is a **Runnable**.

Examples:

```python
prompt
llm
output_parser
retriever
custom_function
```

You can chain them:

```python
chain = prompt | llm | output_parser
```

Execution flow:

```text
Input
  ↓
Prompt
  ↓
LLM
  ↓
Parser
  ↓
Output
```

---

## Example with Output Parser

```python
from langchain_core.output_parsers import StrOutputParser

chain = (
    prompt
    | llm
    | StrOutputParser()
)

response = chain.invoke({"topic": "AI"})
```

---

## Parallel Execution

LCEL supports running multiple branches simultaneously using `RunnableParallel`.

```python
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    joke=prompt1 | llm,
    poem=prompt2 | llm
)

result = chain.invoke({"topic": "dogs"})
```

Output:

```python
{
    "joke": "...",
    "poem": "..."
}
```

---

## RAG Example

A typical Retrieval-Augmented Generation pipeline:

```python
from langchain_core.runnables import RunnablePassthrough

chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

Flow:

```text
Question
   ↓
Retriever
   ↓
Context + Question
   ↓
Prompt
   ↓
LLM
   ↓
Answer
```

---

## Common Runnable Methods

All LCEL chains support:

```python
chain.invoke(input)      # Single input
chain.batch(inputs)      # Multiple inputs
chain.stream(input)      # Stream output
chain.ainvoke(input)     # Async
chain.abatch(inputs)     # Async batch
```

---

## Key LCEL Operators

### Pipe (`|`)

Sequential execution:

```python
prompt | llm | parser
```

### Dictionary Composition

Run branches and merge outputs:

```python
{
    "summary": summarize_chain,
    "keywords": keyword_chain
}
```

### RunnablePassthrough

Pass input unchanged:

```python
RunnablePassthrough()
```

### RunnableLambda

Wrap Python functions:

```python
from langchain_core.runnables import RunnableLambda

uppercase = RunnableLambda(
    lambda x: x.upper()
)

chain = uppercase | llm
```

---

## Why LCEL Matters

LCEL is now the **recommended way** to build chains in modern LangChain because it:

* Replaces many older chain classes
* Makes RAG pipelines cleaner
* Supports streaming by default
* Works well with LangGraph
* Provides a unified Runnable interface

A good mental model is:

> **LCEL is to LangChain what SQL is to databases**—a concise language for expressing data flow between components.



## Que-2
### question : What are retrievers in LangChain?
### ans : 
In LangChain, **retrievers** are components that **fetch relevant documents or pieces of information** in response to a query. They sit between your data source (documents, vector database, APIs, etc.) and your LLM.

Think of a retriever as:

> **Question → Retriever → Relevant Context → LLM → Answer**

### Why use a retriever?

LLMs don't automatically know the contents of your private documents or databases. A retriever helps by finding the most relevant information and passing it to the model.

For example:

```text
User: "What is our company's vacation policy?"

Retriever:
  Searches employee handbook
  Finds relevant sections

LLM:
  Generates answer using retrieved text
```

---

## Basic Example

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

vectorstore = FAISS.load_local(
    "docs_index",
    OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

docs = retriever.invoke(
    "How does the refund policy work?"
)
```

Here:

* **FAISS** stores document embeddings.
* `as_retriever()` converts the vector store into a retriever.
* `invoke()` returns relevant documents.

---

## Common Retriever Types

### 1. Vector Store Retriever

The most common type.

Uses embeddings and similarity search.

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)
```

Returns the 5 most similar documents.

---

### 2. BM25 Retriever

Keyword-based retrieval (traditional search).

```python
from langchain_community.retrievers import BM25Retriever

retriever = BM25Retriever.from_texts(texts)
```

Good when exact keywords matter.

---

### 3. MultiQuery Retriever

Uses the LLM to generate multiple versions of a query.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
```

Query:

```text
"What are the company's leave policies?"
```

May expand to:

```text
"vacation policy"
"PTO rules"
"employee leave guidelines"
```

Then combines results.

---

### 4. Parent Document Retriever

Retrieves small chunks but returns larger parent documents.

Useful when chunking documents aggressively.

---

### 5. Contextual Compression Retriever

Retrieves documents and then filters/compresses them.

```python
from langchain.retrievers import ContextualCompressionRetriever
```

Reduces token usage by keeping only relevant sections.

---

### 6. Ensemble Retriever

Combines multiple retrievers.

Example:

```python
BM25 + Vector Search
```

This often improves retrieval quality.

---

## Retrievers in RAG

Retrievers are a core part of **RAG (Retrieval-Augmented Generation)**, a technique in which an LLM answers questions using external knowledge.

```text
User Question
      ↓
Retriever
      ↓
Relevant Documents
      ↓
Prompt + Context
      ↓
LLM
      ↓
Answer
```

Typical LangChain RAG flow:

```python
retriever = vectorstore.as_retriever()

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
```

---

## Retriever vs Vector Store

| Vector Store            | Retriever                   |
| ----------------------- | --------------------------- |
| Stores embeddings       | Retrieves relevant docs     |
| Handles indexing        | Handles search interface    |
| FAISS, Chroma, Pinecone | Wrapper around search logic |
| Can do many operations  | Focused on retrieval        |

A retriever often wraps a vector store:

```python
vectorstore → retriever → LLM
```

### Key idea

A **retriever is LangChain's standardized interface for finding relevant information** from one or more data sources. It's the component that powers most RAG applications by supplying the LLM with context before it generates an answer.



## que-3
### question : What is a vector store in langchain?
### ans : 
In LangChain, a **vector store** is a database that stores **embeddings** (vector representations of text, images, or other data) and allows you to perform **similarity search** on them.

### Why is it needed?

Large Language Models can't directly search through thousands of documents efficiently. Instead:

1. Convert documents into embeddings using an embedding model.
2. Store those embeddings in a vector store.
3. When a user asks a question:

   * Convert the question into an embedding.
   * Search the vector store for the most similar document chunks.
   * Send the retrieved chunks to the LLM as context.

This process is called **Retrieval-Augmented Generation (RAG)**.

---

### Example Flow

```text
Documents
    ↓
Text Splitter
    ↓
Embeddings Model
    ↓
Vector Store
    ↓
Similarity Search
    ↓
Relevant Chunks
    ↓
LLM
    ↓
Answer
```

---

### LangChain Example

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_texts(
    texts=[
        "LangChain is a framework for LLM applications",
        "Vector databases store embeddings"
    ],
    embedding=embeddings
)

results = vectorstore.similarity_search(
    "What stores embeddings?"
)

print(results[0].page_content)
```

Output:

```text
Vector databases store embeddings
```

---

### Common Vector Stores Supported by LangChain

| Vector Store | Type                 |
| ------------ | -------------------- |
| Chroma       | Local/Open-source    |
| FAISS        | In-memory/Local      |
| Pinecone     | Managed cloud        |
| Weaviate     | Open-source + Cloud  |
| Qdrant       | Open-source + Cloud  |
| Milvus       | Open-source          |
| PGVector     | PostgreSQL extension |

---

### Key LangChain Methods

```python
# Add documents
vectorstore.add_documents(docs)

# Similarity search
vectorstore.similarity_search(query)

# Similarity search with score
vectorstore.similarity_search_with_score(query)

# Convert to retriever
retriever = vectorstore.as_retriever()
```

---

### Vector Store vs Traditional Database

| Traditional DB      | Vector Store                                       |
| ------------------- | -------------------------------------------------- |
| Exact matching      | Semantic matching                                  |
| SQL queries         | Similarity search                                  |
| Stores rows/columns | Stores embeddings (vectors)                        |
| Finds "apple"       | Finds "apple", "fruit", "orchard" based on meaning |

For example, if a document contains:

> "The capital of France is Paris."

and the user asks:

> "What city is the French capital?"

A vector store can retrieve that document even though the wording is different, because the embeddings capture **semantic meaning**, not just exact keywords.

In LangChain, vector stores are the core component used to build RAG systems, document Q&A bots, knowledge bases, and semantic search applications.

## que-4
### que - What are output parsers?
### ans :

In **LangChain**, **output parsers** are components that take the raw text generated by an LLM and convert it into a structured format that your application can reliably use.

Without an output parser:

```python
response = llm.invoke("Give me a person's name and age")
print(response)
# "John Doe, age 32"
```

Your code would have to manually parse the string.

With an output parser:

```python
{
  "name": "John Doe",
  "age": 32
}
```

The parser handles the conversion for you.

---

## Why Output Parsers Are Needed

LLMs naturally produce text, but applications often need:

* JSON
* Lists
* Dictionaries
* Pydantic objects
* Dataclasses
* Structured API responses

Output parsers make the output:

* Consistent
* Validatable
* Easier to integrate with code

---

## Common Output Parsers in LangChain

### 1. StrOutputParser

Returns plain text.

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

result = parser.invoke("Hello World")
# "Hello World"
```

Useful when you just need a string.

---

### 2. JsonOutputParser

Converts LLM output into JSON.

```python
from langchain_core.output_parsers import JsonOutputParser
```

Example:

Prompt:

```text
Return:
{
  "name": "Alice",
  "age": 25
}
```

Output:

```python
{
    "name": "Alice",
    "age": 25
}
```

---

### 3. PydanticOutputParser

Validates output against a Pydantic model.

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
```

Parser:

```python
from langchain.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=Person)
```

Expected result:

```python
Person(name="Alice", age=25)
```

This provides type safety and validation.

---

### 4. CommaSeparatedListOutputParser

Parses comma-separated values.

```python
from langchain.output_parsers import CommaSeparatedListOutputParser
```

Input:

```text
apple, banana, mango
```

Output:

```python
["apple", "banana", "mango"]
```

---

### 5. StructuredOutputParser

Defines a custom schema using response fields.

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

schemas = [
    ResponseSchema(name="name", description="person name"),
    ResponseSchema(name="age", description="person age")
]

parser = StructuredOutputParser.from_response_schemas(schemas)
```

Output:

```python
{
    "name": "Alice",
    "age": 25
}
```

---

## How Output Parsers Work in a Chain

A common pattern:

```python
prompt | llm | parser
```

Example:

```python
from langchain_core.output_parsers import JsonOutputParser

chain = prompt | llm | JsonOutputParser()

result = chain.invoke({
    "topic": "AI"
})
```

Flow:

```text
Prompt
   ↓
LLM
   ↓
Raw Text
   ↓
Output Parser
   ↓
Python Object
```

---

## Modern Alternative: Structured Output

With newer models, LangChain often recommends using **structured output** directly instead of manually parsing JSON.

Example:

```python
structured_llm = llm.with_structured_output(Person)

result = structured_llm.invoke(
    "Generate a person"
)
```

Output:

```python
Person(name="Alice", age=25)
```

This is generally more reliable than prompting the model to generate JSON and then parsing it.

---

## Example End-to-End

```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class Movie(BaseModel):
    title: str
    year: int

llm = ChatOpenAI(model="gpt-4o")

structured_llm = llm.with_structured_output(Movie)

movie = structured_llm.invoke(
    "Give me a famous sci-fi movie"
)

print(movie)
```

Output:

```python
Movie(
    title="Interstellar",
    year=2014
)
```

### Summary

| Parser                           | Output Type              |
| -------------------------------- | ------------------------ |
| `StrOutputParser`                | String                   |
| `JsonOutputParser`               | Dictionary/JSON          |
| `PydanticOutputParser`           | Pydantic model           |
| `CommaSeparatedListOutputParser` | List                     |
| `StructuredOutputParser`         | Custom structured schema |

Think of an output parser as the **bridge between an LLM's natural-language response and the structured data your application needs**.


## que-5 - What is LangGraph?
### ans :
**LangGraph** is an open-source framework from [LangChain](https://www.langchain.com/langgraph?utm_source=chatgpt.com) for building **stateful, multi-step AI agents** as graphs.

Think of it this way:

* **LangChain** helps you connect LLMs, tools, and prompts.
* **LangGraph** helps you control the *flow* of an AI application when it needs to make decisions, loop, remember state, or coordinate multiple agents.

### Why use LangGraph?

Simple chatbots follow a linear sequence:

```
User → LLM → Response
```

More advanced AI systems often need:

* Tool calling
* Memory
* Human approval steps
* Multi-agent collaboration
* Retry loops
* Conditional branching

LangGraph lets you model these workflows as a graph:

```text
          ┌─────────┐
          │  Start  │
          └────┬────┘
               │
               ▼
        ┌─────────────┐
        │   Planner   │
        └──────┬──────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌──────────┐     ┌──────────┐
│ Research │     │ Database │
└────┬─────┘     └────┬─────┘
     └──────┬─────────┘
            ▼
      ┌──────────┐
      │  Writer  │
      └────┬─────┘
           ▼
         End
```

### Core Concepts

#### 1. State

The graph maintains shared state that flows between nodes.

```python
class State(TypedDict):
    question: str
    research: str
    answer: str
```

Each node can read and update this state.

---

#### 2. Nodes

Nodes are functions that perform work.

```python
def research_node(state):
    return {"research": "collected information"}
```

---

#### 3. Edges

Edges define what happens next.

```python
graph.add_edge("research", "writer")
```

---

#### 4. Conditional Routing

The graph can decide where to go next.

```python
def router(state):
    if state["needs_search"]:
        return "search"
    return "answer"
```

This enables agent-like behavior.

---

### Example Use Cases

#### AI Research Agent

1. Receive question
2. Search web
3. Evaluate results
4. Search again if needed
5. Generate answer

#### Customer Support Agent

1. Understand request
2. Query CRM
3. Check knowledge base
4. Escalate to human if necessary

#### Multi-Agent System

* Planner agent
* Research agent
* Coding agent
* Reviewer agent

All coordinated through a graph.

---

### Why LangGraph became popular

Before LangGraph, many agent workflows were difficult to manage because they relied on recursive chains or custom orchestration logic.

LangGraph provides:

✅ Persistent state
✅ Cycles and loops
✅ Checkpointing
✅ Human-in-the-loop workflows
✅ Streaming support
✅ Multi-agent orchestration
✅ Better production reliability

---

### Simple LangGraph Example

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    message: str

def chatbot(state):
    return {"message": state["message"] + " processed"}

graph = StateGraph(State)

graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

app = graph.compile()

result = app.invoke({"message": "hello"})
print(result)
```

---

### LangGraph vs LangChain

| Feature             | LangChain   | LangGraph      |
| ------------------- | ----------- | -------------- |
| Chains              | ✅           | ✅              |
| Tool Calling        | ✅           | ✅              |
| Agent Workflows     | ⚠️ Basic    | ✅ Advanced     |
| Loops               | ❌ Limited   | ✅ Native       |
| Stateful Execution  | ⚠️ Partial  | ✅ Core Feature |
| Multi-Agent Systems | ⚠️ Possible | ✅ Designed For |
| Human-in-the-loop   | ⚠️ Limited  | ✅ Built-in     |

In practice, many production AI agent systems today use **LangChain components inside LangGraph workflows**, with LangGraph acting as the orchestration layer.

## que-6 - 6. How do I add memory to a LangChain chain?
### ans :
In LangChain, you add memory to a chain so it can remember information from previous interactions and include that context in future prompts.

A common approach is to attach a memory object when creating the chain:

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

memory = ConversationBufferMemory()

chain = ConversationChain(
    llm=llm,
    memory=memory
)

response1 = chain.predict(input="My name is Alice.")
response2 = chain.predict(input="What is my name?")
print(response2)
```

The memory stores the conversation history, allowing the model to recall earlier information.

### Common memory types

* `ConversationBufferMemory` – stores the full conversation history.
* `ConversationBufferWindowMemory` – keeps only the most recent *k* messages.
* `ConversationSummaryMemory` – summarizes older conversations to save tokens.
* `ConversationTokenBufferMemory` – retains history up to a token limit.

Example with a window:

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5)
```

### Using memory with custom chains

If you're building an `LLMChain` with a prompt:

```python
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""
Conversation:
{history}

Human: {input}
AI:
"""
)

memory = ConversationBufferMemory(memory_key="history")

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)
```

### Newer LangChain applications

For production systems, LangChain increasingly recommends using persistent state and memory management through LangGraph, which provides more robust long-term memory and workflow control than the older chain-level memory classes.

In short: create a memory object, pass it to your chain, and ensure your prompt includes the memory variable (such as `history`) if you're using a custom prompt.


## que- 7. How do I stream responses in LangChain? 
### ans : 

In modern LangChain, streaming is usually done with `.stream()` (synchronous) or `.astream()` (asynchronous). These methods let you receive tokens or intermediate updates as they're generated instead of waiting for the full response. ([LangChain Docs][1])

### Stream LLM tokens (Python)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5")

for chunk in llm.stream("Explain recursion in one paragraph"):
    print(chunk.content, end="", flush=True)
```

### Async streaming

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5")

async for chunk in llm.astream("Explain recursion"):
    print(chunk.content, end="")
```

### Stream an agent's progress

If you're using an agent, you can stream step-by-step updates:

```python
for update in agent.stream(
    {"messages": [{"role": "user", "content": "What's the weather in SF?"}]},
    stream_mode="updates",
):
    print(update)
```

Common stream modes include:

| Mode       | What you receive                          |
| ---------- | ----------------------------------------- |
| `messages` | LLM tokens as they are generated          |
| `updates`  | Agent/graph state updates after each step |
| `custom`   | Custom events emitted by your code        |

([LangChain Docs][1])

### Stream tokens from an agent

```python
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "Tell me a joke"}]},
    stream_mode="messages",
):
    if token.content:
        print(token.content, end="")
```

This is the mode typically used to implement a ChatGPT-style typing effect. ([LangChain Docs][2])

### Stream detailed execution events

For maximum visibility into what a chain, agent, or runnable is doing:

```python
async for event in chain.astream_events(
    {"input": "Hello"},
    version="v2",
):
    print(event)
```

This emits start, stream, and end events for each runnable in the workflow. ([LangChain][3])

Documentation:

* [LangChain Streaming Guide](https://docs.langchain.com/oss/python/langchain/streaming?utm_source=chatgpt.com)
* [LangGraph Streaming Guide](https://docs.langchain.com/oss/python/langgraph/streaming?utm_source=chatgpt.com)

If you're building a FastAPI, Flask, Streamlit, or Next.js app, I can also show how to connect LangChain streaming to a frontend via SSE or WebSockets.

[1]: https://docs.langchain.com/oss/python/langchain/streaming?utm_source=chatgpt.com "Streaming - Docs by LangChain"
[2]: https://docs.langchain.com/oss/javascript/langgraph/streaming?utm_source=chatgpt.com "Streaming - Docs by LangChain"
[3]: https://reference.langchain.com/python/langchain-core/runnables/base/Runnable/astream_events?utm_source=chatgpt.com "astream_events | langchain_core"

## que - 8. How do I use ChatOpenAI in LangChain?
### ans : 
`ChatOpenAI` is LangChain's wrapper around OpenAI chat models. It lets you invoke models, use message-based conversations, stream responses, call tools, and integrate with chains and agents. It is provided by the `langchain-openai` package. ([LangChain Docs][1])

### 1. Install dependencies

```bash
pip install -U langchain-openai
```

The OpenAI integration is distributed separately from the core LangChain package. ([LangChain Docs][1])

### 2. Set your API key

```python
import os

os.environ["OPENAI_API_KEY"] = "your-api-key"
```

Or set it in your shell:

```bash
export OPENAI_API_KEY="your-api-key"
```

([LangChain Docs][1])

### 3. Create a ChatOpenAI model

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-5",
    temperature=0
)
```

Common parameters include:

* `model`
* `temperature`
* `max_completion_tokens`
* `timeout`
* `max_retries`

([LangChain Docs][1])

### 4. Simple invocation

```python
response = llm.invoke("What is LangChain?")

print(response.content)
```

`invoke()` returns an `AIMessage` object, and the generated text is available in `.content`. ([Medium][2])

### 5. Use message-based conversations

```python
from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

response = llm.invoke([
    SystemMessage(content="You are a helpful Python tutor."),
    HumanMessage(content="Explain decorators.")
])

print(response.content)
```

`ChatOpenAI` is designed for chat-style interactions where inputs are sequences of messages rather than plain strings. ([Stack Overflow][3])

### 6. Use with a prompt template

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

chain = prompt | llm

result = chain.invoke({
    "question": "What is dependency injection?"
})

print(result.content)
```

This is the most common LangChain pattern: Prompt → Model → Output. ([GitHub][4])

### 7. Parse text output

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "question": "What is polymorphism?"
})

print(result)
```

The parser converts the `AIMessage` into a plain string. ([Stack Overflow][5])

### 8. Stream responses

```python
for chunk in llm.stream("Write a short poem about AI"):
    print(chunk.content, end="", flush=True)
```

`ChatOpenAI` supports token-level streaming for real-time output. ([LangChain Docs][1])

### 9. Async usage

```python
response = await llm.ainvoke(
    "Explain async programming."
)

print(response.content)
```

Native async support is built in. ([LangChain Docs][1])

### Complete Example

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    model="gpt-5",
    temperature=0
)

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple terms."
)

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "topic": "vector databases"
})

print(result)
```

This pattern—`PromptTemplate → ChatOpenAI → OutputParser`—is the foundation for most LangChain applications. ([LangChain Docs][1])

[1]: https://docs.langchain.com/oss/python/integrations/chat/openai?utm_source=chatgpt.com "ChatOpenAI integration - Docs by LangChain"
[2]: https://medium.com/%40mustafa_akca/understanding-and-using-langchains-chatopenai-simplified-ai-response-handling-e6b17dd99d13?utm_source=chatgpt.com "Understanding and Using LangChain's ChatOpenAI"
[3]: https://stackoverflow.com/questions/76950609/what-is-the-difference-between-openai-and-chatopenai-in-langchain?utm_source=chatgpt.com "What is the difference between OpenAI and ChatOpenAI in ..."
[4]: https://github.com/langchain-ai/langchain?utm_source=chatgpt.com "langchain-ai/langchain: The agent engineering platform."
[5]: https://stackoverflow.com/questions/77782167/modulenotfounderror-no-module-named-langchain-openai?utm_source=chatgpt.com "No module named 'langchain_openai'"

## que - 9. How do I use Chroma as a vector store?
### ans :
Chroma (often called **ChromaDB**) is a vector database commonly used in Retrieval-Augmented Generation (RAG) applications. It stores embeddings (vector representations of text) and lets you perform similarity searches against them.

## 1. Install Chroma

```bash
pip install chromadb
```

If you're generating embeddings locally:

```bash
pip install sentence-transformers
```

---

## 2. Create a Chroma Collection

A collection is similar to a table in a relational database.

```python
import chromadb

client = chromadb.Client()

collection = client.create_collection(
    name="documents"
)
```

---

## 3. Add Documents

You can either provide embeddings yourself or let Chroma generate them via an embedding function.

### Option A: Supply Embeddings Manually

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "Chroma is a vector database.",
    "Vector databases power RAG systems.",
    "Python is a programming language."
]

embeddings = model.encode(docs).tolist()

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=["doc1", "doc2", "doc3"]
)
```

---

## 4. Query the Collection

```python
query = "What database is used for RAG?"

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print(results["documents"])
```

Example output:

```python
[['Vector databases power RAG systems.',
  'Chroma is a vector database.']]
```

---

## 5. Persist Data to Disk

Without persistence, data disappears when the process exits.

```python
import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    "documents"
)
```

Now your vectors survive restarts.

---

## 6. Use Chroma with LangChain

Install:

```bash
pip install langchain langchain-chroma
```

Example:

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

vectorstore = Chroma(
    collection_name="my_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
```

Add documents:

```python
vectorstore.add_texts([
    "Chroma is a vector store.",
    "LangChain integrates with Chroma."
])
```

Search:

```python
docs = vectorstore.similarity_search(
    "What is Chroma?",
    k=2
)

for doc in docs:
    print(doc.page_content)
```

---

## 7. Typical RAG Workflow

```text
Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
Chroma Vector Store
    ↓
Similarity Search
    ↓
Relevant Chunks
    ↓
LLM Prompt
    ↓
Answer
```

A minimal ingestion pipeline looks like:

```python
chunks = split_documents(raw_documents)

embeddings = embed_model.encode(chunks)

collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
```

Then at query time:

```python
query_embedding = embed_model.encode(question)

matches = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

context = "\n".join(matches["documents"][0])
```

Pass `context` to your LLM prompt.

---

## Advanced Features

Chroma also supports:

* Metadata filtering
* Deleting documents
* Updating documents
* Multi-tenant collections
* Cloud-hosted deployments
* Hybrid retrieval architectures

Example metadata filtering:

```python
collection.add(
    documents=["Machine learning guide"],
    metadatas=[{"category": "ai"}],
    ids=["ml1"]
)

results = collection.query(
    query_embeddings=[query_embedding],
    where={"category": "ai"}
)
```

If you're building a RAG application with OpenAI, LangChain, LlamaIndex, or a custom pipeline, I can show a complete end-to-end example using your preferred framework.

 

## que- 10. What is RunnablePassthrough? 
### ans :
`RunnablePassthrough` is a utility in **[LangChain](https://www.langchain.com?utm_source=chatgpt.com)** that passes its input through unchanged. It's most useful when building chains where you want to preserve the original input while also sending it through other processing steps.

Think of it as an identity function:

```python
RunnablePassthrough()(x) == x
```

## Why use it?

In RAG pipelines, you often need:

* The user's original question
* Retrieved context from a vector store

Both need to be available when constructing the final prompt.

Without `RunnablePassthrough`, the original question might get overwritten by intermediate steps.

---

## Basic Example

```python
from langchain_core.runnables import RunnablePassthrough

passthrough = RunnablePassthrough()

result = passthrough.invoke("Hello")
print(result)
```

Output:

```text
Hello
```

---

## Common RAG Pattern

Suppose you have:

```python
retriever = vectorstore.as_retriever()
```

You want:

```python
{
    "context": retriever(question),
    "question": question
}
```

This is where `RunnablePassthrough` shines:

```python
from langchain_core.runnables import RunnablePassthrough

chain = {
    "context": retriever,
    "question": RunnablePassthrough()
}
```

If the input is:

```python
"What is Chroma?"
```

the chain produces something like:

```python
{
    "context": [
        "Chroma is a vector database...",
        "It stores embeddings..."
    ],
    "question": "What is Chroma?"
}
```

---

## Full RAG Example

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

Flow:

```text
User Question
      │
      ▼
RunnablePassthrough ─────► question
      │
      ▼
Retriever ───────────────► context
      │
      ▼
Prompt Template
      │
      ▼
LLM
      │
      ▼
Answer
```

---

## Using `.assign()`

`RunnablePassthrough` can also add new fields while preserving existing ones.

```python
from langchain_core.runnables import RunnablePassthrough

chain = RunnablePassthrough.assign(
    length=lambda x: len(x["text"])
)

result = chain.invoke({
    "text": "Hello World"
})

print(result)
```

Output:

```python
{
    "text": "Hello World",
    "length": 11
}
```

The original data remains intact while additional values are computed.

---

## Interview Answer

If asked *"What is RunnablePassthrough?"*:

> `RunnablePassthrough` is a LangChain runnable that returns its input unchanged. It is commonly used in LCEL chains to preserve the original input while other components (such as retrievers or transformers) generate additional values. This is especially useful in RAG pipelines where both the user's query and retrieved context must be passed to the prompt.
