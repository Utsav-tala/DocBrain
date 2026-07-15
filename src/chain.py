"""
chain.py — DocBrain v4
Upgraded to a LangGraph ReAct Agent with Tool access.
"""

import os
import re
from dotenv import load_dotenv
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from langgraph.prebuilt import create_react_agent

from src.retriever import load_vectorstore, retrieve, convert_source_to_url
from src.prompt_templates import get_prompt, QUERY_REWRITE_PROMPT
from src.tools import web_search_langchain, scrape_url
from src.link_resolver import resolve_links
from src.output_refiner import refine

load_dotenv()

# ── In-memory session store ───────────────────────────────────────────────────
_history_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _history_store:
        _history_store[session_id] = InMemoryChatMessageHistory()
    return _history_store[session_id]

def clear_session_history(session_id: str) -> None:
    if session_id in _history_store:
        del _history_store[session_id]
        logger.info(f"Cleared history for session: {session_id}")

# ── Web Result Extractor ──────────────────────────────────────────────────────
# Pulls URLs the agent actually used (via web_search_langchain / scrape_url tool
# calls) out of the agent's message trace. These feed link_resolver as
# pre-verified "the agent really looked at this page" candidates — distinct
# from local doc URLs and from anything the LLM might write into prose unprompted.
_RESULT_URL_RE = re.compile(r"URL:\s*(https?://\S+)")
_RESULT_TITLE_RE = re.compile(r"Title:\s*(.+)")

def extract_web_results(agent_messages: list) -> list:
    results = []
    for m in agent_messages:
        if not isinstance(m, ToolMessage):
            continue
        content = m.content if isinstance(m.content, str) else str(m.content)

        titles = _RESULT_TITLE_RE.findall(content)
        urls   = _RESULT_URL_RE.findall(content)
        for title, url in zip(titles, urls):
            results.append({"title": title.strip(), "url": url.strip()})

    return results

# ── Context Formatter ─────────────────────────────────────────────────────────
def format_context(docs: list) -> str:
    formatted = []
    for doc in docs:
        source_path    = doc.metadata.get("source", "")
        live_url       = convert_source_to_url(source_path)
        category_label = doc.metadata.get("category_label", doc.metadata.get("doc_type", "unknown"))
        framework      = doc.metadata.get("framework", "")
        topic          = doc.metadata.get("topic", "")
        title          = doc.metadata.get("title", "")
        priority       = doc.metadata.get("priority", "")
        rank           = doc.metadata.get("rerank_rank", "?")
        score          = doc.metadata.get("rerank_score", "?")

        chunk_text = (
            f"--- [Rank {rank} | Score {score}] ---\n"
            f"Category : {category_label}\n"
            f"Framework: {framework}\n"
            f"Topic    : {topic}\n"
            f"Title    : {title}\n"
            f"Priority : {priority}\n"
            f"Source   : {live_url}\n"
            f"Content  :\n{doc.page_content}\n"
        )
        formatted.append(chunk_text)

    return "\n".join(formatted)

# ── Coverage-gap directive ────────────────────────────────────────────────────
def flag_low_coverage(context: str, rule: dict) -> str:
    """
    Prepend a measurement-driven web-search directive when retrieval signalled a
    coverage gap (top-1 distance >= retriever.GAP_DISTANCE_THRESHOLD).

    This turns "should I fall back to the web?" from the agent's judgment of the
    context into a calibrated number (see retriever.py). The local chunks are still
    passed through unchanged: the threshold favours recall on gaps, so ~1/3 of
    covered questions also trip it, and dropping their context would hurt more than
    the occasional extra search. On a real gap the agent gets an explicit push to
    the web_search tool; on a false alarm it just double-checks against the web.
    """
    if not rule.get("is_coverage_gap"):
        return context
    dist   = rule.get("top1_distance")
    dist_s = f" (top-1 distance {dist:.2f})" if dist is not None else ""
    banner = (
        f"⚠️ LOW LOCAL COVERAGE{dist_s} — retrieval confidence is low for this query. "
        "The local context below may not cover the question. Call the "
        "`web_search_langchain` tool to find or confirm the answer before responding; "
        "do not answer from weak context alone.\n\n"
    )
    return banner + context

# ── Query Rewriter ────────────────────────────────────────────────────────────
def rewrite_query(raw_query: str, history_messages: list, llm: ChatOpenAI) -> str:
    try:
        rewrite_chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
        rewritten = rewrite_chain.invoke({
            "question": raw_query,
            "history" : history_messages,
        })
        rewritten = rewritten.strip()
        if rewritten and rewritten != raw_query:
            logger.info(f"   Query rewritten: '{raw_query}' → '{rewritten}'")
        return rewritten if rewritten else raw_query
    except Exception as e:
        logger.warning(f"   Query rewrite failed ({e}), using original")
        return raw_query

# ── Main Agent ────────────────────────────────────────────────────────────────
def build_rag_chain(model_name: str = "gpt-4o-mini"):
    logger.info(f"Building LangGraph Agent | model={model_name}")

    vectorstore = load_vectorstore()

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    tools = [web_search_langchain, scrape_url]
    agent = create_react_agent(llm, tools)

    def invoke(question: str, session_id: str = "default") -> dict:
        logger.info(f"\n=== DocBrain Agent | session={session_id} ===")
        logger.info(f"    Raw query: {question}")

        history_obj      = get_session_history(session_id)
        history_messages = history_obj.messages

        rewritten_query = rewrite_query(question, history_messages, llm)

        docs, rule = retrieve(rewritten_query, vectorstore)
        intent     = rule["intent"]
        logger.info(f"    Intent: {intent} | Chunks: {len(docs)}")

        context = flag_low_coverage(format_context(docs), rule)

        prompt_template = get_prompt(intent, rewritten_query)
        prompt_value = prompt_template.invoke({
            "question": rewritten_query,
            "context" : context,
            "history" : history_messages,
        })

        agent_result = agent.invoke(
            {"messages": prompt_value.to_messages()},
            config={"recursion_limit": 8}
        )

        final_message = agent_result["messages"][-1]
        answer = final_message.content

        web_results = extract_web_results(agent_result["messages"])
        links = resolve_links(rewritten_query, docs, web_results)

        refiner_report = refine(answer, docs, context, llm, resolved_links=links)
        answer = refiner_report.final_answer

        if not refiner_report.is_grounded:
            logger.warning(f"    Ungrounded claims: {refiner_report.groundedness_issues}")
        if not refiner_report.format_ok:
            logger.warning(f"    Format issues: {refiner_report.format_issues}")

        history_obj.add_message(HumanMessage(content=question))
        history_obj.add_message(AIMessage(content=answer))

        logger.info(f"    Answer generated | intent={intent}")

        return {
            "answer"         : answer,
            "docs"           : docs,
            "intent"         : intent,
            "rewritten_query": rewritten_query,
            "links"          : links,
            "refiner_report" : refiner_report,
        }

    logger.info("Agent ready")
    return invoke

# ── Streaming variant ─────────────────────────────────────────────────────────
def build_streaming_chain(model_name: str = "gpt-4o-mini"):
    logger.info(f"Building streaming LangGraph Agent | model={model_name}")

    vectorstore = load_vectorstore()

    llm_stream = ChatOpenAI(
        model=model_name,
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True,
    )

    llm_rewrite = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    tools = [web_search_langchain, scrape_url]
    agent = create_react_agent(llm_stream, tools)

    def stream(question: str, session_id: str = "default") -> dict:
        logger.info(f"\n=== DocBrain Agent Stream | session={session_id} ===")

        history_obj      = get_session_history(session_id)
        history_messages = history_obj.messages

        rewritten_query = rewrite_query(question, history_messages, llm_rewrite)
        docs, rule      = retrieve(rewritten_query, vectorstore)
        intent          = rule["intent"]
        context         = flag_low_coverage(format_context(docs), rule)
        
        prompt_template = get_prompt(intent, rewritten_query)
        prompt_value = prompt_template.invoke({
            "question": rewritten_query,
            "context" : context,
            "history" : history_messages,
        })

        # Links can be resolved from local docs immediately (no LLM call needed
        # for the local-doc path) — this lets the UI show the button row as soon
        # as the stream finishes, without waiting on a second pass.
        links_holder = {"links": resolve_links(rewritten_query, docs, [])}

        def token_generator():
            full_answer = []
            agent_messages = []

            for msg, metadata in agent.stream(
                {"messages": prompt_value.to_messages()}, 
                stream_mode="messages",
                config={"recursion_limit": 8}
            ):
                agent_messages.append(msg)
                if isinstance(msg, (AIMessage, AIMessageChunk)) and msg.content:
                    # Ignore chunks that are just tool calls without content
                    if not msg.tool_call_chunks and not msg.tool_calls:
                        full_answer.append(msg.content)
                        yield msg.content

            complete = "".join(full_answer)

            # Re-resolve links including any web results the agent used this turn,
            # then strip any unverified inline URLs from the already-streamed text
            # for the version we store in history (the live UI already showed the
            # raw stream — see chain.py integration note re: streaming limitation).
            web_results = extract_web_results(agent_messages)
            if web_results:
                links_holder["links"] = resolve_links(rewritten_query, docs, web_results)

            verified_urls = {l.url for l in links_holder["links"]}
            from src.output_refiner import sanitize_links
            cleaned, stripped = sanitize_links(complete, verified_urls)
            if stripped:
                logger.warning(f"    [stream] Stripped unverified link(s): {stripped}")

            history_obj.add_message(HumanMessage(content=question))
            history_obj.add_message(AIMessage(content=cleaned))
            logger.info(f"    Stream complete | intent={intent}")

        return {
            "stream": token_generator(),
            "docs"  : docs,
            "intent": intent,
            "rewritten_query": rewritten_query,
            "links"          : links_holder,  # dict, mutated post-stream — read AFTER consuming "stream"
        }

    return stream

if __name__ == "__main__":
    chain = build_rag_chain()
    result = chain("What is LCEL?", "test_agent")
    print(result["answer"])
