"""
tools.py — DocBrain Custom Tools
Houses the @tool definitions for the LangGraph Agent.
"""

from langchain_core.tools import tool
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
from loguru import logger

@tool
def web_search_langchain(query: str) -> str:
    """
    Search the live internet for LangChain documentation, concepts, and issues.
    Use this tool ONLY if the provided local documentation context does not contain the answer, 
    or if you suspect the information is missing or out of date.
    """
    logger.info(f"    [TOOL: web_search] Query: {query}")
    try:
        # Simple search is less likely to be blocked
        results = DDGS().text(f"langchain {query}", max_results=4)
        
        if not results:
            return "No web search results found."
            
        formatted_results = []
        for i, r in enumerate(results, 1):
            title = r.get('title', 'Unknown Title')
            href = r.get('href', 'Unknown URL')
            body = r.get('body', 'No snippet')
            formatted_results.append(f"Result {i}:\nTitle: {title}\nURL: {href}\nSnippet: {body}\n")
            
        formatted_results.append("IMPORTANT: Web search only provides short snippets. If you need full code examples, call the `scrape_url` tool on one of the URLs above.")
        return "\n---\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"    [TOOL: web_search] Error: {e}")
        return f"Web search failed: {str(e)}"

@tool
def scrape_url(url: str) -> str:
    """
    Scrape a specific URL to extract its full text content.
    Use this tool after running `web_search_langchain` to read the full content of a promising search result,
    especially if you need code examples or detailed explanations that aren't in the search snippets.
    """
    logger.info(f"    [TOOL: scrape_url] Scraping: {url}")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
            
        text = soup.get_text(separator='\n')
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\\n'.join(chunk for chunk in chunks if chunk)
        
        # Limit to first 10000 characters to avoid blowing up context window
        if len(text) > 10000:
            text = text[:10000] + "... [TRUNCATED]"
            
        return text
        
    except Exception as e:
        logger.error(f"    [TOOL: scrape_url] Error: {e}")
        return f"Failed to scrape URL: {str(e)}"
