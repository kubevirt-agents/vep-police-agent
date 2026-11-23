import os
from datetime import UTC, datetime

from langgraph.graph.state import CompiledStateGraph


def get_api_key() -> str:
    """Read and return the API key.
    
    Priority order:
    1. API_KEY environment variable
    2. API_KEY file (current directory or parent)
    """
    # First check environment variable
    api_key = os.environ.get("API_KEY")
    if api_key:
        return api_key.strip()
    
    # Fall back to file reading (for backward compatibility)
    api_key_path = "API_KEY"
    if not os.path.exists(api_key_path):
        api_key_path = "../API_KEY"
    with open(api_key_path, "r") as f:
        return f.read().strip()


def get_google_token() -> str:
    """Read and return the Google token.
    
    The token should be a JSON string containing Google service account credentials.
    
    Priority order:
    1. GOOGLE_TOKEN environment variable
    2. GOOGLE_TOKEN file (current directory or parent)
    """
    # First check environment variable
    token = os.environ.get("GOOGLE_TOKEN")
    if token:
        return token.strip()
    
    # Fall back to file reading (for backward compatibility)
    token_path = "GOOGLE_TOKEN"
    if not os.path.exists(token_path):
        token_path = "../GOOGLE_TOKEN"
    
    try:
        with open(token_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            "GOOGLE_TOKEN not found in environment variable or file. "
            "Please set GOOGLE_TOKEN environment variable or create GOOGLE_TOKEN file "
            "with your Google service account credentials (JSON)."
        )

def invoke_agent(agent: CompiledStateGraph, prompt: str) -> str:
    response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )

    content = response["messages"][-1].content

    # Handle list of content blocks (e.g., [{'type': 'text', 'text': '...'}])
    if isinstance(content, list):
        text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and "text" in item]
        return "".join(text_parts) if text_parts else str(content)

    # Handle string content directly
    return str(content) if content is not None else ""

def get_model(model_name: str | None = None, node: str | None = None):
    """Get the LLM model instance.

    Args:
        model_name: Optional model name. If not provided, uses default from config.
        node: Optional node/operation name. Resolves the tier, which picks the
            thinking level and per-call timeout (config.get_llm_settings_for_node).

    The SDK-side timeout aborts the HTTP call itself; the executor wrapper in
    llm_helper is only a backstop. max_output_tokens caps output (thinking
    tokens count against it) and max_retries is kept low to avoid multiplying
    wall time with the wrapper-level retries.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    import config

    if model_name is None:
        model_name = config.DEFAULT_MODEL
    thinking_level, timeout = config.get_llm_settings_for_node(node)

    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=get_api_key(),
        thinking_level=thinking_level,
        timeout=timeout,
        max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS,
        max_retries=config.LLM_SDK_MAX_RETRIES,
    )


def log(message: str, node: str = "SYSTEM", level: str = "INFO") -> None:
    """Centralized logging function.
    
    Currently uses print, but can be easily switched to proper logging later.
    
    Args:
        message: Log message
        node: Node/component name (e.g., "scheduler", "fetch_data")
        level: Log level (INFO, WARNING, ERROR, DEBUG)
    """
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level:5s}] [{node:15s}] {message}", flush=True)
