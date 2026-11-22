"""Helper functions for creating LLM agents with MCP tools."""

import concurrent.futures
import json
import re
import time
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from config.config import (
    FETCH_INPUT_TOKEN_BUDGET,
    FETCH_INPUT_TOKEN_HARD_CAP,
    FETCH_TOOL_RESULT_MAX_CHARS,
    LLM_INITIAL_DELAY,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_RETRIES,
    NODE_MAX_ITERATIONS,
)
from services.mcp_factory import get_mcp_tools_by_name
from services.utils import get_model, log

T = TypeVar('T', bound=BaseModel)

# Parallel tool execution: every MCP tool call spawns its own server process
# on its own event loop (services/mcp_factory.make_tool_func), so concurrent
# calls share no state - the pool only bounds how many run at once.
TOOL_CALL_MAX_WORKERS = 6

# Floor for a shrunk tool result - keeps the tool_call/ToolMessage pairing
# meaningful while trimming context.
_MIN_TOOL_RESULT_CHARS = 200


# Server-side deadline error: the SDK timeout aborts the HTTP call, which the
# Google API surfaces as "504 DEADLINE_EXCEEDED" rather than a client-side
# TimeoutError. Word-boundary match so PR numbers like 5043 do not hit.
_DEADLINE_ERROR_RE = re.compile(r"\b504\b|DEADLINE_EXCEEDED")


def _is_deadline_exceeded(e: BaseException) -> bool:
    """True for server-side deadline errors (504 / DEADLINE_EXCEEDED)."""
    return bool(_DEADLINE_ERROR_RE.search(str(e)))


def _invoke_with_timeout(llm, messages, timeout_seconds: int):
    """Invoke LLM with a per-call timeout backstop.

    The SDK-side timeout (set in get_model) is the real abort; this wrapper
    only guards against a hung SDK call. On timeout the executor is abandoned
    (shutdown(wait=False, cancel_futures=True)) so a retry starts immediately
    instead of waiting out the hung call.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(llm.invoke, messages)
    try:
        return future.result(timeout=timeout_seconds)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _invoke_with_retry(llm, messages, operation_type: str, timeout_seconds: int):
    """Invoke LLM with per-call timeout and exponential backoff on retriable errors.

    Retriable errors: rate limits (429/RESOURCE_EXHAUSTED, up to LLM_MAX_RETRIES
    retries, with the tier timeout as the total budget) and call timeouts (at
    most LLM_TIMEOUT_RETRIES retries). A timeout counts whether it aborts
    client-side (TimeoutError) or server-side (504 DEADLINE_EXCEEDED).
    Returns the LLM response or raises if retries are exhausted.
    """
    start_time = time.time()
    delay = LLM_INITIAL_DELAY
    timeout_retries = 0
    rate_limit_retries = 0

    def _handle_timeout(reason: str) -> bool:
        """Shared timeout-retry bookkeeping. Returns True to retry the call."""
        nonlocal timeout_retries, delay
        if timeout_retries >= LLM_TIMEOUT_RETRIES:
            log(f"LLM call {reason} after {timeout_seconds}s, timeout retries exhausted "
                f"({LLM_TIMEOUT_RETRIES} retry(ies))",
                node=operation_type, level="ERROR")
            return False
        timeout_retries += 1
        log(f"LLM call {reason} after {timeout_seconds}s, retrying in {delay}s "
            f"(timeout retry {timeout_retries}/{LLM_TIMEOUT_RETRIES})",
                node=operation_type, level="WARNING")
        time.sleep(delay)
        delay = min(delay * 2, 60)
        return True

    while True:
        try:
            return _invoke_with_timeout(llm, messages, timeout_seconds)
        except concurrent.futures.TimeoutError:
            if not _handle_timeout("timed out"):
                raise
        except Exception as e:
            if _is_deadline_exceeded(e):
                # The SDK-side timeout aborts the HTTP call, which the API
                # surfaces as 504 DEADLINE_EXCEEDED instead of a client-side
                # TimeoutError. Same condition, same handling: the single
                # timeout retry, then re-raise.
                if not _handle_timeout("hit a server-side deadline (504)"):
                    raise
                continue
            error_str = str(e)
            # Check if it's a rate limit error
            if "RESOURCE_EXHAUSTED" not in error_str and "429" not in error_str:
                raise  # Not a rate limit error, re-raise immediately

            if rate_limit_retries >= LLM_MAX_RETRIES:
                log(f"Rate limited {rate_limit_retries} time(s), max retries ({LLM_MAX_RETRIES}) exceeded",
                    node=operation_type, level="ERROR")
                raise
            rate_limit_retries += 1

            elapsed = time.time() - start_time
            if elapsed + delay > timeout_seconds:
                log(f"Rate limit retry would exceed max timeout ({timeout_seconds}s), aborting",
                    node=operation_type, level="ERROR")
                raise

            log(f"Rate limited, retrying in {delay}s (attempt {rate_limit_retries}/{LLM_MAX_RETRIES})",
                node=operation_type, level="WARNING")
            time.sleep(delay)
            delay = min(delay * 2, 60)  # Cap at 60s per retry


def _classify_llm_error(e: BaseException) -> str:
    """Classify an LLM call failure for the LLM_CALL_FAILED log tag."""
    if isinstance(e, concurrent.futures.TimeoutError) or _is_deadline_exceeded(e):
        return "timeout"
    error_str = str(e)
    if "INVALID_ARGUMENT" in error_str or " 400" in f" {error_str}":
        return "invalid_argument"
    return "other"


def _truncate_tool_result(result: Any, max_chars: int = FETCH_TOOL_RESULT_MAX_CHARS) -> str:
    """Cap a tool result string, head-biased.

    VEP files keep front matter and the Summary near the top, so keep ~85%
    head / ~15% tail and mark how many chars were cut.
    """
    text = result if isinstance(result, str) else str(result)
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.85)
    tail = max_chars - head
    return f"{text[:head]}\n[TRUNCATED {len(text) - max_chars} CHARS]\n{text[-tail:] if tail > 0 else ''}"


def _execute_tool_calls(tool_calls: list[dict], tools: list, operation_type: str) -> list[ToolMessage]:
    """Execute a turn's tool calls in parallel, preserving order and ids.

    Every call already spawns its own MCP process (see mcp_factory), so this
    is thread-safe. Results are returned in the original tool_call order with
    matching tool_call_ids - Gemini rejects an AIMessage whose tool_calls
    have no ToolMessage. Never skips a call.
    """
    tools_by_name = {tool.name: tool for tool in tools}

    def run_one(tool_call: dict) -> str:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        log(f"Executing tool: {tool_name} with args: {json.dumps(tool_args, default=str)}",
            node=operation_type, level="DEBUG")
        tool = tools_by_name.get(tool_name)
        if tool is None:
            log(f"Tool {tool_name} not found in available tools: {list(tools_by_name)}",
                node=operation_type, level="WARNING")
            return f"Tool {tool_name} not found"
        try:
            result = str(tool.func(**tool_args))
        except Exception as e:
            log(f"Error executing tool {tool_name}: {e}", node=operation_type, level="ERROR")
            import traceback
            log(f"Tool error traceback: {traceback.format_exc()}", node=operation_type, level="DEBUG")
            return f"Error: {e!s}"
        # Log tool result (truncate if too long)
        result_preview = result[:200] + "... (truncated)" if len(result) > 200 else result
        log(f"Tool {tool_name} result: {result_preview}", node=operation_type, level="DEBUG")
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=TOOL_CALL_MAX_WORKERS) as executor:
        results = list(executor.map(run_one, tool_calls))

    return [
        ToolMessage(content=_truncate_tool_result(result), tool_call_id=tool_call.get("id", ""))
        for tool_call, result in zip(tool_calls, results)
    ]


def _estimate_input_tokens(messages: list, response) -> int:
    """Estimate the input tokens of the NEXT LLM call.

    Prefers the exact count from the last AIMessage's usage_metadata; falls
    back to a chars/4 estimate over the conversation so far.
    """
    usage = getattr(response, "usage_metadata", None)
    input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    if isinstance(input_tokens, int):
        return input_tokens
    return sum(len(str(getattr(m, "content", ""))) for m in messages) // 4


def _shrink_tool_messages(
    tool_messages: list[ToolMessage], base_input_tokens: int, operation_type: str,
) -> list[ToolMessage]:
    """Proportionally shrink the newest ToolMessages to fit the hard cap.

    Keeps every AIMessage tool_call paired with its ToolMessage (Gemini
    rejects unmatched tool_calls) - only the result contents get shorter.
    """
    allowed_chars = max(0, FETCH_INPUT_TOKEN_HARD_CAP - base_input_tokens) * 4
    total_chars = sum(len(m.content) for m in tool_messages)
    if not tool_messages or total_chars <= allowed_chars:
        return tool_messages

    scale = allowed_chars / total_chars
    log(f"Shrinking {len(tool_messages)} tool result(s) ({total_chars} chars) to ~{allowed_chars} chars "
        f"to fit the {FETCH_INPUT_TOKEN_HARD_CAP}-token hard cap",
        node=operation_type, level="WARNING")
    shrunk = []
    for m in tool_messages:
        target = max(_MIN_TOOL_RESULT_CHARS, int(len(m.content) * scale))
        content = _truncate_tool_result(m.content, target) if len(m.content) > target else m.content
        shrunk.append(ToolMessage(content=content, tool_call_id=m.tool_call_id))
    return shrunk


def _structured_output_call(
    llm, messages, response_model: type[T], operation_type: str, timeout_seconds: int,
) -> T:
    """Final structured-output call on the tool-free LLM instance.

    The schema is bound on a tool-free LLM because langchain-google-genai
    merges the response schema onto any bound tools, and Gemini rejects
    tools+response_schema in one call (400). include_raw=True keeps the raw
    AIMessage so its usage can be logged (output tokens count against
    LLM_MAX_OUTPUT_TOKENS, thinking included).
    """
    structured_llm = llm.with_structured_output(response_model, include_raw=True)
    raw_result = _invoke_with_retry(structured_llm, messages, operation_type, timeout_seconds)
    result = raw_result.get("parsed")
    parsing_error = raw_result.get("parsing_error")
    if result is None or parsing_error is not None:
        raise ValueError(f"Structured output parsing failed for {operation_type}: {parsing_error}")
    usage = getattr(raw_result.get("raw"), "usage_metadata", None)
    if isinstance(usage, dict) and usage.get("output_tokens") is not None:
        log(f"LLM_CALL_USAGE op={operation_type} input_tokens={usage.get('input_tokens')} "
            f"output_tokens={usage.get('output_tokens')} (output cap {LLM_MAX_OUTPUT_TOKENS})",
            node=operation_type)
    return result


class NoToolsCalledException(Exception):
    """Raised when LLM completes without calling any tools but tools were required."""


class ToolCallFailedException(Exception):
    """Raised when tools were required, at least one tool call was attempted,
    but every attempted tool call failed (e.g. the MCP server returned an error
    for each call). Without this, a run where every write errored would still
    be reported as a success by the LLM's self-reported structured output."""


def _is_tool_call_error(result: Any) -> bool:
    """Conservatively detect whether a tool call's result string indicates failure.

    Only matches clear, unambiguous error markers - as opposed to any output that
    merely happens to mention the word "error" - to avoid false positives on
    legitimate tool output.
    """
    if not isinstance(result, str):
        return False
    return result.startswith(("Error", "error")) or "MCP error" in result


def invoke_llm_with_tools(
    operation_type: str,
    state_context: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    mcp_names: tuple = ("github",),
    require_tools: bool = False
) -> T:
    """Invoke LLM with MCP tools using structured output.

    Args:
        operation_type: Type of operation (for logging and model/tier selection)
        state_context: Current state context
        system_prompt: System prompt describing the task
        user_prompt: User prompt with specific instructions
        response_model: Pydantic model for structured output
        mcp_names: Tuple of MCP server names to load tools from (default: ("github",)).
            An empty tuple means a tool-free call: no MCP servers are spawned
            and the LLM answers directly with structured output.
        require_tools: If True, raises NoToolsCalledException if LLM doesn't call any tools

    Returns:
        Validated Pydantic model instance

    Raises:
        NoToolsCalledException: If require_tools=True and no tools were called
        ToolCallFailedException: If require_tools=True, at least one tool call
            was attempted, and every attempted tool call failed
    """
    try:
        # Get model for this operation type (node); the node name also picks
        # the tier (thinking level, per-call timeout).
        import config
        model_name = config.get_model_for_node(operation_type)
        log(f"Using model {model_name} for {operation_type}", node=operation_type, level="DEBUG")
        _, timeout_seconds = config.get_llm_settings_for_node(operation_type)

        if not mcp_names:
            # Tool-free path: no MCP servers spawned, no tool loop - the LLM
            # answers directly with structured output.
            log(f"Tool-free call for {operation_type} (no MCP servers requested)", node=operation_type)
            llm = get_model(model_name=model_name, node=operation_type)
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            return _structured_output_call(llm, messages, response_model, operation_type, timeout_seconds)

        # Get MCP tools
        tools = get_mcp_tools_by_name(*mcp_names)
        mcp_list = ", ".join(mcp_names)
        tool_names = [t.name for t in tools] if tools else []
        log(f"Loaded {len(tools)} MCP tools ({mcp_list}) for {operation_type}: {tool_names}", node=operation_type)

        if not tools:
            log(f"No MCP tools available for {operation_type}", node=operation_type, level="ERROR")
            # Return empty response with proper structure
            try:
                return response_model()
            except Exception as e:
                # If model requires fields, try with empty defaults
                log(f"Could not create empty {response_model.__name__}: {e}", node=operation_type, level="WARNING")
                return response_model()

        llm = get_model(model_name=model_name, node=operation_type)

        # Create LLM with tools bound
        # Use tool_choice="any" when require_tools is True to force LLM to call at least one tool
        if require_tools:
            llm_with_tools = llm.bind_tools(tools, tool_choice="any")
            log("Tool choice set to 'any' - forcing LLM to use tools", node=operation_type, level="DEBUG")
        else:
            llm_with_tools = llm.bind_tools(tools)

        # Build messages
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        # First, handle tool calls (if any) - do this without structured output
        # Check for debug mode that limits iterations
        import os
        debug_mode = os.environ.get("DEBUG_MODE")
        if debug_mode == "test-sheets":
            # Increase iterations for test-sheets to allow LLM to complete write operations
            max_iterations = 5
            log(f"Debug mode 'test-sheets' enabled - limiting to {max_iterations} iteration(s)", node=operation_type, level="INFO")
        else:
            # fetch_veps may need to read many issue details; other nodes less
            default_iterations = 30 if operation_type == "fetch_veps" else 10
            max_iterations = NODE_MAX_ITERATIONS.get(operation_type, default_iterations)
        iteration = 0
        tool_call_counts = {}  # Track tool calls for summary
        tool_call_failures = 0  # Track how many attempted tool calls errored
        while iteration < max_iterations:
            iteration += 1
            log(f"Invoking LLM for {operation_type} (iteration {iteration}/{max_iterations})...", node=operation_type)
            response = _invoke_with_retry(llm_with_tools, messages, operation_type, timeout_seconds)
            log(f"LLM invocation completed for {operation_type} (iteration {iteration})", node=operation_type, level="DEBUG")

            # Check if response has tool calls
            if not (hasattr(response, 'tool_calls') and response.tool_calls):
                # No more tool calls, break and get structured output
                # Debug: log what the response looks like
                log(f"Response type: {type(response).__name__}, has tool_calls attr: {hasattr(response, 'tool_calls')}", node=operation_type, level="DEBUG")
                if hasattr(response, 'content'):
                    content_preview = str(response.content)[:200] if response.content else "(empty)"
                    log(f"Response content preview: {content_preview}", node=operation_type, level="DEBUG")
                break

            log(f"LLM made {len(response.tool_calls)} tool call(s), iteration {iteration}", node=operation_type)

            # Track tool calls for summary
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

            # Execute this turn's tool calls in parallel and append the
            # results in the original tool_call order.
            tool_messages = _execute_tool_calls(response.tool_calls, tools, operation_type)
            tool_call_failures += sum(1 for m in tool_messages if _is_tool_call_error(m.content))

            # Context budget check AFTER the tools ran, so every tool_call
            # already has its ToolMessage - an AIMessage with unanswered
            # tool_calls at the end of history would 400 the final call.
            base_input_tokens = _estimate_input_tokens(messages, response)
            added_chars = sum(len(m.content) for m in tool_messages)
            messages.append(response)
            messages.extend(tool_messages)
            projected = base_input_tokens + added_chars // 4

            if projected > FETCH_INPUT_TOKEN_HARD_CAP:
                tool_messages = _shrink_tool_messages(tool_messages, base_input_tokens, operation_type)
                messages[-len(tool_messages):] = tool_messages
                added_chars = sum(len(m.content) for m in tool_messages)
                projected = base_input_tokens + added_chars // 4

            if projected > FETCH_INPUT_TOKEN_BUDGET:
                vep_note = ""
                if isinstance(state_context, dict) and state_context.get("veps"):
                    vep_note = f", {len(state_context['veps'])} VEP(s) in this fetch's scope"
                total_tool_calls = sum(tool_call_counts.values())
                log(f"CONTEXT_BUDGET_STOP op={operation_type}: projected next input ~{projected} tokens "
                    f"(input_tokens={base_input_tokens}, +{added_chars} chars of new tool results) "
                    f"exceeds budget {FETCH_INPUT_TOKEN_BUDGET} after {total_tool_calls} tool call(s){vep_note}; "
                    "stopping the tool loop and answering with the data gathered so far",
                    node=operation_type, level="WARNING")
                break

        # Now get structured output with final messages (including tool results)
        messages.append(HumanMessage(content="Based on the information gathered, please provide your response in the required structured format."))
        log(f"Requesting structured output for {operation_type}...", node=operation_type, level="DEBUG")
        result = _structured_output_call(llm, messages, response_model, operation_type, timeout_seconds)
        log(f"Structured output received for {operation_type}", node=operation_type, level="DEBUG")

        # Response is already a validated Pydantic model!
        log(f"Successfully received structured response for {operation_type}", node=operation_type)

        # Log tool call summary (helpful for debugging sheets updates, etc.)
        if tool_call_counts:
            summary = ", ".join(f"{name}: {count}" for name, count in sorted(tool_call_counts.items()))
            log(f"Tool calls summary: {summary}", node=operation_type)

            total_tool_calls = sum(tool_call_counts.values())
            if require_tools and total_tool_calls > 0 and tool_call_failures == total_tool_calls:
                log(f"WARNING: All {total_tool_calls} tool call(s) failed for {operation_type} - "
                    "treating as failure instead of reporting false success",
                    node=operation_type, level="WARNING")
                raise ToolCallFailedException(f"All {total_tool_calls} tool call(s) failed for {operation_type}")
        else:
            # No tools were called - this is often a sign of LLM hallucination
            log(f"WARNING: LLM completed {operation_type} without calling any tools!", node=operation_type, level="WARNING")
            if require_tools:
                raise NoToolsCalledException(f"LLM completed {operation_type} without calling any tools")

        return result

    except (NoToolsCalledException, ToolCallFailedException):
        # These signal a specific, actionable failure mode to the caller (e.g.
        # update_sheets.py retries the update) - let them propagate instead of
        # being swallowed by the generic fallback below.
        raise
    except Exception as e:
        # Loud failure tag: 400 INVALID_ARGUMENT usually means context length
        # or schema problems - make those trivially greppable before falling back.
        log(f"LLM_CALL_FAILED op={operation_type} kind={_classify_llm_error(e)}: {e}",
            node=operation_type, level="ERROR")
        import traceback
        log(f"Traceback: {traceback.format_exc()}", node=operation_type, level="ERROR")
        # Return empty response with proper structure
        # Use Pydantic model introspection to provide defaults for required fields
        try:
            # Try to create with empty dict first (works if all fields have defaults)
            return response_model()
        except (TypeError, ValueError, AttributeError):
            # Build defaults from model fields
            defaults = {}
            try:
                # Use Pydantic v2 model_fields if available
                if hasattr(response_model, 'model_fields'):
                    for field_name, field_info in response_model.model_fields.items():
                        if field_info.is_required():
                            # Provide sensible defaults based on field type
                            field_type = str(field_info.annotation) if hasattr(field_info, 'annotation') else ''
                            if 'list' in field_type.lower() or field_name in ['updated_veps', 'alerts']:
                                defaults[field_name] = []
                            elif field_name == 'success':
                                defaults[field_name] = False
                            elif 'dict' in field_type.lower():
                                defaults[field_name] = {}
                            elif 'Optional' in field_type or field_name.endswith('_id'):
                                defaults[field_name] = None
                            else:
                                defaults[field_name] = None
                else:
                    # Fallback for Pydantic v1 or models without model_fields
                    # Check common field names
                    if hasattr(response_model, '__annotations__'):
                        annotations = response_model.__annotations__
                        for field_name in annotations:
                            if field_name in ['updated_veps', 'alerts']:
                                defaults[field_name] = []
                            elif field_name == 'success':
                                defaults[field_name] = False
                            else:
                                defaults[field_name] = None

                return response_model(**defaults)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as final_error:
                log(f"Could not create {response_model.__name__} with defaults: {final_error}", node=operation_type, level="ERROR")
                # Last resort: try with minimal known defaults
                minimal_defaults = {
                    'updated_veps': [],
                    'alerts': [],
                    'success': False,
                }
                try:
                    return response_model(**{k: v for k, v in minimal_defaults.items() if hasattr(response_model, k)})
                except (TypeError, ValueError, AttributeError):
                    # This will fail but at least we tried everything
                    raise ValueError(f"Could not create {response_model.__name__} with defaults. Error: {final_error}")


def invoke_llm_check(
    check_type: str,
    state_context: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    use_tools: bool = True
) -> T:
    """Invoke LLM to perform a check with GitHub MCP tools using structured output.

    Convenience wrapper around invoke_llm_with_tools for check nodes.

    Args:
        check_type: Type of check ("deadlines", "activity", "compliance", "exceptions")
        state_context: Current state context (veps, release_schedule, etc.)
        system_prompt: System prompt describing the task
        user_prompt: User prompt with specific instructions
        response_model: Pydantic model for structured output
        use_tools: If False, make a direct structured-output call with no
            GitHub tools bound and no MCP server spawned.

    Returns:
        Validated Pydantic model instance
    """
    return invoke_llm_with_tools(
        check_type, state_context, system_prompt, user_prompt, response_model,
        mcp_names=("github",) if use_tools else (),
    )


def invoke_llm_fetch(
    fetch_type: str,
    state_context: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    response_model: type[T]
) -> T:
    """Invoke lightweight LLM to fetch context data with GitHub MCP tools.

    Used by fetch nodes (check_deadlines, check_activity, check_compliance, check_exceptions)
    to gather raw data without analysis. The LLM uses GitHub MCP tools to fetch data and
    returns it in a structured format for later analysis by analyze_combined.

    Args:
        fetch_type: Type of fetch ("deadlines", "activity", "compliance", "exceptions")
        state_context: Current state context (veps, release_schedule, etc.)
        system_prompt: System prompt describing what data to fetch
        user_prompt: User prompt with specific instructions
        response_model: Pydantic model for structured output (typically FetchResponse)

    Returns:
        Validated Pydantic model instance with context data
    """
    return invoke_llm_with_tools(fetch_type, state_context, system_prompt, user_prompt, response_model, mcp_names=("github",))
