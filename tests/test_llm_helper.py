"""Tests for invoke_llm_with_tools and its helpers.

Covers: the except-path default builder (regression: required list fields
got None instead of []), tool-result truncation, the fetch-loop context
budget (stop leaves no dangling tool_calls; over-hard-cap results shrink
while keeping the tool_call/ToolMessage pairing), parallel tool execution
order/ids, the timeout retry count, the tool-free path for empty mcp_names,
per-node iteration overrides, and the LLM_CALL_FAILED failure tag.
"""

import concurrent.futures
import time
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from services.llm_helper import (
    ToolCallFailedException,
    _execute_tool_calls,
    _invoke_with_retry,
    _is_tool_call_error,
    _shrink_tool_messages,
    _truncate_tool_result,
    invoke_llm_with_tools,
)
from services.response_models import FetchResponse


def test_exception_path_defaults_required_list_field_to_empty_list():
    with patch("services.llm_helper.get_mcp_tools_by_name", side_effect=RuntimeError("boom")):
        result = invoke_llm_with_tools(
            "check_deadlines", {}, "system prompt", "user prompt", FetchResponse
        )

    assert result.context_updates == []


def test_is_tool_call_error_detects_unambiguous_error_markers():
    assert _is_tool_call_error("Error calling tool write_range: boom")
    assert _is_tool_call_error("error: something bad happened")
    assert _is_tool_call_error("Something went wrong. MCP error -32603: bad params")


def test_is_tool_call_error_does_not_flag_normal_output():
    assert not _is_tool_call_error("Wrote 5 rows successfully")
    assert not _is_tool_call_error("No errors found in the sheet")
    assert not _is_tool_call_error(123)


def _fake_tool(name, func):
    tool = MagicMock()
    tool.name = name
    tool.func = func
    return tool


def _fake_llm(responses, structured=None, usage=None):
    """Build a fake llm where .invoke() returns each response in order.

    .bind_tools() returns the same llm; .with_structured_output() returns a
    stand-in whose invoke() returns an include_raw-style dict
    ({"raw", "parsed", "parsing_error"}).
    """
    llm = MagicMock()
    responses_iter = iter(responses)
    llm.invoke.side_effect = lambda messages: next(responses_iter)
    llm.bind_tools.return_value = llm
    raw_message = MagicMock(usage_metadata=usage or {"input_tokens": 100, "output_tokens": 50})
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = {
        "raw": raw_message,
        "parsed": structured if structured is not None else MagicMock(),
        "parsing_error": None,
    }
    llm.with_structured_output.return_value = structured_llm
    return llm


def test_all_tool_calls_failing_raises_tool_call_failed_exception():
    """Bug B: every attempted sheets write errors - this must be a failure,
    not a silently-reported success."""
    failing_tool = _fake_tool(
        "write_range",
        lambda **kwargs: "Error calling tool write_range: MCP error -32603: Missing required parameters",
    )
    llm = _fake_llm([
        MagicMock(tool_calls=[{"name": "write_range", "args": {"spreadsheetId": "x"}, "id": "1"}], content=""),
        MagicMock(tool_calls=[], content="done"),
    ])

    with patch("services.llm_helper.get_mcp_tools_by_name", return_value=[failing_tool]), \
         patch("services.llm_helper.get_model", return_value=llm), \
         patch("config.get_model_for_node", return_value="fake-model"), \
         patch("config.get_llm_settings_for_node", return_value=(None, 300)), \
         pytest.raises(ToolCallFailedException):
        invoke_llm_with_tools(
            "update_sheets", {}, "system prompt", "user prompt", FetchResponse,
            mcp_names=("google-sheets",), require_tools=True,
        )


def test_successful_tool_call_does_not_raise():
    succeeding_tool = _fake_tool("write_range", lambda **kwargs: "Wrote 5 rows successfully")
    llm = _fake_llm([
        MagicMock(tool_calls=[{"name": "write_range", "args": {"spreadsheetId": "x"}, "id": "1"}], content=""),
        MagicMock(tool_calls=[], content="done"),
    ])

    with patch("services.llm_helper.get_mcp_tools_by_name", return_value=[succeeding_tool]), \
         patch("services.llm_helper.get_model", return_value=llm), \
         patch("config.get_model_for_node", return_value="fake-model"), \
         patch("config.get_llm_settings_for_node", return_value=(None, 300)):
        # Should not raise - at least one tool call succeeded.
        invoke_llm_with_tools(
            "update_sheets", {}, "system prompt", "user prompt", FetchResponse,
            mcp_names=("google-sheets",), require_tools=True,
        )


# ---- Tool-result truncation ----


def test_truncate_tool_result_is_head_biased_with_marker():
    long_result = "H" * 15000 + "T" * 15000  # 30000 chars
    truncated = _truncate_tool_result(long_result, max_chars=10000)

    assert "[TRUNCATED 20000 CHARS]" in truncated
    # ~85% head, ~15% tail, split around the marker
    head, _, tail = truncated.partition("\n[TRUNCATED 20000 CHARS]\n")
    assert head == "H" * 8500
    assert tail == "T" * 1500
    assert len(truncated) < len(long_result)


def test_truncate_tool_result_noop_at_or_under_limit():
    assert _truncate_tool_result("short", max_chars=100) == "short"
    # Non-string results are coerced, not truncated.
    assert _truncate_tool_result(12345, max_chars=100) == "12345"


# ---- Parallel tool execution ----


def test_parallel_tool_calls_preserve_order_and_ids():
    """Tool calls run concurrently but results keep tool_call order and ids."""
    def slow_a(**kwargs):
        time.sleep(0.3)
        return "result-a"

    def slow_b(**kwargs):
        time.sleep(0.3)
        return "result-b"

    tool_a = _fake_tool("tool_a", slow_a)
    tool_b = _fake_tool("tool_b", slow_b)
    tool_calls = [
        {"name": "tool_a", "args": {}, "id": "call-1"},
        {"name": "tool_b", "args": {}, "id": "call-2"},
    ]

    start = time.monotonic()
    messages = _execute_tool_calls(tool_calls, [tool_a, tool_b], "test_op")
    elapsed = time.monotonic() - start

    assert [m.tool_call_id for m in messages] == ["call-1", "call-2"]
    assert [m.content for m in messages] == ["result-a", "result-b"]
    # Both calls slept 0.3s; sequential execution would take ~0.6s.
    assert elapsed < 0.55


def test_parallel_tool_calls_never_skip_failed_calls():
    def boom(**kwargs):
        raise RuntimeError("MCP died")

    tool = _fake_tool("tool_a", boom)
    tool_calls = [{"name": "tool_a", "args": {}, "id": "call-1"}]

    messages = _execute_tool_calls(tool_calls, [tool], "test_op")

    assert len(messages) == 1
    assert messages[0].tool_call_id == "call-1"
    assert messages[0].content.startswith("Error:")


# ---- Context budget ----


def _budget_patches(llm, tools):
    return [
        patch("services.llm_helper.get_mcp_tools_by_name", return_value=tools),
        patch("services.llm_helper.get_model", return_value=llm),
        patch("config.get_model_for_node", return_value="fake-model"),
        patch("config.get_llm_settings_for_node", return_value=(None, 300)),
    ]


def test_budget_stop_leaves_no_dangling_tool_calls(capsys):
    """Past the input token budget the tool loop stops after exactly the
    calls already made, with every tool_call answered by a ToolMessage."""
    big_result = "x" * 100000  # truncated to 20000 chars -> ~5000 tokens
    tool = _fake_tool("get_file_contents", lambda **kwargs: big_result)
    response_with_calls = MagicMock(
        tool_calls=[{"name": "get_file_contents", "args": {}, "id": "call-1"}],
        content="",
        usage_metadata={"input_tokens": 700000},
    )
    llm = _fake_llm([response_with_calls])

    patches = _budget_patches(llm, [tool])
    with patches[0], patches[1], patches[2], patches[3]:
        invoke_llm_with_tools(
            "check_compliance", {"veps": ["v1", "v2", "v3"]},
            "system prompt", "user prompt", FetchResponse,
        )

    # The loop stopped instead of iterating again.
    assert llm.invoke.call_count == 1
    # Loud warning with the numbers.
    out = capsys.readouterr().out
    assert "CONTEXT_BUDGET_STOP" in out
    assert "op=check_compliance" in out
    assert "3 VEP(s) in this fetch's scope" in out

    # Final structured call history: ..., AIMessage(tool_calls), ToolMessage,
    # HumanMessage - never an AIMessage with unanswered tool_calls at the end.
    final_messages = llm.with_structured_output.return_value.invoke.call_args[0][0]
    assert isinstance(final_messages[0], SystemMessage)
    tool_msgs = [m for m in final_messages if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_msgs] == ["call-1"]
    assert isinstance(final_messages[-1], HumanMessage)
    assert isinstance(final_messages[-2], ToolMessage)


def test_budget_stop_over_hard_cap_shrinks_results():
    """Projected input past the hard cap shrinks the newest ToolMessages
    proportionally while keeping every tool_call paired."""
    big_result = "y" * 30000
    tool = _fake_tool("get_file_contents", lambda **kwargs: big_result)
    response_with_calls = MagicMock(
        tool_calls=[
            {"name": "get_file_contents", "args": {}, "id": "call-1"},
            {"name": "get_file_contents", "args": {}, "id": "call-2"},
        ],
        content="",
        usage_metadata={"input_tokens": 895000},
    )
    llm = _fake_llm([response_with_calls])

    patches = _budget_patches(llm, [tool])
    with patches[0], patches[1], patches[2], patches[3]:
        invoke_llm_with_tools(
            "check_compliance", {}, "system prompt", "user prompt", FetchResponse,
        )

    final_messages = llm.with_structured_output.return_value.invoke.call_args[0][0]
    tool_msgs = [m for m in final_messages if isinstance(m, ToolMessage)]
    # Pairing kept: both ids answered, in order.
    assert [m.tool_call_id for m in tool_msgs] == ["call-1", "call-2"]
    # 895000 + 40000/4 = 905000 > 900000 hard cap -> results shrunk to fit
    # (allowed ~20000 chars total, both were 20000 after per-result truncation).
    total_chars = sum(len(m.content) for m in tool_msgs)
    assert total_chars < 22000
    assert all(len(m.content) < 20000 for m in tool_msgs)


def test_shrink_tool_messages_keeps_pairing_and_floor():
    messages = [
        ToolMessage(content="a" * 40000, tool_call_id="call-1"),
        ToolMessage(content="b" * 40000, tool_call_id="call-2"),
    ]

    shrunk = _shrink_tool_messages(messages, base_input_tokens=890000, operation_type="test_op")

    assert [m.tool_call_id for m in shrunk] == ["call-1", "call-2"]
    # (900000 - 890000) * 4 = 40000 allowed chars for 80000 total -> halved.
    total = sum(len(m.content) for m in shrunk)
    assert total < 42000
    assert all(len(m.content) >= 200 for m in shrunk)


def test_shrink_tool_messages_noop_when_under_cap():
    messages = [ToolMessage(content="small", tool_call_id="call-1")]
    assert _shrink_tool_messages(messages, base_input_tokens=100, operation_type="test_op") is messages


# ---- Timeout retries ----


def test_timeout_retried_at_most_once():
    llm = MagicMock()
    llm.invoke.side_effect = lambda messages: time.sleep(0.3)

    with patch("services.llm_helper.LLM_INITIAL_DELAY", 0), \
         patch("services.llm_helper.LLM_TIMEOUT_RETRIES", 1), \
         pytest.raises(concurrent.futures.TimeoutError):
        _invoke_with_retry(llm, [], "test_op", timeout_seconds=0.1)

    # First attempt + exactly one retry.
    assert llm.invoke.call_count == 2


def test_deadline_exceeded_retried_at_most_once():
    """A server-side 504 DEADLINE_EXCEEDED gets the single timeout retry."""
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError(
        "504 DEADLINE_EXCEEDED. {'error': {'code': 504, 'status': 'DEADLINE_EXCEEDED'}}"
    )

    with patch("services.llm_helper.LLM_INITIAL_DELAY", 0), \
         patch("services.llm_helper.LLM_TIMEOUT_RETRIES", 1), \
         pytest.raises(RuntimeError, match="DEADLINE_EXCEEDED"):
        _invoke_with_retry(llm, [], "test_op", timeout_seconds=30)

    # First attempt + exactly one retry.
    assert llm.invoke.call_count == 2


def test_deadline_exceeded_retry_can_succeed(capsys):
    llm = MagicMock()
    ok = MagicMock()
    llm.invoke.side_effect = [
        RuntimeError("504 DEADLINE_EXCEEDED. {'error': {'code': 504}}"),
        ok,
    ]

    with patch("services.llm_helper.LLM_INITIAL_DELAY", 0), \
         patch("services.llm_helper.LLM_TIMEOUT_RETRIES", 1):
        result = _invoke_with_retry(llm, [], "test_op", timeout_seconds=30)

    assert result is ok
    assert llm.invoke.call_count == 2
    assert "hit a server-side deadline (504)" in capsys.readouterr().out


def test_deadline_error_match_is_word_bounded():
    """PR/issue numbers like 5043 must not be mistaken for a 504 deadline."""
    from services.llm_helper import _is_deadline_exceeded

    assert _is_deadline_exceeded(RuntimeError("504 DEADLINE_EXCEEDED"))
    assert _is_deadline_exceeded(RuntimeError("upstream returned 504 after 30s"))
    assert not _is_deadline_exceeded(RuntimeError("error fetching PR 5043: not found"))
    assert not _is_deadline_exceeded(RuntimeError("400 INVALID_ARGUMENT"))


# ---- Per-node iteration overrides ----


def test_node_max_iterations_override_caps_the_loop():
    always_calls = MagicMock(
        tool_calls=[{"name": "tool_a", "args": {}, "id": "1"}], content="",
    )
    llm = _fake_llm([always_calls, always_calls, always_calls])
    tool = _fake_tool("tool_a", lambda **kwargs: "ok")

    with patch("services.llm_helper.get_mcp_tools_by_name", return_value=[tool]), \
         patch("services.llm_helper.get_model", return_value=llm), \
         patch("config.get_model_for_node", return_value="fake-model"), \
         patch("config.get_llm_settings_for_node", return_value=(None, 300)), \
         patch("services.llm_helper.NODE_MAX_ITERATIONS", {"check_compliance": 3}):
        invoke_llm_with_tools("check_compliance", {}, "system prompt", "user prompt", FetchResponse)

    assert llm.invoke.call_count == 3
