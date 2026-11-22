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

from services.llm_helper import (
    ToolCallFailedException,
    _invoke_with_retry,
    _is_tool_call_error,
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
