"""Tests for the core tool implementations."""

from pathlib import Path

from avoid_agent.agent.tools.core import edit_file, read_file


def test_read_file_full_contents_includes_line_metadata(tmp_path: Path):
    target = tmp_path / "sample.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")

    result = read_file(str(target))

    assert result.content == "one\ntwo\nthree\n"
    assert result.details["proof"]["start_line"] == 1
    assert result.details["proof"]["end_line"] == 3
    assert result.details["proof"]["total_lines"] == 3
    assert result.details["proof"]["truncated"] is False


def test_read_file_range_returns_requested_window(tmp_path: Path):
    target = tmp_path / "sample.txt"
    target.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    result = read_file(str(target), start_line=2, limit=2)

    assert result.content == "two\nthree\n"
    assert result.details["proof"]["start_line"] == 2
    assert result.details["proof"]["end_line"] == 3
    assert result.details["proof"]["total_lines"] == 4
    assert result.details["proof"]["truncated"] is True


def test_read_file_invalid_range_returns_error(tmp_path: Path):
    target = tmp_path / "sample.txt"
    target.write_text("one\n", encoding="utf-8")

    result = read_file(str(target), start_line=0)

    assert result.content == "Error: start_line must be >= 1"


def test_edit_file_exact_string_mode_remains_supported(tmp_path: Path):
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    result = edit_file(str(target), old_string="beta", new_string="gamma")

    assert result.content == f"Edit applied to {target}"
    assert target.read_text(encoding="utf-8") == "alpha\ngamma\n"
    assert result.details["proof"]["edit_mode"] == "string"


def test_edit_file_line_range_mode_replaces_selected_lines(tmp_path: Path):
    target = tmp_path / "sample.txt"
    target.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    result = edit_file(str(target), start_line=2, end_line=3, replacement="middle\n")

    assert result.content == f"Edit applied to {target}"
    assert target.read_text(encoding="utf-8") == "one\nmiddle\nfour\n"
    assert result.details["proof"]["edit_mode"] == "line_range"
    assert result.details["proof"]["start_line"] == 2
    assert result.details["proof"]["end_line"] == 3


def test_edit_file_rejects_mixed_modes(tmp_path: Path):
    target = tmp_path / "sample.txt"
    target.write_text("one\n", encoding="utf-8")

    result = edit_file(
        str(target),
        old_string="one",
        new_string="two",
        start_line=1,
        end_line=1,
        replacement="two\n",
    )

    assert "either string replacement or line-range replacement" in result.content
