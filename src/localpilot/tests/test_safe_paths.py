import pytest
from pathlib import Path
from localpilot.utils.security import sanitize_filename, prevent_path_traversal


def test_sanitize_normal():
    assert sanitize_filename("my_file.pdf") == "my_file.pdf"


def test_sanitize_path_sep():
    result = sanitize_filename("../../etc/passwd")
    assert "/" not in result
    assert ".." not in result or result == "passwd"


def test_sanitize_special_chars():
    result = sanitize_filename("hello world!@#.pdf")
    assert " " not in result
    assert "@" not in result


def test_prevent_traversal(tmp_path):
    safe = prevent_path_traversal(tmp_path, "safe.txt")
    assert safe.parent == tmp_path


def test_prevent_traversal_attack(tmp_path):
    with pytest.raises(ValueError):
        prevent_path_traversal(tmp_path, "../../etc/passwd")
