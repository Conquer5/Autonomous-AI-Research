from __future__ import annotations

from research_radar.telegram.authorization import is_authorized
from research_radar.telegram.formatter import split_html_blocks, split_plain_text


def test_authorization_is_fail_closed() -> None:
    assert not is_authorized(1, frozenset())
    assert not is_authorized(None, frozenset({1}))
    assert not is_authorized(2, frozenset({1}))
    assert is_authorized(1, frozenset({1}))


def test_long_plain_messages_are_split_below_limit() -> None:
    text = "paragraph " * 1000
    chunks = split_plain_text(text, limit=200)

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_html_is_split_at_balanced_blocks() -> None:
    text = "\n\n".join(f"<b>{index}</b> result" for index in range(20))
    chunks = split_html_blocks(text, limit=80)

    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks)
    assert all(chunk.count("<b>") == chunk.count("</b>") for chunk in chunks)
