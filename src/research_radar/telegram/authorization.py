"""Fail-closed Telegram authorization."""

from __future__ import annotations


def is_authorized(user_id: int | None, allowed_user_ids: frozenset[int]) -> bool:
    return user_id is not None and bool(allowed_user_ids) and user_id in allowed_user_ids
