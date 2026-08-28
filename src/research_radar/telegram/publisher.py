"""Proactive Telegram delivery used by the one-shot digest command."""

from __future__ import annotations

from telegram import Bot
from telegram.constants import ParseMode

from research_radar.schemas import ResearchDigest
from research_radar.telegram.formatter import format_digest, split_html_blocks


async def publish_digest(*, token: str, chat_ids: frozenset[int], digest: ResearchDigest) -> int:
    chunks = split_html_blocks(format_digest(digest))
    sent = 0
    async with Bot(token) as bot:
        for chat_id in sorted(chat_ids):
            for chunk in chunks:
                await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
                sent += 1
    return sent
