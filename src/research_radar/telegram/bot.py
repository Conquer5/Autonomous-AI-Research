"""Build the async python-telegram-bot application."""

from __future__ import annotations

import logging

from telegram.ext import Application, CommandHandler, MessageHandler, filters

from research_radar.service import RadarService
from research_radar.telegram.handlers import TelegramHandlers

logger = logging.getLogger(__name__)


async def telegram_error_handler(update: object, context: object) -> None:
    del update
    error = getattr(context, "error", None)
    logger.error(
        "Unhandled Telegram framework error",
        extra={"event": "telegram_framework_error", "error_type": type(error).__name__},
    )


def build_telegram_application(
    *, token: str, service: RadarService, allowed_user_ids: frozenset[int]
) -> Application:
    handlers = TelegramHandlers(service, allowed_user_ids)
    application = Application.builder().token(token).build()
    commands = {
        "start": handlers.start,
        "help": handlers.help,
        "research": handlers.research,
        "deepresearch": handlers.deepresearch,
        "trending": handlers.trending,
        "papers": handlers.papers,
        "repos": handlers.repos,
        "web": handlers.web,
        "analyze": handlers.analyze,
        "digest": handlers.digest,
        "weekly": handlers.weekly,
        "status": handlers.status,
        "memory": handlers.memory,
        "skills": handlers.skills,
    }
    for command, callback in commands.items():
        application.add_handler(CommandHandler(command, callback))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.natural_message)
    )
    application.add_error_handler(telegram_error_handler)
    return application
