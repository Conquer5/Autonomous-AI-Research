"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import NoReturn

from telegram import Update

from research_radar.bootstrap import build_container
from research_radar.config import AppSettings
from research_radar.exceptions import ConfigurationError
from research_radar.observability.logging import configure_logging
from research_radar.telegram.bot import build_telegram_application
from research_radar.telegram.formatter import format_digest
from research_radar.telegram.publisher import publish_digest

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous AI Research Radar")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("bot", help="jalankan bot Telegram")
    digest = subparsers.add_parser("digest", help="buat dan kirim digest satu kali")
    digest.add_argument("--force", action="store_true", help="abaikan cooldown (BUKAN deduplikasi)")
    digest.add_argument(
        "--resend", action="store_true", help="kirim ulang semua evidence termasuk duplikat"
    )
    digest.add_argument("--dry-run", action="store_true", help="cetak hasil tanpa mengirim")
    parser.set_defaults(command="bot", force=False, dry_run=False, resend=False)
    return parser


def _run_bot(settings: AppSettings) -> NoReturn:
    try:
        token = settings.require_telegram_token()
        container = build_container(settings, require_runtime=True)
    except ConfigurationError as exc:
        logger.error(exc.public_message, extra={"event": "configuration_error"})
        raise SystemExit(2) from exc

    application = build_telegram_application(
        token=token,
        service=container.service,
        allowed_user_ids=settings.telegram_allowed_user_ids,
    )
    try:
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=False,
        )
    finally:
        asyncio.run(container.aclose())
    raise SystemExit(0)


async def _run_digest(
    settings: AppSettings, *, force: bool, dry_run: bool, resend: bool = False
) -> None:
    container = build_container(settings, require_runtime=False)
    run_id = container.digest.current_run_id
    registry = container.digest.evidence_registry
    try:
        # Concurrency protection: only one digest at a time
        if registry is not None and not registry.acquire_lock(run_id):
            logger.info(
                "Another digest is already running",
                extra={"event": "digest_skipped_locked", "run_id": run_id},
            )
            return
        try:
            if not force and not await container.digest.is_due():
                logger.info(
                    "Digest cooldown active",
                    extra={"event": "digest_skipped_cooldown", "run_id": run_id},
                )
                return
            digest = await container.digest.generate(force=resend)
            if dry_run:
                print(format_digest(digest))
                logger.info(
                    "Dry-run completed (evidence discovered but not marked as delivered)",
                    extra={"event": "digest_dry_run", "run_id": run_id},
                )
                return
            token = settings.require_telegram_token()
            if digest.evidence_urls() or digest.warnings:
                await publish_digest(
                    token=token,
                    chat_ids=settings.telegram_allowed_user_ids,
                    digest=digest,
                )
            await container.digest.record_sent(digest)
            logger.info(
                "Digest completed",
                extra={
                    "event": "digest_sent",
                    "run_id": run_id,
                    "evidence_count": len(digest.evidence_urls()),
                },
            )
        finally:
            if registry is not None:
                registry.release_lock(run_id)
    finally:
        await container.aclose()


def main() -> None:
    args = _parser().parse_args()
    settings = AppSettings()
    configure_logging(settings.log_level)
    if args.command == "digest":
        try:
            asyncio.run(
                _run_digest(settings, force=args.force, dry_run=args.dry_run, resend=args.resend)
            )
        except ConfigurationError as exc:
            logger.error(exc.public_message, extra={"event": "configuration_error"})
            raise SystemExit(2) from exc
        return
    _run_bot(settings)


if __name__ == "__main__":
    main()
