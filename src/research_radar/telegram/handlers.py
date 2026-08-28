"""Telegram command and natural-language handlers."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes

from research_radar.exceptions import RadarError
from research_radar.research.models import ResearchMode
from research_radar.service import RadarService
from research_radar.telegram.authorization import is_authorized
from research_radar.telegram.formatter import (
    format_digest,
    format_papers,
    format_repositories,
    format_repository_analysis,
    format_research_result,
    format_status,
    format_web_results,
    split_html_blocks,
    split_plain_text,
)

logger = logging.getLogger(__name__)


class TelegramHandlers:
    def __init__(self, service: RadarService, allowed_user_ids: frozenset[int]) -> None:
        self.service = service
        self.allowed_user_ids = allowed_user_ids

    async def _authorized_user(self, update: Update) -> int | None:
        user_id = update.effective_user.id if update.effective_user else None
        if is_authorized(user_id, self.allowed_user_ids):
            return user_id
        if update.effective_message:
            await update.effective_message.reply_text("Akses ditolak.")
        return None

    async def _send_plain(self, update: Update, text: str) -> None:
        if not update.effective_message:
            return
        for chunk in split_plain_text(text):
            await update.effective_message.reply_text(chunk)

    async def _send_html(self, update: Update, text: str) -> None:
        if not update.effective_message:
            return
        for chunk in split_html_blocks(text):
            await update.effective_message.reply_text(chunk, parse_mode=ParseMode.HTML)

    async def _safe_call(
        self,
        update: Update,
        operation: Callable[[], Awaitable[str]],
        *,
        html_result: bool,
    ) -> None:
        if not update.effective_message:
            return
        await update.effective_message.reply_chat_action(ChatAction.TYPING)
        try:
            text = await operation()
            if html_result:
                await self._send_html(update, text)
            else:
                await self._send_plain(update, text)
        except RadarError as exc:
            await self._send_plain(update, f"Operasi gagal: {exc}")
        except Exception:
            logger.exception(
                "Unexpected Telegram handler failure", extra={"event": "handler_error"}
            )
            await self._send_plain(
                update,
                "Terjadi kesalahan internal. Detail telah dicatat tanpa menampilkan stack trace.",
            )

    @staticmethod
    def _args(context: ContextTypes.DEFAULT_TYPE) -> str:
        return " ".join(context.args or []).strip()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if await self._authorized_user(update) is None:
            return
        await self._send_html(
            update,
            "<b>🔥 Autonomous AI Research Radar</b>\n\n"
            "Kirim pertanyaan riset atau gunakan /help untuk melihat command.",
        )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if await self._authorized_user(update) is None:
            return
        await self._send_plain(
            update,
            "/research <goal> — riset otonom cepat berbasis bukti\n"
            "/deepresearch <goal> — riset otonom mendalam multi-iterasi\n"
            "/repos <query> — cari repository GitHub\n"
            "/papers <query> — cari paper arXiv 30 hari terakhir\n"
            "/trending [query] — repository baru 7 hari terakhir\n"
            "/web <query> — cari sumber web terbaru\n"
            "/analyze <owner/repo atau URL> — ambil metadata, README, release, commit\n"
            "/digest — radar berita, GitHub, dan arXiv terbaru\n"
            "/weekly — alias untuk /digest\n"
            "/memory — ringkasan memory Hermes\n"
            "/skills — daftar skill Hermes yang relevan\n"
            "/status — health dan metrik proses ini",
        )

    async def research(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = await self._authorized_user(update)
        if user_id is None:
            return
        query = self._args(context)
        if not query:
            await self._send_plain(update, "Gunakan: /research <tujuan riset>")
            return
        await self._safe_call(
            update,
            lambda: self._research_text(query, user_id, mode=ResearchMode.QUICK),
            html_result=True,
        )

    async def deepresearch(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = await self._authorized_user(update)
        if user_id is None:
            return
        query = self._args(context)
        if not query:
            await self._send_plain(update, "Gunakan: /deepresearch <tujuan riset>")
            return
        await self._safe_call(
            update,
            lambda: self._research_text(query, user_id, mode=ResearchMode.DEEP),
            html_result=True,
        )

    async def natural_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        user_id = await self._authorized_user(update)
        if user_id is None or not update.effective_message or not update.effective_message.text:
            return
        text = update.effective_message.text.strip()
        await self._safe_call(
            update,
            lambda: self._research_text(text, user_id, mode=ResearchMode.QUICK),
            html_result=True,
        )

    async def _research_text(self, query: str, user_id: int, mode: ResearchMode) -> str:
        result = await self.service.research(query, user_id=user_id, mode=mode)
        return format_research_result(result)

    async def _agent_text(self, text: str, user_id: int) -> str:
        return (await self.service.ask_agent(text, user_id=user_id)).text

    async def repos(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = await self._authorized_user(update)
        if user_id is None:
            return
        query = self._args(context)
        if not query:
            await self._send_plain(update, "Gunakan: /repos <kata kunci>")
            return
        await self._safe_call(
            update,
            lambda: self._repository_text(query, user_id),
            html_result=True,
        )

    async def _repository_text(self, query: str, user_id: int) -> str:
        return format_repositories(await self.service.search_repositories(query, user_id=user_id))

    async def trending(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = await self._authorized_user(update)
        if user_id is None:
            return
        query = self._args(context) or "AI agent"
        await self._safe_call(
            update,
            lambda: self._trending_text(query, user_id),
            html_result=True,
        )

    async def _trending_text(self, query: str, user_id: int) -> str:
        result = await self.service.search_repositories(query, user_id=user_id, days=7)
        return format_repositories(result)

    async def papers(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = await self._authorized_user(update)
        if user_id is None:
            return
        query = self._args(context)
        if not query:
            await self._send_plain(update, "Gunakan: /papers <kata kunci>")
            return
        await self._safe_call(
            update,
            lambda: self._paper_text(query, user_id),
            html_result=True,
        )

    async def _paper_text(self, query: str, user_id: int) -> str:
        return format_papers(await self.service.search_papers(query, user_id=user_id))

    async def web(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = await self._authorized_user(update)
        if user_id is None:
            return
        query = self._args(context)
        if not query:
            await self._send_plain(update, "Gunakan: /web <kata kunci>")
            return
        await self._safe_call(
            update,
            lambda: self._web_text(query, user_id),
            html_result=True,
        )

    async def _web_text(self, query: str, user_id: int) -> str:
        return format_web_results(await self.service.search_web(query, user_id=user_id))

    async def analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = await self._authorized_user(update)
        if user_id is None:
            return
        reference = self._args(context)
        if not reference:
            await self._send_plain(update, "Gunakan: /analyze <owner/repo atau URL GitHub>")
            return
        await self._safe_call(
            update,
            lambda: self._analysis_text(reference, user_id),
            html_result=True,
        )

    async def _analysis_text(self, reference: str, user_id: int) -> str:
        analysis = await self.service.analyze_repository(reference, user_id=user_id)
        return format_repository_analysis(analysis)

    async def weekly(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        user_id = await self._authorized_user(update)
        if user_id is None:
            return
        await self._safe_call(
            update,
            lambda: self._digest_text(),
            html_result=True,
        )

    async def digest(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.weekly(update, context)

    async def _digest_text(self) -> str:
        return format_digest(await self.service.generate_digest(force=True))

    async def memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        user_id = await self._authorized_user(update)
        if user_id is None:
            return
        await self._safe_call(
            update,
            lambda: self._agent_text(
                "Summarize the research-relevant persistent memory you currently have. "
                "Do not reveal secrets or unrelated private data.",
                user_id,
            ),
            html_result=False,
        )

    async def skills(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        user_id = await self._authorized_user(update)
        if user_id is None:
            return
        await self._safe_call(
            update,
            lambda: self._agent_text(
                "List the Hermes skills currently available that are relevant to AI research. "
                "Be concise and do not claim a skill exists unless you can inspect it.",
                user_id,
            ),
            html_result=False,
        )

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if await self._authorized_user(update) is None:
            return
        await self._safe_call(
            update,
            lambda: self._status_text(),
            html_result=True,
        )

    async def _status_text(self) -> str:
        return format_status(await self.service.status())
