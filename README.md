# Autonomous AI Research Radar

Radar riset AI berbasis bukti untuk mengikuti perkembangan AI, mengevaluasi
cara kerja coding agent, dan menemukan model/tool yang berguna dengan biaya
lebih rendah. Telegram menjadi antarmuka; Gemini dan Hermes membantu analisis,
sementara aplikasi mengelola sumber, batas penggunaan, dan penyimpanan bukti.

**Status 11 September 2026:** inti Phase 3 dan sebagian Phase 4 sudah tersedia.
Panduan berversi untuk fondasi Phase 6 sudah terhubung ke planner. Phase 5–6
belum selesai sepenuhnya. Lihat [status dan batas implementasi](docs/ai-efficiency-radar.md).

## Alur

```mermaid
flowchart LR
    T[Telegram] --> F[Fokus dan jendela waktu]
    F --> P[Planner: Gemini atau Hermes + panduan terpilih]
    P --> S[Sumber tersedia: GitHub / arXiv / RSS / Brave]
    S --> E[Filter tanggal dan registrasi bukti]
    E --> V[Analisis konsensus dan verifikasi]
    V --> A[Jawaban, sumber, dan keterbatasan]
    A --> DB[SQLite: bukti dan diagnostik]
```

Fokus default: perkembangan AI, efisiensi coding agent/konteks, akses model
hemat biaya, serta local AI. Penghematan token dan kesetaraan model harus diuji;
klaim pengembang dibedakan dari hasil pengukuran.

## Penggunaan

- `/research Codex token optimization terbaru` — QUICK, default 14 hari terakhir.
- `/deepresearch Bandingkan metode coding agent hemat token 30 hari terakhir` — DEEP.
- `/research Jelaskan metode transformer dari paper 2017` — sumber historis diperbolehkan.
- `/digest` — briefing lintas sumber; `/weekly` adalah alias manualnya.
- `/repos`, `/papers`, `/web`, `/trending`, `/analyze` — adapter sumber langsung.
- `/skills` — daftar panduan berversi Radar; tidak memanggil model.
- `/memory` — meminta Hermes merangkum memorinya; belum retrieval memori riset aplikasi.
- `/status`, `/help`, `/start` — status dan bantuan.

Pesan biasa memakai alur QUICK. Mode otomatis menggunakan Gemini FAST untuk
QUICK dan Hermes untuk DEEP bila masing-masing tersedia. Satu panduan sesuai
topik dimuat ke permintaan planning, dengan fallback deterministik saat gagal.

## Menjalankan lokal

Python 3.12 atau 3.13, Hermes sebagai proses terpisah, dan credential operator.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
research-radar bot
```

Isi `.env` lokal sebelum menjalankan bot. Telegram memakai allowlist numeric user
ID; jangan gunakan token bot yang sama untuk polling gateway Hermes dan aplikasi.
GitHub publik, arXiv, dan RSS dapat digunakan tanpa Brave; Gemini diperlukan untuk
sintesis berbasis model. [Setup Hermes](docs/hermes-setup.md).

Konfigurasi baru:

```env
RESEARCH_PLANNER_BACKEND=auto
RESEARCH_RECENT_DAYS=14
```

Pilihan backend: `auto`, `gemini`, `hermes`, `deterministic`. Lihat [.env.example](.env.example)
untuk query digest, model dan sumber. Nilai `.env` yang sudah ada tetap mengalahkan
default kode. Harga dan kuota model harus dicek pada akun/provider; tidak ada
jaminan akses gratis atau penghematan tertentu.

arXiv memakai antrean satu koneksi per instance dengan jeda tiga detik dan timeout
khusus `ARXIV_TIMEOUT_SECONDS=60`. Jika server mengembalikan HTTP 429, percobaan
berikutnya menunggu setidaknya 60 detik; `Retry-After` yang lebih panjang tetap
dihormati. Setelah retry habis, kueri lain ditunda selama cooldown, dan digest
menjelaskan pembatasan akses tersebut (bukan menganggap pencarian kosong).
Hindari menjalankan beberapa proses pencarian arXiv sekaligus: pembatasan arXiv
berlaku gabungan untuk mesin yang digunakan, sesuai
[ketentuan API arXiv](https://info.arxiv.org/help/api/tou.html).

## Verifikasi

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src
.venv/bin/pytest
```

Live checks bersifat opt-in dan memakai akses/kuota provider:

```bash
RUN_LIVE_TESTS=1 .venv/bin/pytest -m live tests/integration
```

## Dokumentasi

- [Perubahan radar, panduan Hermes, batas waktu, dan roadmap](docs/ai-efficiency-radar.md)
- [Arsitektur saat ini](docs/architecture.md)
- [Hasil hardening dan live pilot 8 September](docs/retrieval-hardening.md)
- [Digest dan penjadwalan](docs/autonomous-digest.md)
- [Systemd](deploy/systemd/README.md)
- [Roadmap awal](docs/phase-1-plan.md) dan [snapshot Phase 1–2](docs/phase-2-status.md)

Verifikasi klaim masih heuristic; snippet bukan bukti penuh. Memori riset yang
bisa dipakai kembali, evaluasi peningkatan skill otomatis, dan deployment Docker
belum tuntas. Lulus tes offline tidak membuktikan kesiapan produksi atau kualitas
jawaban live.
