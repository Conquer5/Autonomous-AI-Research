# Autonomous Digest

## Tujuan

Digest dirancang untuk mesin lama: laptop tidak menjalankan model lokal atau
index besar. Ia hanya melakukan HTTP I/O, parsing XML/JSON kecil, ranking
heuristic, penyimpanan SQLite, dan pengiriman Telegram. Inferensi Gemini terjadi
di cloud dan bersifat opsional.

Default fokus riset:

- AI agents dan agent frameworks;
- efficient local AI dan CPU-only inference;
- model dan alat AI gratis/open-source.

Ubah `DIGEST_TOPICS` di `.env` untuk mengganti fokus tanpa menyentuh kode.
`DIGEST_SEARCH_QUERIES` berisi query pendek yang dipakai untuk GitHub dan arXiv;
pisahkan keduanya agar fokus naratif boleh spesifik tanpa membuat pencarian terlalu
sempit.

## Pipeline

```text
RSS resmi ─┐
GitHub ────┼─ collect bounded ─ deduplicate ─ rank ─ optional Gemini summary
arXiv ─────┤                                      └─ deterministic fallback
Brave ─────┘                                                    │
                                                                 └─ Telegram
```

Satu kegagalan sumber menghasilkan peringatan parsial dan tidak membatalkan
sumber lain. URL baru dicatat ke SQLite setelah delivery CLI berhasil. Mode
`--force` mengabaikan cooldown dan seen-URL filter; `--dry-run` tidak mengubah
state.

## Model

- Main: `gemini-3.5-flash`
- Fast/digest: `gemini-3.5-flash-lite`
- Reasoning: `gemini-3.5-flash`

Model dapat diganti melalui `.env`. Digest tetap berjalan tanpa
`GEMINI_API_KEY`, tetapi tidak membuat ringkasan pembuka berbasis model.

## Scheduling

Unit pada `deploy/systemd` memakai user service sehingga tidak membutuhkan root.
`OnStartupSec=2min` menjalankan radar setelah user manager aktif dan
`OnCalendar=*-*-* 07:30:00 Asia/Jakarta` menjadwalkan briefing harian.
`Persistent=true` mengejar jadwal harian yang terlewat ketika laptop mati.

Default cooldown 12 jam mencegah trigger startup dan kalender mengirim dua kali.
Bot dan digest diberi nilai `Nice`, `CPUWeight`, `IOWeight`, dan `MemoryHigh`
yang konservatif agar pekerjaan background tidak mendominasi perangkat.
