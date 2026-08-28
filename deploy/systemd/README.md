# systemd user services

Unit ini menjalankan Hermes dan bot saat user login, lalu menjalankan digest dua
menit setelah user manager aktif dan setiap hari pukul 07.30 WIB. Cooldown
aplikasi mencegah dua pemicu yang berdekatan mengirim laporan ganda.

Gunakan unit Hermes yang dibuat installer resminya, lalu tautkan unit aplikasi
tanpa akses root:

```bash
hermes gateway install
systemctl --user link "$PWD/deploy/systemd/research-radar-bot.service"
systemctl --user link "$PWD/deploy/systemd/research-radar-digest.service"
systemctl --user link "$PWD/deploy/systemd/research-radar-digest.timer"
systemctl --user daemon-reload
systemctl --user enable --now hermes-gateway.service research-radar-bot.service
systemctl --user enable --now research-radar-digest.timer
```

Periksa tanpa membuka log berisi credential:

```bash
systemctl --user status hermes-gateway.service research-radar-bot.service
systemctl --user list-timers research-radar-digest.timer
journalctl --user -u research-radar-digest.service -n 50 --no-pager
```

Untuk mengubah jadwal, edit `OnCalendar` lalu jalankan `systemctl --user
daemon-reload` dan restart timer. Unit memakai `Nice`, `CPUWeight`, `IOWeight`,
dan `MemoryHigh` agar pekerjaan background tidak mendominasi laptop lama.
