"""Email a compact job report + attached figures.

Usage:
    python slurm/send_report.py <subject> <log_path> [figure_glob ...]

Env vars:
    EMAIL_TO           (required) recipient address
    EMAIL_FROM         optional; defaults to slurm@<hostname>
    EMAIL_SMTP         optional; defaults to 'localhost'
    EMAIL_SMTP_PORT    optional; defaults to 25
    EMAIL_SMTP_USER    optional; if set, triggers STARTTLS + login
    EMAIL_SMTP_PASS    optional; paired with EMAIL_SMTP_USER

If EMAIL_TO is unset the script exits silently — the job stays green.
"""
from __future__ import annotations
import glob
import mimetypes
import os
import socket
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path


# Lines matching any of these substrings are kept in the summary body.
# Cheap heuristic; edit here to tune what shows up in the email.
INTERESTING = (
    '=====', 'ENV', 'DATA', 'BANK',
    '[full]', '[lidar]', '[visual]', '[',
    'linkage', 'candidates', 'funnel', 'after ',
    'Bank saved', 'Comparison plot', 'Plot ->', 'Plot ',
    'Started', 'Finished', 'Git',
    # channel_isolation: the run summary, the result table, and the
    # environment/device banner.
    'Metrics ->', 'Figures ->', 'coverage', 'device:', 'feature matrix',
    'environment:', 'locations:',
)


def summarise(text: str) -> str:
    hits = [line for line in text.splitlines()
            if any(tag in line for tag in INTERESTING)]
    return '\n'.join(hits) if hits else '(no matching lines in log)'


def main() -> int:
    to = os.environ.get('EMAIL_TO')
    if not to:
        print('EMAIL_TO not set — skipping email.', file=sys.stderr)
        return 0

    if len(sys.argv) < 3:
        print('usage: send_report.py <subject> <log_path> [fig_glob ...]',
              file=sys.stderr)
        return 2

    subject   = sys.argv[1]
    log_path  = sys.argv[2]
    fig_globs = sys.argv[3:]

    body = '(no log file)'
    p = Path(log_path)
    if p.exists():
        body = summarise(p.read_text(errors='replace'))

    msg = EmailMessage()
    msg['From'] = os.environ.get('EMAIL_FROM', f'slurm@{socket.gethostname()}')
    msg['To']   = to
    msg['Subject'] = subject
    msg.set_content(body)

    n_attached = 0
    for spec in fig_globs:
        for path in sorted(glob.glob(spec)):
            data = Path(path).read_bytes()
            ctype, _ = mimetypes.guess_type(path)
            maintype, _, subtype = (ctype or 'application/octet-stream').partition('/')
            msg.add_attachment(data, maintype=maintype, subtype=subtype or 'octet-stream',
                               filename=Path(path).name)
            n_attached += 1

    host    = os.environ.get('EMAIL_SMTP', 'localhost')
    port    = int(os.environ.get('EMAIL_SMTP_PORT', '25'))
    user    = os.environ.get('EMAIL_SMTP_USER')
    pwd     = os.environ.get('EMAIL_SMTP_PASS')
    tls_env = os.environ.get('EMAIL_SMTP_TLS', '').lower() in ('1', 'true', 'yes')
    # Auto-upgrade on the standard submission port even if the env var
    # wasn't set; auth also requires TLS.
    use_tls = tls_env or port == 587 or bool(user and pwd)

    try:
        with smtplib.SMTP(host, port, timeout=30) as s:
            s.ehlo()
            if use_tls:
                s.starttls()
                s.ehlo()
                print(f'  starttls: on ({host}:{port})', file=sys.stderr)
            if user and pwd:
                s.login(user, pwd)
                print(f'  auth as {user}', file=sys.stderr)
            s.send_message(msg)
    except Exception as e:
        print(f'send failed via {host}:{port} — {e}', file=sys.stderr)
        return 1

    print(f'sent to {to} via {host}:{port}, {n_attached} attachment(s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
