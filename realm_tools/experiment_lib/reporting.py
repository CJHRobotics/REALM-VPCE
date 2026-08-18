"""Emailed experiment reports.

Every long-running experiment on the cluster ends the same way: mail the
person who submitted it a summary and the figures. This module holds the
part that never changes — SMTP configuration, attachment budgeting, and
failing quietly so a mail problem can never fail an otherwise good job —
and leaves each experiment to say what its own report should contain.

Usage
-----
Subclass `ExperimentReport`, implement `title()` and `body()`, and point
`figures()` / `data_files()` at what should be attached::

    class MyReport(ExperimentReport):
        experiment = 'my-experiment'

        def title(self):
            return f'{self.env_name}: {len(self.metrics)} runs'

        def body(self):
            return self.section('Results', self.table(self.metrics))

        def figures(self):
            return sorted(glob.glob(f'{self.fig_dir}/*.png'))

    MyReport(metrics=df, fig_dir=d, job_id=jid, exit_status=st).send()

Bodies are built from the experiment's own results, not scraped out of the
job log. Scraping was how the previous mailer worked and it silently
dropped the results table, which is the one thing worth reading.

Environment
-----------
    EMAIL_TO         required; without it every send is a silent no-op
    EMAIL_FROM       defaults to <user>@<hostname>
    EMAIL_SMTP       defaults to localhost
    EMAIL_SMTP_PORT  defaults to 25; 587 implies STARTTLS
    EMAIL_SMTP_USER  optional; if set, triggers STARTTLS + login
    EMAIL_SMTP_PASS  paired with EMAIL_SMTP_USER
"""

from __future__ import annotations

import getpass
import mimetypes
import os
import smtplib
import socket
import traceback
from abc import ABC, abstractmethod
from email.message import EmailMessage
from pathlib import Path

# Most institutional mail servers reject well before this; staying under it
# is the difference between a report and a bounce.
DEFAULT_MAX_ATTACHMENT_BYTES = 18 * 1024 ** 2


# ------------------------------------------------------------------ transport

def smtp_settings():
    """SMTP configuration from the environment."""
    port = int(os.environ.get('EMAIL_SMTP_PORT', '25'))
    user = os.environ.get('EMAIL_SMTP_USER')
    pwd = os.environ.get('EMAIL_SMTP_PASS')
    explicit_tls = os.environ.get('EMAIL_SMTP_TLS', '').lower() in ('1', 'true', 'yes')
    return dict(
        to=os.environ.get('EMAIL_TO'),
        sender=os.environ.get('EMAIL_FROM',
                              f'{getpass.getuser()}@{socket.gethostname()}'),
        host=os.environ.get('EMAIL_SMTP', 'localhost'),
        port=port,
        user=user,
        password=pwd,
        # Auth requires TLS, and 587 is the submission port, so upgrade on
        # either even when the flag was not set.
        use_tls=explicit_tls or port == 587 or bool(user and pwd),
    )


def send_email(subject, body, attachments=(), to=None,
               max_attachment_bytes=DEFAULT_MAX_ATTACHMENT_BYTES, verbose=True):
    """Send one mail. Returns True on success.

    Never raises: a mail failure must not change the outcome of the job that
    produced the results.
    """
    cfg = smtp_settings()
    to = to or cfg['to']
    if not to:
        if verbose:
            print('  [report] EMAIL_TO not set — skipping email')
        return False

    msg = EmailMessage()
    msg['From'] = cfg['sender']
    msg['To'] = to
    msg['Subject'] = subject
    msg.set_content(body)

    budget, attached, skipped = max_attachment_bytes, [], []
    for path in attachments:
        p = Path(path)
        if not p.exists():
            skipped.append((p.name, 'missing'))
            continue
        size = p.stat().st_size
        if size > budget:
            skipped.append((p.name, f'{size/1e6:.1f} MB over budget'))
            continue
        ctype, _ = mimetypes.guess_type(str(p))
        maintype, _, subtype = (ctype or 'application/octet-stream').partition('/')
        msg.add_attachment(p.read_bytes(), maintype=maintype,
                           subtype=subtype or 'octet-stream', filename=p.name)
        attached.append(p.name)
        budget -= size

    if skipped and verbose:
        for name, why in skipped:
            print(f'  [report] not attached: {name} ({why})')

    try:
        with smtplib.SMTP(cfg['host'], cfg['port'], timeout=30) as s:
            s.ehlo()
            if cfg['use_tls']:
                s.starttls()
                s.ehlo()
            if cfg['user'] and cfg['password']:
                s.login(cfg['user'], cfg['password'])
            s.send_message(msg)
    except Exception as exc:                                # noqa: BLE001
        if verbose:
            print(f'  [report] send failed via {cfg["host"]}:{cfg["port"]} — {exc}')
        return False

    if verbose:
        print(f'  [report] sent to {to} with {len(attached)} attachment(s): '
              f'{", ".join(attached) if attached else "none"}')
    return True


# ------------------------------------------------------------------ base class

class ExperimentReport(ABC):
    """Base class for an experiment's emailed report.

    Subclasses implement `title()` and `body()`. Everything else has a
    working default.
    """

    #: short slug used in the subject line
    experiment = 'experiment'

    def __init__(self, *, env_name=None, job_id=None, exit_status=0,
                 log_path=None, fig_dir=None, out_dir=None, **extra):
        self.env_name = env_name
        self.job_id = job_id or os.environ.get('SLURM_JOB_ID', 'local')
        self.exit_status = exit_status
        self.log_path = log_path
        self.fig_dir = fig_dir
        self.out_dir = out_dir
        for k, v in extra.items():
            setattr(self, k, v)

    # ---- subclasses must provide ----------------------------------------

    @abstractmethod
    def title(self) -> str:
        """One line describing the outcome. Used in the subject."""

    @abstractmethod
    def body(self) -> str:
        """The report text, built from results rather than from the log."""

    # ---- subclasses may override ----------------------------------------

    def figures(self):
        """Image paths to attach."""
        return []

    def data_files(self):
        """Data files (csv, json) to attach."""
        return []

    def subject(self):
        status = 'ok' if self.exit_status == 0 else f'FAILED ({self.exit_status})'
        env = f' {self.env_name}' if self.env_name else ''
        return f'[REALM-VPCE]{env} {self.experiment} — {self.title()} [{status}, job {self.job_id}]'

    def header(self):
        lines = [f'experiment : {self.experiment}']
        if self.env_name:
            lines.append(f'environment: {self.env_name}')
        lines += [f'job        : {self.job_id}',
                  f'exit status: {self.exit_status}']
        if self.log_path:
            lines.append(f'log        : {self.log_path}')
        if self.out_dir:
            lines.append(f'outputs    : {self.out_dir}')
        return '\n'.join(lines)

    def footer(self):
        if self.exit_status == 0:
            return ''
        return ('\n\nThe job exited non-zero. Results above may be partial; '
                'check the log.')

    def compose(self):
        return f'{self.header()}\n\n{"=" * 68}\n\n{self.body()}{self.footer()}\n'

    def send(self, dry_run=False, verbose=True):
        try:
            text = self.compose()
        except Exception:                                   # noqa: BLE001
            text = ('Report generation failed; the job itself may be fine.\n\n'
                    + traceback.format_exc())
        attachments = list(self.data_files()) + list(self.figures())
        if dry_run:
            print(f'SUBJECT: {self.subject()}\n')
            print(text)
            print(f'ATTACHMENTS: {[str(a) for a in attachments]}')
            return True
        return send_email(self.subject(), text, attachments, verbose=verbose)

    # ---- formatting helpers ---------------------------------------------

    @staticmethod
    def section(heading, text):
        return f'{heading}\n{"-" * len(heading)}\n{text}\n'

    @staticmethod
    def table(df, columns=None, float_format='%.3f', max_rows=60):
        """Fixed-width rendering of a DataFrame for a plain-text mail."""
        if df is None or not len(df):
            return '(no rows)'
        d = df[columns] if columns else df
        if len(d) > max_rows:
            return (d.head(max_rows).to_string(index=False, float_format=float_format)
                    + f'\n... {len(d) - max_rows} more rows')
        return d.to_string(index=False, float_format=float_format)

    @staticmethod
    def bullets(items):
        return '\n'.join(f'  - {i}' for i in items)
