"""Email a job log summary + attached figures.

Usage:
    python slurm/send_report.py <subject> <log_path> [figure_glob ...]

This is the *legacy* reporter: it scrapes a summary out of the job log with a
line filter, which is fragile — it has silently dropped a results table
before. New experiments should subclass
`realm_tools.experiment_lib.reporting.ExperimentReport` and build their
report from their own results instead. This script remains for job scripts
that predate that, and now shares the same SMTP transport so there is only
one implementation of the mailing itself.

Environment variables are documented in realm_tools/experiment_lib/reporting.py.
If EMAIL_TO is unset the script exits silently and the job stays green.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from realm_tools.experiment_lib.reporting import send_email  # noqa: E402


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
    if not os.environ.get('EMAIL_TO'):
        print('EMAIL_TO not set — skipping email.', file=sys.stderr)
        return 0
    if len(sys.argv) < 3:
        print('usage: send_report.py <subject> <log_path> [fig_glob ...]',
              file=sys.stderr)
        return 2

    subject, log_path, fig_globs = sys.argv[1], sys.argv[2], sys.argv[3:]

    p = Path(log_path)
    body = summarise(p.read_text(errors='replace')) if p.exists() else '(no log file)'

    attachments = []
    for spec in fig_globs:
        attachments.extend(sorted(glob.glob(spec)))

    return 0 if send_email(subject, body, attachments) else 1


if __name__ == '__main__':
    sys.exit(main())
