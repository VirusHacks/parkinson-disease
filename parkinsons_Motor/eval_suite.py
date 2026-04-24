"""Backward-compatible benchmark entrypoint.

This wrapper intentionally delegates to `parkinsons_Motor.evaluation.eval_suite`
so existing commands keep working while the real implementation lives in the
cleaner `evaluation/` package.
"""

from parkinsons_Motor.evaluation.eval_suite import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
