"""judge_lab: specialist judge heads trained from the owner's review verdicts.

Three rules (see README.md):
  1. The owner's verdict is the only ground truth.
  2. Exam images are never used for training or fitting.
  3. Heads only output scores; they never make pass/fail decisions.
"""

__version__ = "0.1.0"
