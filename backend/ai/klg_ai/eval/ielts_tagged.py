"""Gemini-tagging IELTS benchmark — the recall-fixed arm.

Same 11 transcripts, same engine + metrics as ``ielts_graded`` (the no-tagging
baseline), but the chat endpoint runs with ``tagger="gemini"``: Gemini *proposes*
the grammar concepts each turn uses (from the full map catalog) instead of merely
vetoing the semantic mapper's guesses. This lifts the recall ceiling that capped
the baseline at B2, so the system's read should climb the CEFR ladder with the
band — testing whether tagging raises the band↔system correlation.

Requires the backend running with a Gemini key. Writes the ``tagged`` arm into
``docs/ielts-case-study/results.json`` next to ``ungraded`` + ``graded``.
Run: ``python -m klg_ai.eval.ielts_tagged``.
"""
from __future__ import annotations

from klg_ai.eval.ielts_graded import _print_summary, run_graded

if __name__ == "__main__":
    _print_summary(run_graded(arm="tagged", tagger="gemini",
                              mapper_label="gemini tagging (proposes)"))
