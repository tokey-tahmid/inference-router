"""Experiment driver for the inference-router (P3).

Deliberately a thin layer over P1's workload generator, which is imported from
the shared venv rather than reimplemented. If the two ever diverged, P1's
prefix-caching numbers would stop being comparable to P3's routing numbers,
which is the entire point of pairing them.
"""

__all__ = ["experiment", "loadgen", "metrics_scrape", "provenance", "stats"]
