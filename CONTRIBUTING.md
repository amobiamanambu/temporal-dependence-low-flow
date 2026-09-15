# Contributing

Bug reports and reproducibility improvements are welcome. Please open a GitHub
issue that includes the operating system, Python version, command run, complete
error message, and the smallest input that reproduces the problem.

For code changes:

1. Keep scientific defaults in the JSON configuration rather than embedding
   new values in scripts.
2. Preserve eight-character USGS station identifiers as strings.
3. Do not mix development, calibration, and held-out evaluation periods.
4. Add or update a unit test for any scoring or data-schema change.
5. Run `python -m unittest discover -s tests` and
   `python scripts/run_sample_workflow.py` before proposing the change.
6. Do not commit raw provider data, generated continental intermediates,
   credentials, or personal absolute paths.

Methodological changes that alter a reported estimand should be proposed
separately from bug fixes and should document how the change affects archived
results.
