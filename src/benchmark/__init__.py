"""Benchmark engine for the intra-physician prescribing discordance study.

The modules in this package separate three concerns that the historical
``src.analysis.analysis_pipeline`` mixed in single functions:

- ``estimators``: one class per method, with ``fit`` (learn weights or a
  similarity from training data) separated from the construction of patient
  pairs and the discordance estimate on evaluation data;
- ``designs``: in-sample, cross-fitted and oracle evaluation designs;
- ``runner``: orchestration of the experimental plan and the long output table.
"""
