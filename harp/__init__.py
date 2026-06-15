"""HARP — Hierarchical Active Region Pruning.

Efficient, train-based data selection for finetuning large language models.

The entry point is :func:`harp`, which returns a :class:`HarpResult` holding the
two complementary first-order selections described in the paper:

* ``HarpResult.harp_c`` — HARP-C, the *conservative* envelope (controls redundancy)
* ``HarpResult.harp_e`` — HARP-E, the *expansive* envelope (rewards complementary regions)

Paper: "HARP: Efficient Data Selection for Finetuning Large Language Models"
(arXiv:2606.07690).
"""

from .selector import harp, HarpResult, greedy_taskaware

__all__ = ["harp", "HarpResult", "greedy_taskaware"]
__version__ = "0.1.0"
