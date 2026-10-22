"""specbench: a unified speculative-decoding evaluation module.

Two evaluation protocols (harnesses) that must never be mixed in a single tok/s
comparison:

- ``sglang``: launches a real SGLang server and measures end-to-end,
  wall-clock throughput plus ``spec_accept_length`` from the server response.
  This is the production-representative harness.
- ``hf_direct``: loads the target + draft models directly with HuggingFace and
  drives ``draft_model.spec_generate`` in-process. Useful for methods whose
  SGLang serving path does not exist yet.

Both harnesses emit the same canonical :class:`specbench.metrics.BenchResult`
schema so results are stored consistently. Cross-harness numbers are NOT
comparable for tok/s (different inference stacks); only ``accept_length`` is
harness-independent.

Methods are declared once in :mod:`specbench.methods`. Adding a new method
(e.g. DFlare / Domino / DSpark) is a registry entry, not a new script.
"""

from specbench.config import BenchConfig
from specbench.methods import MethodSpec, get_method, list_methods
from specbench.metrics import BenchResult

__all__ = [
    "BenchConfig",
    "BenchResult",
    "MethodSpec",
    "get_method",
    "list_methods",
]
