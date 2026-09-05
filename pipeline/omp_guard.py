"""Force LightGBM to load before PyTorch.

On macOS, LightGBM and PyTorch each ship their own libomp. If torch's copy is loaded
first, LightGBM's C Dataset constructor segfaults (SIGSEGV, no Python traceback) the
moment it is handed real data. Importing lightgbm first avoids it.

Import this module as the FIRST import of any entrypoint that uses both libraries:

    import omp_guard  # noqa: F401  # must precede torch

This is deliberately a separate module rather than a bare `import lightgbm` at the
top of each script, because an import sorter would happily move `import lightgbm`
below `import torch` and silently reintroduce a hard crash.
"""

import lightgbm as _lightgbm  # noqa: F401  # isort:skip  — must load before torch

__all__: list[str] = []
