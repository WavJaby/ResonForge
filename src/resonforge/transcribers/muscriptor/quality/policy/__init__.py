"""How chunks are joined, judged and recovered.

One importer outside this package: `quality_plan`. Anything a caller needs
from in here -- the stream, the plan's defaults, the wire shape of a
diagnostics event -- it asks the plan for, so a policy change never reaches
the pipeline as a changed import.
"""
