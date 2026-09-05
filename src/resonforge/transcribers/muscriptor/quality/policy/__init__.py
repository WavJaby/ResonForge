"""How chunks are joined, judged and recovered.

Two importers outside this package: `quality_plan`, which puts it in play, and `transcription`, which still builds the plan's defaults, routes its diagnostics events and prepares recovery candidates for the scheduler -- policy work that has not moved in here yet.
"""
