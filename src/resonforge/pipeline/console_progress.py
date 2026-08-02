"""Single-threaded rendering for parallel transcription progress."""

from __future__ import annotations

import logging
import shutil
import time

from tqdm.auto import tqdm

_BAR_FORMAT = (
    "{desc} {percentage:3.0f}%|{bar}| "
    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
)
_OVERALL_BAR_FORMAT = (
    "{desc} {percentage:5.1f}%|{bar}| "
    "{n:.2f}/{total:.0f} [{elapsed}<{remaining}, {rate_fmt}]"
)


class TranscriptionProgress:
    def __init__(self, names: list[str], *, enabled: bool) -> None:
        self.enabled = enabled
        self.started = time.monotonic()
        self.state: dict[str, tuple[int, int]] = {
            name: (0, 0) for name in names
        }
        self.bars = {
            name: tqdm(
                total=None,
                desc=f"{name:<7}",
                position=index,
                leave=True,
                disable=not enabled,
                unit="chunk",
                bar_format="{desc} waiting",
                dynamic_ncols=True,
            )
            for index, name in enumerate(names)
        }
        self.overall = tqdm(
            total=len(names),
            desc="overall",
            position=len(names),
            leave=True,
            disable=not enabled,
            unit="stem",
            bar_format=_OVERALL_BAR_FORMAT,
            dynamic_ncols=True,
        )

    def update(self, stem: str, completed: int, total: int) -> None:
        previous, _ = self.state.get(stem, (0, 0))
        self.state[stem] = (completed, total)
        bar = self.bars[stem]
        bar.total = total
        bar.bar_format = _BAR_FORMAT
        bar.update(completed - previous)
        bar.refresh()

        self._refresh_overall()

    def _refresh_overall(self) -> None:
        completed_stems = sum(
            min(1.0, completed / total) if total > 0 else 0.0
            for completed, total in self.state.values()
        )
        self.overall.update(completed_stems - self.overall.n)
        self.overall.refresh()

    def complete(self, stem: str, elapsed: float, detail: str) -> None:
        _, total = self.state[stem]
        self.state[stem] = (total, total) if total > 0 else (1, 1)
        self._refresh_overall()
        message = f"{stem:<7} done {elapsed:.1f}s"
        if detail:
            message += f"; {detail}"
        width = shutil.get_terminal_size(fallback=(120, 24)).columns
        if len(message) >= width:
            message = message[: max(1, width - 2)].rstrip() + "…"
        bar = self.bars[stem]
        bar.set_description_str(message, refresh=False)
        bar.bar_format = "{desc}"
        bar.refresh()

    def close(self) -> None:
        self.overall.set_description_str(
            f"overall done {time.monotonic() - self.started:.1f}s",
            refresh=False,
        )
        self.overall.bar_format = "{desc}"
        self.overall.refresh()
        for bar in self.bars.values():
            bar.close()
        self.overall.close()

    def log_completion(
        self,
        logger: logging.Logger,
        stem: str,
        elapsed: float,
        gpu_label: str,
        summary: str,
    ) -> None:
        message = f"timed {stem:<7} {elapsed:.3f}s ({gpu_label})"
        if summary:
            message += f"; {summary}"
        logger.info(message, extra={"console": not self.enabled})
