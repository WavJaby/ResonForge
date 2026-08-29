"""Immutable in-memory MIDI documents."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import mido


@dataclass(frozen=True)
class MidiDocument:
    """A standard MIDI file carried between pipeline stages as bytes."""

    data: bytes

    @classmethod
    def from_mido(cls, midi: mido.MidiFile) -> MidiDocument:
        stream = BytesIO()
        midi.save(file=stream)
        return cls(stream.getvalue())

    @classmethod
    def read(cls, path: str | Path) -> MidiDocument:
        return cls(Path(path).read_bytes())

    def to_mido(self) -> mido.MidiFile:
        return mido.MidiFile(file=BytesIO(self.data))

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_bytes(self.data)
        temporary.replace(output)
        return output
