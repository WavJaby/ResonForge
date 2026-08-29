# ResonForge

ResonForge converts one or more songs to MIDI through one memory-first,
job-oriented pipeline:

```text
CLI / future HTTP adapter
  -> PipelineService
  -> input staging + BS-RoFormer cache
  -> stem selection and preprocessing
  -> global MuScriptor GPU scheduler
  -> in-memory events, MIDI, and postprocessing
  -> optional files + global telemetry
```

MuScriptor is the only enabled transcriber. MT3 and Basic Pitch adapters remain
disabled until they implement the same in-memory result contract.

## Setup

```powershell
uv python install 3.11
uv sync --locked
```

FFmpeg is required for non-WAV inputs. BS-RoFormer weights belong under
`models/bs-roformer/`; MuScriptor/Hugging Face assets are cached under
`models/huggingface/`. Model-download logs are process-global and stay outside
session directories.

## CLI

One song:

```powershell
uv run resonforge "path/to/song.opus" --gpus 0,1
```

Multiple songs with shared options:

```powershell
uv run resonforge `
  "path/to/first.opus" `
  "path/to/second.opus" `
  --song-jobs 2 `
  --gpus 0,1 `
  --skip other
```

`--song-jobs` bounds concurrent pipeline sessions and defaults to 2. All input
files share the same CLI options. A future API adapter may create distinct
`JobRequest` configurations per song without changing the pipeline service.
Explicit `--gpus` IDs are checked against the visible CUDA devices before any
job starts. Sessions may share one GPU; CUDA work from distinct model lanes is
serialized per device while the scheduler still batches compatible rows.

Useful controls:

- `--transcribers default:muscriptor.medium,piano:muscriptor.large`: shared
  default plus per-stem model overrides.
- `--only-other`: transcribe only the separated `other` stem.
- `--skip other,vocal`: omit selected stems.
- `--combine-band bass,guitar,piano`: transcribe one in-memory combined stem.
- `--adaptive-gate-stems guitar:strong`: enable adaptive preprocessing.
- `--hard-zero-stems other --hard-zero-threshold-dbfs -80`: zero quiet blocks.
- `--disable-muscriptor-anomaly-detection`: disable rolling stall/repetition
  diagnostics only.
- `--disable-muscriptor-overlap-detection`: disable low-overlap primary
  triggers only; replay and candidate verification remain active.
- `--disable-muscriptor-recovery`: disable A/B candidate generation; hard
  pitch, adaptive quality, and non-EOS safety still discard unsafe chunks.
- `--disable-muscriptor-silence-split`: transcribe each stem as one region.
- `--disable-gate-presence-filter`: do not skip globally absent stems.
- `--clean-midi`: enable destructive MIDI cleanup; default output is faithful.
  Steps: interval normalization, quiet-onset filtering
  (`--silence-threshold-dbfs`), duration clamping to the source length, and
  unsupported-retrigger merging. Onset-burst filtering is disabled pending a
  decision on removing it (`docs/deferred-work.md`).
- `--mp3-out`: publish an MP3 mix in addition to `final.mid`.
- `--no-file-output`: run through the in-memory result path without publishing
  MIDI, archives, metadata, SVG, or logs.
- `--experimental-muscriptor-scheduler-waterfall`: render a session SVG from
  the global waterfall database.

`--mix-only --mix-from-run SESSION_ID` accepts exactly one input because the
source archive is session-specific.

Anomaly detection, overlap detection, recovery, silence splitting, adaptive
completed-chunk quality, hard pitch envelopes, and non-EOS safety default on.
Only the first four are optional runtime features and expose disable flags.
Safety checks have no CLI off switch. `--clean-midi` remains independently
opt-in and never controls generation acceptance.

## Concurrency and memory

Each song is one session and remains assigned to one GPU for its MuScriptor
work. Multiple songs can share a GPU or be distributed across the configured
devices. Stems and regions from all sessions enter the same process-global,
model-aware scheduler; there is no separate CLI scheduler.

BS-RoFormer cache misses are admitted through one process-global separation
slot because its external process uses the default CUDA device. Cache hits do
not occupy this slot. This prevents concurrent cache misses from loading two
separation models onto the same GPU.

Decoded stem audio uses a process-global byte budget. Stems are admitted
lazily, converted to compact features/events/MIDI, and released after their
consumer finishes. Intermediate gated, combined, and postprocessed WAV files
are not published.

## Performance metrics

Every completed session returns and records:

- `notes_per_second`: final MIDI note-on count / end-to-end wall time.
- `generated_tokens_per_second`: all scheduler-generated token rows / wall
  time, including recovery candidates and discarded work.
- `processing_seconds_per_audio_minute`: seconds required per 60 seconds of
  source audio. Lower is better.
- `realtime_factor`: wall seconds / audio seconds. Lower is better.
- `audio_seconds_per_wall_second`: inverse real-time factor. Higher is better.

For multiple songs, the CLI also prints batch throughput using total completed
audio/notes/tokens divided by shared batch wall time. It never sums per-session
elapsed time, so overlapping jobs are measured correctly.

## Files and caches

```text
output/
  telemetry/
    resonforge.sqlite3
  <song>/
    .bs_roformer/
      <cache-hash>/
        .lock
        bass.wav
        drums.wav
        guitar.wav
        other.wav
        piano.wav
        vocals.wav
    runs/
      <YYMMDD-HHMMSS-hash>/
        final.mid
        final.mp3                         # optional
        metadata.json
        <session>-scheduler-waterfall.svg # optional
        artifacts/
          scheduler-telemetry.json.gz
          transcriber.tar.gz
        logs/
          pipeline.log.gz
          muscriptor.tar.gz
```

The separation cache key includes the source SHA-256, sample rate, and channel
layout. Cache publication is protected by an atomic OS file lock. Run output,
logs, transcriber artifacts, and waterfall ownership are session-specific.

The global SQLite waterfall contains separation, pipeline phases, model load,
condition/prefill/decode, recovery, postprocessing, and output events keyed by
session ID. A session SVG includes overlapping sessions with reduced opacity.

## Validation

```powershell
uv run pytest -q
uv run ruff check src test
git diff --check
```

MuScriptor is a git submodule and must be committed first. Then commit the
updated submodule pointer in ResonForge. Do not commit model weights, output,
benchmark-transfer data, cookies, or local downloader configuration.
