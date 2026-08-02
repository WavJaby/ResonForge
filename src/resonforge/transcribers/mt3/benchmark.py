"""Benchmark one MT3 model over multiple audio files without reloading it."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from .runtime import load_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="mt3_pytorch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[4]
    checkpoint_dir = args.checkpoint_dir or root / "models" / "mt3"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(checkpoint_dir=checkpoint_dir, service_root=root)

    started_at = datetime.now().astimezone()
    load_started = time.perf_counter()
    model = runtime.load_model(args.model, device=args.device)
    model_load_seconds = time.perf_counter() - load_started
    results: list[dict[str, object]] = []

    for index, audio_path in enumerate(args.audio, 1):
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        label = audio_path.stem.rsplit("_", 1)[-1]
        output = args.output_dir / f"{label}.{args.model}.mid"
        print(f"[{index}/{len(args.audio)}] {label}: {audio_path.name}", flush=True)
        started = time.perf_counter()
        audio, sample_rate = runtime.load_audio(str(audio_path))
        midi = model.transcribe(audio, sr=sample_rate)
        midi.save(str(output))
        elapsed = time.perf_counter() - started
        results.append(
            {
                "label": label,
                "input": str(audio_path),
                "output": str(output),
                "seconds": round(elapsed, 3),
                "audio_samples": int(len(audio)),
                "sample_rate": sample_rate,
            }
        )
        print(f"  {elapsed:.3f}s -> {output}", flush=True)

    report = {
        "started_at": started_at.isoformat(),
        "model": args.model,
        "device": args.device,
        "torch": runtime.torch.__version__,
        "gpu": (
            runtime.torch.cuda.get_device_name(0)
            if runtime.torch.cuda.is_available()
            else None
        ),
        "model_load_seconds": round(model_load_seconds, 3),
        "total_inference_seconds": round(
            sum(float(item["seconds"]) for item in results), 3
        ),
        "files": results,
    }
    report_path = args.output_dir / "benchmark.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
