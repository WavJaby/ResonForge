"""Pure SVG rendering for scheduler event traces."""

from __future__ import annotations

import colorsys
import html

from resonforge.scheduler.events import SchedulerEvent


def _group_color(hue: int, lightness: float) -> str:
    red, green, blue = colorsys.hls_to_rgb(hue / 360, lightness, 0.78)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


ACTION_COLORS = {
    "separation": _group_color(42, 0.48),
    "separation_cache_hit": _group_color(42, 0.68),
    "stem_selection": _group_color(95, 0.68),
    "presence_analysis": _group_color(125, 0.68),
    "transcription": _group_color(205, 0.42),
    "midi_postprocess": _group_color(315, 0.68),
    "output": _group_color(350, 0.68),
    "cpu_planning": _group_color(188, 0.72),
    "audio_prepare": _group_color(175, 0.72),
    "audio_features": _group_color(160, 0.72),
    "tempo_analysis": _group_color(55, 0.72),
    "region_plan": _group_color(110, 0.72),
    "preparation_queue_wait": _group_color(190, 0.52),
    "producer_queue_wait": _group_color(195, 0.52),
    "cpu_region_prepare": _group_color(145, 0.72),
    "cpu_producer": _group_color(215, 0.72),
    "cpu_verify": _group_color(25, 0.72),
    "model_load": _group_color(215, 0.42),
    "model_activate": _group_color(270, 0.42),
    "capacity_plan": _group_color(185, 0.42),
    "capacity_resize": _group_color(165, 0.42),
    "session_init": _group_color(135, 0.62),
    "session_setup": _group_color(85, 0.62),
    "session_prefill": _group_color(275, 0.62),
    "session_arena_init": _group_color(48, 0.62),
    "session_first_decode": _group_color(205, 0.62),
    "decode": _group_color(220, 0.52),
    "admit": _group_color(35, 0.52),
    "prefill": _group_color(265, 0.52),
    "checkpoint": _group_color(330, 0.82),
    "preempt_condition": _group_color(300, 0.82),
    "preempt_restore": _group_color(310, 0.82),
    "resume": _group_color(175, 0.82),
    "restore": _group_color(150, 0.82),
    "discard": _group_color(5, 0.82),
}

_LEGEND_GROUPS = (
    ("PIPELINE", "#1e293b", ("separation", "separation_cache_hit", "stem_selection", "presence_analysis", "transcription", "midi_postprocess", "output")),
    ("CPU", "#1e293b", ("audio_prepare", "audio_features", "tempo_analysis", "region_plan", "preparation_queue_wait", "producer_queue_wait", "cpu_planning", "cpu_region_prepare", "cpu_producer", "cpu_verify")),
    (
        "MODEL",
        "#1e293b",
        ("model_load", "model_activate", "capacity_plan", "capacity_resize"),
    ),
    ("SESSION", "#1e293b", ("session_init", "session_setup", "session_prefill", "session_arena_init", "session_first_decode")),
    ("GENERATION", "#1e293b", ("admit", "prefill", "decode")),
    (
        "CONTROL",
        "#1e293b",
        ("checkpoint", "preempt_condition", "preempt_restore", "resume",
         "restore", "discard"),
    ),
)

_SCHEDULER_ACTIONS = frozenset(
    action
    for group, _background, actions in _LEGEND_GROUPS
    if group in {"MODEL", "SESSION", "GENERATION", "CONTROL"}
    for action in actions
)

_PIPELINE_ACTIONS = frozenset(_LEGEND_GROUPS[0][2])
_MODEL_ACTIONS = frozenset(
    {"model_load", "model_activate", "capacity_plan", "capacity_resize"}
)


def render_scheduler_waterfall_svg(events: list[SchedulerEvent]) -> str:
    duration = max((float(event["end_seconds"]) for event in events), default=1.0)
    used_actions = {str(event.get("action", "")) for event in events}
    legend_groups = tuple(
        (group, background, tuple(action for action in actions if action in used_actions))
        for group, background, actions in _LEGEND_GROUPS
        if any(action in used_actions for action in actions)
    )
    model_devices = _model_devices(events)
    resource_lanes = list(
        dict.fromkeys(
            _resource_lane_key(event)
            for event in events
            if event.get("lane") and not _is_scheduler_event(event)
        )
    )
    scheduler_lanes = _scheduler_lanes(events, model_devices)
    lane_keys = [("resource", *lane) for lane in resource_lanes] + [
        ("scheduler", *lane) for lane in scheduler_lanes
    ]
    lanes = {key: index for index, key in enumerate(lane_keys)}
    width, left, lane_height = 1600, 170, 20
    legend_start_y = 52
    top = 70 + len(legend_groups) * 24
    plot_width = width - left - 25
    height = top + max(1, len(lane_keys)) * lane_height + 62
    max_active = max(
        (int(event.get("active_after", 0)) for event in events), default=0
    )
    max_occupied = max(
        (int(event.get("occupied_after", 0)) for event in events), default=0
    )
    max_displaced = max(
        (int(event.get("displaced_count", 0)) for event in events), default=0
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#101827"/>',
        '<style>text{font:12px ui-monospace,Consolas,monospace;fill:#d1d5db}.small{font-size:10px;fill:#9ca3af}</style>',
        f'<text x="20" y="24">Scheduler waterfall: {len(events)} turns, {duration:.3f}s, max active {max_active}, max occupied {max_occupied}, max displaced {max_displaced}</text>',
    ]

    def opacity(event: SchedulerEvent, slot: dict[str, object] | None = None) -> str:
        owner = event.get("view_owner_session_id")
        if not owner:
            return "1"
        selected = (
            slot.get("pipeline_session_id")
            if slot is not None
            else event.get("pipeline_session_id")
        )
        participants = event.get("pipeline_session_ids", [])
        return "1" if selected == owner or owner in participants else "0.18"
    for row, (group, background, actions) in enumerate(legend_groups):
        y = legend_start_y + row * 24
        parts.append(
            f'<rect x="12" y="{y - 13}" width="{width - 24}" height="21" rx="3" fill="{background}"/>'
        )
        parts.append(f'<text x="20" y="{y + 2}" class="small">{group}</text>')
        legend_x = 104
        for action in actions:
            color = ACTION_COLORS[action]
            parts.extend(
                (
                    f'<rect x="{legend_x}" y="{y - 9}" width="11" height="11" rx="1" fill="{color}" stroke="#d1d5db" stroke-width="0.5"/>',
                    f'<text x="{legend_x + 17}" y="{y + 1}" class="small">{action}</text>',
                )
            )
            legend_x += 42 + len(action) * 7
    for second in range(0, int(duration) + 1, max(1, int(duration // 12) or 1)):
        x = left + second / duration * plot_width
        parts.extend(
            (
                f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{height - 35}" stroke="#263244"/>',
                f'<text x="{x:.1f}" y="{top - 14}" class="small">{second}s</text>',
            )
        )
    for key, lane in lanes.items():
        y = top + lane * lane_height
        kind, *fields = key
        label = (
            f"{fields[0]}: {fields[1]}"
            if kind == "resource"
            else " ".join(field for field in fields if field)
        )
        parts.append(
            f'<text x="10" y="{y + 12}" class="small">{html.escape(label)}</text>'
        )
        parts.append(
            f'<line x1="{left}" y1="{y + 15}" x2="{width - 25}" y2="{y + 15}" stroke="#1f2937"/>'
        )
    for event in events:
        start, end = float(event["start_seconds"]), float(event["end_seconds"])
        x = left + start / duration * plot_width
        bar_width = max(1.0, (end - start) / duration * plot_width)
        color = ACTION_COLORS.get(str(event["action"]), "#64748b")
        batch_size = _event_batch_size(event)
        scheduler_event = _is_scheduler_event(event)
        if scheduler_event:
            device, model = _scheduler_identity(event, model_devices)
            batch_label = (
                "B-"
                if str(event.get("action", "")) in _MODEL_ACTIONS
                else f"B{batch_size}"
            )
            if str(event.get("action", "")) in {
                "capacity_plan",
                "capacity_resize",
            }:
                title = html.escape(
                    f"{device} {model} {event['action']} "
                    f"C{event.get('capacity', 0)}->"
                    f"T{event.get('target_capacity', 0)}"
                )
            else:
                title = html.escape(
                    f'{device} {model} {event["action"]} {batch_label}'
                    f'{_memory_title_suffix(event)}'
                )
        else:
            title = html.escape(str(event["action"]))
        if event.get("lane") and not scheduler_event:
            lane = lanes.get(("resource", *_resource_lane_key(event)))
            if lane is not None:
                y = top + lane * lane_height + 2
                parts.append(
                    f'<rect x="{x:.1f}" y="{y}" width="{bar_width:.1f}" height="12" fill="{color}" opacity="{opacity(event)}"><title>{title}</title></rect>'
                )
            continue
        if scheduler_event:
            for scheduler_key in _scheduler_lane_keys(event, model_devices):
                lane = lanes.get(("scheduler", *scheduler_key))
                if lane is not None:
                    y = top + lane * lane_height + 2
                    parts.append(
                        f'<rect x="{x:.1f}" y="{y}" width="{bar_width:.1f}" height="12" fill="{color}" opacity="{opacity(event)}"><title>{title}</title></rect>'
                    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _model_devices(events: list[SchedulerEvent]) -> dict[str, str]:
    candidates: dict[str, set[str]] = {}
    for event in events:
        model = str(event.get("model", ""))
        if not model:
            continue
        device = str(event.get("target_device", ""))
        lane = str(event.get("lane", ""))
        if not device and " model " in lane:
            device = lane.partition(" model ")[0]
        if device:
            candidates.setdefault(model, set()).add(device)
    return {
        model: next(iter(devices))
        for model, devices in candidates.items()
        if len(devices) == 1
    }


def _event_batch_size(event: SchedulerEvent) -> int:
    action = str(event.get("action", ""))
    before = event.get("slots_before", [])
    after = event.get("slots_after", [])
    if action == "decode":
        return max(1, int(event.get("active_before", 0)))
    if action in {
        "checkpoint",
        "preempt_condition",
        "preempt_restore",
        "resume",
        "restore",
        "discard",
    }:
        return 1
    before_sequences = {
        slot.get("sequence") for slot in before if slot.get("sequence") is not None
    }
    added = {
        slot.get("sequence") for slot in after if slot.get("sequence") is not None
    } - before_sequences
    if added:
        return len(added)
    return max(
        1,
        int(event.get("active_before", 0)),
        int(event.get("active_after", 0)),
    )


def _resource_lane_key(event: SchedulerEvent) -> tuple[str, str]:
    resource = str(event.get("resource", "cpu"))
    lane = str(event["lane"])
    action = str(event.get("action", ""))
    if action not in _PIPELINE_ACTIONS:
        return resource, lane
    session_lane, separator, phase = lane.rpartition(" / ")
    if not separator or phase not in _PIPELINE_ACTIONS:
        session_lane = lane
    return "pipeline", session_lane


def _scheduler_identity(
    event: SchedulerEvent,
    model_devices: dict[str, str],
) -> tuple[str, str]:
    model = str(event.get("model", "unknown"))
    device = str(event.get("target_device", "")) or model_devices.get(model, "gpu")
    return device, model


def _scheduler_lane_keys(
    event: SchedulerEvent,
    model_devices: dict[str, str],
) -> tuple[tuple[str, str, str], ...]:
    device, model = _scheduler_identity(event, model_devices)
    action = str(event.get("action", ""))
    if action in _MODEL_ACTIONS:
        return ((device, "", "MODEL"),)
    return tuple(
        (device, model, f"B{slot}")
        for slot in range(1, _event_batch_size(event) + 1)
    )


def _scheduler_lanes(
    events: list[SchedulerEvent],
    model_devices: dict[str, str],
) -> list[tuple[str, str, str]]:
    lifecycle_devices: dict[str, None] = {}
    max_widths: dict[tuple[str, str], int] = {}
    for event in events:
        if not _is_scheduler_event(event):
            continue
        device, model = _scheduler_identity(event, model_devices)
        if str(event.get("action", "")) in _MODEL_ACTIONS:
            lifecycle_devices.setdefault(device, None)
            continue
        identity = (device, model)
        max_widths[identity] = max(
            max_widths.get(identity, 0),
            _event_batch_size(event),
        )
    lanes = [(device, "", "MODEL") for device in lifecycle_devices]
    for (device, model), max_width in max_widths.items():
        lanes.extend(
            (device, model, f"B{slot}") for slot in range(1, max_width + 1)
        )
    return lanes


def _is_scheduler_event(event: SchedulerEvent) -> bool:
    return bool(event.get("model")) and str(event.get("action", "")) in (
        _SCHEDULER_ACTIONS
    )


def _format_mib(value: int) -> str:
    return f"{value / 1024**2:.1f} MiB"


def _memory_title_suffix(event: SchedulerEvent) -> str:
    if not any(
        field in event
        for field in ("resident_bytes", "owned_total_bytes")
    ):
        return ""
    return (
        f" | resident {_format_mib(int(event.get('resident_bytes', 0)))}"
        f", displaced {int(event.get('displaced_count', 0))}"
        f", owned {_format_mib(int(event.get('owned_total_bytes', 0)))}"
    )
