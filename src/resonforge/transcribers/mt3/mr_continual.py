"""MR-MT3 adapter with segment memory across sequential chunks."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path

import torch
from mt3_infer.adapters.mr_mt3 import MRMT3Adapter
from mt3_infer.exceptions import CheckpointError, InferenceError
from mt3_infer.models.mr_mt3.t5 import (
    T5ForConditionalGeneration,
    T5Stack,
)
from mt3_infer.utils.framework import check_torch_version, get_device
from torch import nn
from torch.nn import functional as F
from transformers import T5Config

from resonforge.transcribers.mt3.runtime import invalid_midi_programs

SEGMENT_MEMORY_LENGTH = 64
TIE_VOCAB_TOKEN = 1134
MAX_DECODE_LENGTH = 1024


class MRMT3ContinualModel(T5ForConditionalGeneration):
    """Baseline MT3 plus the paper's one-layer segment-memory encoder."""

    def __init__(
        self,
        config: T5Config,
        *,
        segmem_length: int = SEGMENT_MEMORY_LENGTH,
    ) -> None:
        super().__init__(config)
        self.segmem_proj = nn.Linear(
            self.model_dim,
            self.model_dim,
            bias=False,
        )
        segmem_config = copy.deepcopy(config)
        segmem_config.is_decoder = False
        segmem_config.use_cache = False
        segmem_config.is_encoder_decoder = False
        segmem_config.num_layers = 1
        segmem_config.dropout_rate = 0
        self.segmem_encoder = T5Stack(
            segmem_config,
            self.segmem_proj,
            "segmem",
        )
        self.segmem_length = segmem_length

    @torch.inference_mode()
    def generate_contiguous(
        self,
        inputs: torch.Tensor,
        *,
        forbidden_token_ids: Iterable[int] = (),
        max_length: int = MAX_DECODE_LENGTH,
    ) -> torch.Tensor:
        """Greedily decode chunks in order with previous-token memory."""
        encoded = self.encoder(
            inputs_embeds=self.proj(inputs),
            return_dict=True,
        ).last_hidden_state
        forbidden = tuple(sorted(set(int(token) for token in forbidden_token_ids)))
        previous_tokens: torch.Tensor | None = None
        chunks: list[torch.Tensor] = []

        for chunk_index in range(encoded.shape[0]):
            decoder_tokens = torch.zeros(
                (1, 1),
                dtype=torch.long,
                device=inputs.device,
            )
            if previous_tokens is None:
                previous_tokens = torch.zeros(
                    (1, max_length),
                    dtype=torch.long,
                    device=inputs.device,
                )
                previous_tokens[0, 0] = TIE_VOCAB_TOKEN
                previous_tokens[0, 1] = self.config.eos_token_id

            memory = self.segmem_encoder(
                self.decoder_embed_tokens(previous_tokens)
            )[0][:, : self.segmem_length, :]
            context = torch.cat(
                (encoded[chunk_index].unsqueeze(0), memory),
                dim=1,
            )

            for _ in range(max_length - 1):
                decoded = self.decoder(
                    input_ids=decoder_tokens,
                    encoder_hidden_states=context,
                    return_dict=True,
                )[0]
                logits = self.lm_head(decoded[:, -1, :])
                if forbidden:
                    logits[:, forbidden] = -torch.inf
                next_token = torch.argmax(logits, dim=-1)
                decoder_tokens = torch.cat(
                    (decoder_tokens, next_token.unsqueeze(1)),
                    dim=1,
                )
                if next_token.item() == self.config.eos_token_id:
                    break

            decoder_tokens = F.pad(
                decoder_tokens,
                (0, max_length - decoder_tokens.shape[1]),
                value=self.config.pad_token_id,
            )
            chunks.append(decoder_tokens)
            previous_tokens = decoder_tokens

        return torch.cat(chunks, dim=0)


def _checkpoint_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise CheckpointError("MR-MT3 checkpoint is not a state dictionary")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise CheckpointError("MR-MT3 checkpoint has no usable state_dict")

    tensor_state = {
        str(key): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    for prefix in ("model.", "module."):
        if tensor_state and all(key.startswith(prefix) for key in tensor_state):
            tensor_state = {
                key[len(prefix):]: value
                for key, value in tensor_state.items()
            }
    return tensor_state


class MRMT3ContinualAdapter(MRMT3Adapter):
    """`mt3_infer`-compatible adapter with real MR-MT3 memory inference."""

    def __init__(self, *, valid_programs: Iterable[int] | None = None) -> None:
        super().__init__()
        self.valid_programs = (
            None
            if valid_programs is None
            else frozenset(int(program) for program in valid_programs)
        )

    def load_model(self, checkpoint_path: str, device: str = "auto") -> None:
        check_torch_version()
        self.device_str = get_device(device)
        path = Path(checkpoint_path)
        if not path.is_file():
            raise CheckpointError(f"MR-MT3 checkpoint not found: {path}")
        if path.stat().st_size < 500_000_000:
            raise CheckpointError(
                f"{path} is too small to contain the MR-MT3 memory weights; "
                "the baseline mt3.pth is not a valid substitute"
            )

        try:
            config = T5Config.from_dict(self.model_config)
            self.model = MRMT3ContinualModel(config)
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            state = _checkpoint_state_dict(checkpoint)
            incompatible = self.model.load_state_dict(state, strict=False)
            missing_memory = [
                key
                for key in incompatible.missing_keys
                if key.startswith(("segmem_encoder.", "segmem_proj."))
            ]
            if missing_memory:
                raise CheckpointError(
                    "checkpoint is missing segment-memory weights: "
                    + ", ".join(missing_memory[:5])
                )
            self.model.to(self.device_str)
            self.model.eval()
            self._initialize_vocab()
            self._model_loaded = True
        except CheckpointError:
            raise
        except Exception as error:
            raise CheckpointError(
                f"failed to load MR-MT3 checkpoint {path}: {error}"
            ) from error

    def _forbidden_program_tokens(self) -> tuple[int, ...]:
        if self.valid_programs is None:
            return ()
        invalid_programs = invalid_midi_programs(self.valid_programs)
        min_program_id, max_program_id = self.codec.event_type_range("program")
        codec_ids = [
            min_program_id + program
            for program in invalid_programs
            if min_program_id + program <= max_program_id
        ]
        # MT3 vocabulary reserves PAD/EOS/UNK before codec event IDs.
        return tuple(codec_id + 3 for codec_id in codec_ids)

    @torch.inference_mode()
    def forward(self, features):
        inputs = features["inputs"].to(self.device_str)
        try:
            outputs = self.model.generate_contiguous(
                inputs,
                forbidden_token_ids=self._forbidden_program_tokens(),
            )
            after_eos = torch.cumsum(
                (outputs == self.model.config.eos_token_id).float(),
                dim=-1,
            )
            outputs = outputs - 3
            outputs = torch.where(after_eos.bool(), -1, outputs)
            outputs = outputs[:, 1:]
            return {
                "tokens": outputs.cpu().numpy(),
                "frame_times": features["frame_times"],
            }
        except Exception as error:
            raise InferenceError(
                f"MR-MT3 continual forward pass failed: {error}"
            ) from error
