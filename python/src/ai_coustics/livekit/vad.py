from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import aic_sdk
import numpy as np
from livekit import agents, rtc

from .common import _license_key, _mono_pcm16, _pcm16_to_float32
from .log import log_fields, logger

_FRAME_USERDATA_VAD_ATTRIBUTE = "ai_coustics.vad"
_SLOW_WARNING_INTERVAL = 10.0
_SLOW_INFERENCE_BACKLOG_THRESHOLD = 0.2
_INFERENCE_ERROR_LOG_INTERVAL = 10.0
_DEFAULT_SPEECH_HOLD_DURATION = 0.25
_DEFAULT_MINIMUM_SPEECH_DURATION = 0.05
_MISSING_METADATA_WARNING_FRAMES = 10


@dataclass
class VADParameters:
    """Runtime-adjustable SDK VAD parameters; ``None`` retains the current value."""

    sensitivity: float | None = None
    speech_hold_duration: float | None = None
    minimum_speech_duration: float | None = None


@dataclass(frozen=True)
class _InferenceResult:
    pcm: bytes
    sample_rate: int
    probability: float
    detected: bool
    sensitivity: float
    speech_hold_duration: float
    minimum_speech_duration: float
    inference_duration: float
    prediction_delay_samples: int


@dataclass(frozen=True)
class _FrameMetadata:
    results: tuple[_InferenceResult, ...] = ()
    error: Exception | None = None


class VADProcessor(rtc.FrameProcessor[rtc.AudioFrame]):
    """Pass-through processor that runs one SDK VAD and annotates input frames.

    Install this processor at the start of RoomIO's ``noise_cancellation`` path. Every
    :class:`VADStream` created by the owning :class:`VAD` then consumes the same immutable
    inference snapshots from ``AudioFrame.userdata`` without running the model again.
    """

    def __init__(self, *, model: aic_sdk.Model, license_key: str) -> None:
        aic_sdk.set_sdk_id(8)  # type: ignore[attr-defined]
        try:
            native_vad = aic_sdk.Vad(model, license_key)
        except Exception as error:
            raise RuntimeError(f"Failed to create ai-coustics VAD: {error}") from error

        self._model = model
        self._model_id = model.get_id()
        self._native_vad = native_vad
        self._context = native_vad.get_context()
        self._enabled = True
        self._closed = False
        self._stream_info: dict[str, str] = {}
        self._format: tuple[int, int] | None = None
        self._inference_buffer = bytearray()
        self._prediction_delay_samples = 0
        self._processing_backlog = 0.0
        self._slow_warning_active = False
        self._last_slow_warning = 0.0
        self._failed_blocks = 0
        self._last_inference_error_log = -math.inf

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if self._closed or value == self._enabled:
            return
        if value:
            self._reset()
        self._enabled = value

    @property
    def context(self) -> aic_sdk.VadContext:
        if self._closed:
            raise RuntimeError("Cannot get context from a closed ai-coustics VAD processor")
        return self._context

    def _reset(self) -> None:
        if not self._closed:
            self._context.reset()
        self._format = None
        self._inference_buffer.clear()
        self._prediction_delay_samples = 0
        self._processing_backlog = 0.0
        self._slow_warning_active = False

    def _on_stream_info_updated(
        self,
        *,
        room_name: str,
        participant_identity: str,
        publication_sid: str,
    ) -> None:
        stream_info = {
            "room_name": room_name,
            "participant_identity": participant_identity,
            "publication_sid": publication_sid,
        }
        if self._stream_info and self._stream_info != stream_info:
            self._reset()
        self._stream_info = stream_info

    def _on_stream_info_cleared(self) -> None:
        if self._stream_info:
            self._reset()
            self._stream_info = {}

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if not self._enabled:
            return frame

        try:
            results = self._process_frame(frame)
            metadata = _FrameMetadata(results=tuple(results))
        except Exception as error:
            metadata = _FrameMetadata(error=error)

        frame.userdata[_FRAME_USERDATA_VAD_ATTRIBUTE] = metadata
        return frame

    def _process_frame(self, frame: rtc.AudioFrame) -> list[_InferenceResult]:
        native_vad = self._native_vad
        context = self._context
        mono = _mono_pcm16(frame)

        if self._format is None or self._format[0] != frame.sample_rate:
            inference_block_size = self._model.get_optimal_block_size(frame.sample_rate)
            try:
                native_vad.initialize(
                    aic_sdk.ProcessorConfig(
                        sample_rate=frame.sample_rate,
                        block_size=inference_block_size,
                        variable_block_size=False,
                    )
                )
            except Exception as error:
                raise RuntimeError(
                    "ai-coustics VAD initialization failed "
                    f"(model={self._model_id}, sample_rate={frame.sample_rate}, "
                    f"block_size={inference_block_size}): {error}"
                ) from error
            self._format = (frame.sample_rate, inference_block_size)
            self._inference_buffer.clear()
            self._prediction_delay_samples = context.get_prediction_delay()
        else:
            inference_block_size = self._format[1]

        self._inference_buffer.extend(mono.tobytes())
        results: list[_InferenceResult] = []
        block_bytes = inference_block_size * np.dtype(np.int16).itemsize
        while len(self._inference_buffer) >= block_bytes:
            pcm = bytes(self._inference_buffer[:block_bytes])
            del self._inference_buffer[:block_bytes]

            started = time.perf_counter()
            try:
                native_vad.process(_pcm16_to_float32(memoryview(pcm)))
                probability = context.raw_vad_probability()
                detected = context.is_speech_detected()
            except Exception as error:
                # LiveKit does not restart a failed VAD stream. A failed block is silence,
                # so the stream continues and speech ends if the failures continue.
                probability = 0.0
                detected = False
                self._failed_blocks += 1
                failed = time.perf_counter()
                if failed - self._last_inference_error_log >= _INFERENCE_ERROR_LOG_INTERVAL:
                    logger.error(
                        "VAD: inference failed; treating the block as silence",
                        extra=self._diagnostic_fields(
                            error_type=type(error).__name__,
                            error_message=str(error),
                            failed_blocks=self._failed_blocks,
                        ),
                        exc_info=(type(error), error, error.__traceback__),
                    )
                    self._last_inference_error_log = failed
                    self._failed_blocks = 0
            completed = time.perf_counter()
            inference_duration = completed - started
            block_duration = inference_block_size / frame.sample_rate
            self._processing_backlog = max(
                0.0,
                self._processing_backlog + inference_duration - block_duration,
            )

            results.append(
                _InferenceResult(
                    pcm=pcm,
                    sample_rate=frame.sample_rate,
                    probability=probability,
                    detected=detected,
                    sensitivity=context.get_parameter(aic_sdk.VadParameter.Sensitivity),
                    speech_hold_duration=context.get_parameter(
                        aic_sdk.VadParameter.SpeechHoldDuration
                    ),
                    minimum_speech_duration=context.get_parameter(
                        aic_sdk.VadParameter.MinimumSpeechDuration
                    ),
                    inference_duration=inference_duration,
                    prediction_delay_samples=self._prediction_delay_samples,
                )
            )

            if self._processing_backlog == 0.0:
                self._slow_warning_active = False
            elif self._processing_backlog >= _SLOW_INFERENCE_BACKLOG_THRESHOLD and (
                not self._slow_warning_active
                or completed - self._last_slow_warning >= _SLOW_WARNING_INTERVAL
            ):
                self._slow_warning_active = True
                self._last_slow_warning = completed
                logger.warning(
                    "VAD: inference falling behind realtime",
                    extra=self._diagnostic_fields(
                        inference_duration=inference_duration,
                        block_duration=block_duration,
                        realtime_factor=inference_duration / block_duration,
                        processing_backlog=self._processing_backlog,
                        sample_rate=frame.sample_rate,
                        block_size=inference_block_size,
                    ),
                )

        return results

    def _diagnostic_fields(self, **fields: object) -> dict[str, object]:
        diagnostics: dict[str, object] = log_fields(
            "vad",
            model_provider="ai-coustics",
            model_name=self._model_id,
            **self._stream_info,
        )
        if self._format is not None:
            diagnostics.update(
                {
                    "sample_rate": self._format[0],
                    "block_size": self._format[1],
                }
            )
        diagnostics.update(fields)
        return diagnostics

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._enabled = False
        self._inference_buffer.clear()
        self._native_vad.terminate_session()


class VAD(agents.vad.VAD):
    """LiveKit VAD whose shared ``processor`` performs the SDK inference."""

    def __init__(
        self,
        *,
        model: aic_sdk.Model,
        license_key: str | None = None,
        vad_parameters: VADParameters | None = None,
        prefix_padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
    ) -> None:
        if prefix_padding_duration < 0.0:
            raise ValueError("prefix_padding_duration must be greater than or equal to zero")
        if max_buffered_speech <= 0.0:
            raise ValueError("max_buffered_speech must be greater than zero")

        model_sample_rate = model.get_optimal_sample_rate()
        model_block_size = model.get_optimal_block_size(model_sample_rate)
        super().__init__(
            capabilities=agents.vad.VADCapabilities(
                update_interval=model_block_size / model_sample_rate
            )
        )

        self._model_id = model.get_id()
        self._prefix_padding_duration = prefix_padding_duration
        self._max_buffered_speech = max_buffered_speech
        self._parameters = VADParameters(
            speech_hold_duration=_DEFAULT_SPEECH_HOLD_DURATION,
            minimum_speech_duration=_DEFAULT_MINIMUM_SPEECH_DURATION,
        )
        self._processor = VADProcessor(model=model, license_key=_license_key(license_key))
        self._default_speech_hold_duration = self._processor.context.get_parameter(
            aic_sdk.VadParameter.SpeechHoldDuration
        )
        self.set_parameters(self._parameters)
        if vad_parameters is not None:
            self.set_parameters(vad_parameters)

    @property
    def processor(self) -> VADProcessor:
        """The processor that must be installed in RoomIO's noise-cancellation path."""

        return self._processor

    @property
    def model(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "ai-coustics"

    @property
    def min_silence_duration(self) -> float | None:
        if self._parameters.speech_hold_duration is not None:
            return self._parameters.speech_hold_duration
        return self._default_speech_hold_duration

    def stream(self) -> VADStream:
        return VADStream(self)

    def set_parameters(self, parameters: VADParameters) -> None:
        if parameters.sensitivity is not None and self._apply_parameter(
            aic_sdk.VadParameter.Sensitivity, "sensitivity", parameters.sensitivity
        ):
            self._parameters.sensitivity = parameters.sensitivity
        if parameters.speech_hold_duration is not None and self._apply_parameter(
            aic_sdk.VadParameter.SpeechHoldDuration,
            "speech_hold_duration",
            parameters.speech_hold_duration,
        ):
            self._parameters.speech_hold_duration = parameters.speech_hold_duration
        if parameters.minimum_speech_duration is not None and self._apply_parameter(
            aic_sdk.VadParameter.MinimumSpeechDuration,
            "minimum_speech_duration",
            parameters.minimum_speech_duration,
        ):
            self._parameters.minimum_speech_duration = parameters.minimum_speech_duration

    def _apply_parameter(
        self,
        parameter: aic_sdk.VadParameter,
        name: str,
        value: float,
    ) -> bool:
        try:
            self._processor.context.set_parameter(parameter, value)
        except Exception as error:
            logger.warning(
                "VAD: parameter rejected; keeping the current value",
                extra=self._processor._diagnostic_fields(
                    parameter=name,
                    parameter_value=value,
                    error_type=type(error).__name__,
                    error_message=str(error),
                ),
                exc_info=(type(error), error, error.__traceback__),
            )
            return False
        return True


class VADStream(agents.vad.VADStream):
    def __init__(self, vad: VAD) -> None:
        self._processor = vad.processor
        self._prefix_padding_duration = vad._prefix_padding_duration
        self._max_buffered_speech = vad._max_buffered_speech
        self._missing_metadata_frames = 0
        super().__init__(vad)

    async def _main_task(self) -> None:
        try:
            await self._run_main_task()
        except Exception as error:
            logger.error(
                "VAD: stream failed",
                extra=self._processor._diagnostic_fields(
                    error_type=type(error).__name__,
                    error_message=str(error),
                ),
                exc_info=(type(error), error, error.__traceback__),
            )
            raise

    async def _run_main_task(self) -> None:
        input_sample_rate = 0
        prefix_frames: deque[rtc.AudioFrame] = deque()
        prefix_samples = 0
        speech_pcm = bytearray()
        speech_samples = 0
        speech_buffer_full = False
        speaking = False
        speech_duration = 0.0
        silence_duration = 0.0
        raw_speech_duration = 0.0
        raw_silence_duration = 0.0
        current_sample = 0
        timestamp = 0.0

        def reset_state() -> None:
            nonlocal input_sample_rate, prefix_samples, speech_samples, speech_buffer_full
            nonlocal speaking, speech_duration, silence_duration
            nonlocal raw_speech_duration, raw_silence_duration, current_sample, timestamp
            input_sample_rate = 0
            prefix_frames.clear()
            prefix_samples = 0
            speech_pcm.clear()
            speech_samples = 0
            speech_buffer_full = False
            speaking = False
            speech_duration = 0.0
            silence_duration = 0.0
            raw_speech_duration = 0.0
            raw_silence_duration = 0.0
            current_sample = 0
            timestamp = 0.0

        def update_prefix_buffer(frame: rtc.AudioFrame, result: _InferenceResult) -> None:
            nonlocal prefix_samples
            prefix_frames.append(frame)
            prefix_samples += frame.samples_per_channel
            activation_samples = max(
                frame.samples_per_channel,
                int(result.minimum_speech_duration * frame.sample_rate),
            )
            target_samples = (
                int(self._prefix_padding_duration * frame.sample_rate)
                + result.prediction_delay_samples
                + activation_samples
            )
            while len(prefix_frames) > 1 and (
                prefix_samples - prefix_frames[0].samples_per_channel >= target_samples
            ):
                prefix_samples -= prefix_frames.popleft().samples_per_channel

        def append_speech_audio(frame: rtc.AudioFrame, prediction_delay_samples: int) -> None:
            nonlocal speech_samples, speech_buffer_full
            max_samples = (
                int((self._prefix_padding_duration + self._max_buffered_speech) * frame.sample_rate)
                + prediction_delay_samples
            )
            remaining_samples = max(0, max_samples - speech_samples)
            copied_samples = min(frame.samples_per_channel, remaining_samples)
            if copied_samples:
                speech_pcm.extend(frame.data.cast("b")[: copied_samples * 2])
                speech_samples += copied_samples
            if copied_samples < frame.samples_per_channel and not speech_buffer_full:
                speech_buffer_full = True
                logger.warning(
                    "VAD: maximum buffered speech reached; ignoring further audio",
                    extra=self._processor._diagnostic_fields(
                        max_buffered_speech=self._max_buffered_speech,
                    ),
                )

        def speech_event_frames() -> list[rtc.AudioFrame]:
            if not speech_samples:
                return []
            return [
                rtc.AudioFrame(
                    data=bytes(speech_pcm),
                    sample_rate=input_sample_rate,
                    num_channels=1,
                    samples_per_channel=speech_samples,
                )
            ]

        async for input_frame in self._input_ch:
            if isinstance(input_frame, self._FlushSentinel):
                reset_state()
                continue
            if not isinstance(input_frame, rtc.AudioFrame):
                continue

            metadata = input_frame.userdata.get(_FRAME_USERDATA_VAD_ATTRIBUTE)
            if not isinstance(metadata, _FrameMetadata):
                self._missing_metadata_frames += 1
                if self._missing_metadata_frames == _MISSING_METADATA_WARNING_FRAMES:
                    logger.error(
                        "VAD: no inference metadata found; pass vad.processor as RoomIO "
                        "noise_cancellation",
                        extra=self._processor._diagnostic_fields(
                            missing_metadata_frames=self._missing_metadata_frames,
                        ),
                    )
                continue
            self._missing_metadata_frames = 0
            if metadata.error is not None:
                error = metadata.error
                cause = error.__cause__ if isinstance(error.__cause__, Exception) else error
                raise RuntimeError(str(error)) from cause

            for result in metadata.results:
                if not input_sample_rate:
                    input_sample_rate = result.sample_rate
                elif result.sample_rate != input_sample_rate:
                    logger.error(
                        "VAD: received frame with a different sample rate",
                        extra=self._processor._diagnostic_fields(
                            sample_rate=input_sample_rate,
                            received_sample_rate=result.sample_rate,
                        ),
                    )
                    continue

                inference_audio = rtc.AudioFrame(
                    data=result.pcm,
                    sample_rate=result.sample_rate,
                    num_channels=1,
                    samples_per_channel=len(result.pcm) // 2,
                )
                block_duration = inference_audio.samples_per_channel / input_sample_rate
                prediction_delay_duration = result.prediction_delay_samples / input_sample_rate
                current_sample += inference_audio.samples_per_channel
                timestamp += block_duration

                if result.probability >= result.sensitivity:
                    raw_speech_duration += block_duration
                    raw_silence_duration = 0.0
                else:
                    raw_silence_duration += block_duration
                    raw_speech_duration = 0.0
                aligned_raw_speech_duration = (
                    raw_speech_duration + prediction_delay_duration
                    if raw_speech_duration > 0.0
                    else 0.0
                )
                aligned_raw_silence_duration = (
                    raw_silence_duration + prediction_delay_duration
                    if raw_silence_duration > 0.0
                    else 0.0
                )

                if speaking:
                    speech_duration += block_duration
                else:
                    silence_duration += block_duration
                update_prefix_buffer(inference_audio, result)
                if speaking:
                    append_speech_audio(inference_audio, result.prediction_delay_samples)

                self._event_ch.send_nowait(
                    agents.vad.VADEvent(
                        type=agents.vad.VADEventType.INFERENCE_DONE,
                        samples_index=current_sample,
                        timestamp=timestamp,
                        speech_duration=speech_duration,
                        silence_duration=silence_duration,
                        frames=[inference_audio],
                        probability=result.probability,
                        inference_duration=result.inference_duration,
                        speaking=speaking,
                        raw_accumulated_silence=aligned_raw_silence_duration,
                        raw_accumulated_speech=aligned_raw_speech_duration,
                    )
                )

                if result.detected and not speaking:
                    speaking = True
                    silence_duration = 0.0
                    speech_duration = max(block_duration, aligned_raw_speech_duration)
                    speech_pcm.clear()
                    speech_samples = 0
                    speech_buffer_full = False
                    for prefix_frame in prefix_frames:
                        append_speech_audio(prefix_frame, result.prediction_delay_samples)
                    self._event_ch.send_nowait(
                        agents.vad.VADEvent(
                            type=agents.vad.VADEventType.START_OF_SPEECH,
                            samples_index=current_sample,
                            timestamp=timestamp,
                            speech_duration=speech_duration,
                            silence_duration=0.0,
                            frames=speech_event_frames(),
                            probability=result.probability,
                            inference_duration=result.inference_duration,
                            speaking=True,
                            raw_accumulated_speech=aligned_raw_speech_duration,
                        )
                    )
                # The SDK hold is a rolling-majority window, so short utterances can make
                # detected drop before a full hold interval of raw silence has elapsed.
                elif (
                    not result.detected
                    and speaking
                    and aligned_raw_silence_duration >= result.speech_hold_duration
                ):
                    speaking = False
                    silence_duration = aligned_raw_silence_duration
                    completed_speech_duration = max(0.0, speech_duration - silence_duration)
                    self._event_ch.send_nowait(
                        agents.vad.VADEvent(
                            type=agents.vad.VADEventType.END_OF_SPEECH,
                            samples_index=current_sample,
                            timestamp=timestamp,
                            speech_duration=completed_speech_duration,
                            silence_duration=silence_duration,
                            frames=speech_event_frames(),
                            probability=result.probability,
                            inference_duration=result.inference_duration,
                            speaking=False,
                            raw_accumulated_silence=aligned_raw_silence_duration,
                        )
                    )
                    speech_duration = 0.0
                    speech_pcm.clear()
                    speech_samples = 0
                    speech_buffer_full = False
