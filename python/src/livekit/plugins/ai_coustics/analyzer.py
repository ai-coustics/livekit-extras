from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import aic_sdk
import numpy as np
from opentelemetry import metrics

from livekit import rtc

from .log import log_fields, logger
from .processor import _license_key, _pcm16_to_float32

_DEFAULT_ANALYSIS_INTERVAL = 5.0
_METER = metrics.get_meter("ai-coustics-livekit-extras")
_ANALYSIS_COUNT = _METER.create_counter(
    "ai_coustics.analyzer.analysis",
    description="Number of ai-coustics buffered audio analyses",
)
_INFERENCE_DURATION = _METER.create_histogram(
    "ai_coustics.analyzer.inference_duration",
    unit="s",
    description="Duration of ai-coustics buffered audio analysis",
)
_SCORE = _METER.create_histogram(
    "ai_coustics.analyzer.score",
    description="Audio analysis score produced by ai-coustics",
)
_METRIC_BASE_ATTRIBUTES = {"model_provider": "ai-coustics"}
_RESULT_FIELDS = (
    "risk_score",
    "speaker_reverb",
    "speaker_loudness",
    "interfering_speech",
    "noise",
    "codec_degradation",
    "packet_loss",
)


@dataclass(frozen=True)
class AnalysisEvent:
    """Metadata and SDK result emitted after one successful buffered analysis."""

    result: aic_sdk.AnalysisResult
    timestamp: float
    """Unix timestamp in seconds recorded after inference completed."""

    inference_duration: float
    """Elapsed inference time in seconds."""

    sequence: int
    model_id: str
    room_name: str | None
    participant_identity: str | None
    publication_sid: str | None


class Collector(rtc.FrameProcessor[rtc.AudioFrame]):
    """Transparent LiveKit frame processor that collects audio for an :class:`Analyzer`."""

    def __init__(
        self,
        native_collector: aic_sdk.Collector,
        *,
        reset_analyzer: Callable[[], None],
        close_analyzer: Callable[[], None],
    ) -> None:
        self._collector: aic_sdk.Collector | None = native_collector
        self._reset_analyzer = reset_analyzer
        self._close_analyzer = close_analyzer
        self._format: tuple[int, int, int] | None = None
        self._stream_info: dict[str, str] = {}
        self._has_buffered_audio = False
        self._enabled = True
        self._closed = False

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
    def initialized(self) -> bool:
        return self._has_buffered_audio and self._collector is not None

    @property
    def stream_info(self) -> dict[str, str]:
        return self._stream_info.copy()

    def _on_stream_info_updated(
        self,
        *,
        room_name: str,
        participant_identity: str,
        publication_sid: str,
    ) -> None:
        self._stream_info = {
            "room_name": room_name,
            "participant_identity": participant_identity,
            "publication_sid": publication_sid,
        }
        self._reset()

    def _on_stream_info_cleared(self) -> None:
        self._stream_info = {}
        self._reset()

    def _reset(self) -> None:
        if self._closed:
            return
        self._has_buffered_audio = False
        try:
            self._reset_analyzer()
        except Exception as error:
            logger.error(
                "Analyzer: reset failed",
                extra=log_fields(
                    "analyzer",
                    error_type=type(error).__name__,
                    error_message=str(error),
                ),
                exc_info=(type(error), error, error.__traceback__),
            )

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        collector = self._collector
        if not self._enabled or collector is None:
            return frame

        try:
            stream_format = (
                frame.sample_rate,
                frame.num_channels,
                frame.samples_per_channel,
            )
            if self._format != stream_format:
                collector.initialize(
                    aic_sdk.ProcessorConfig(
                        sample_rate=frame.sample_rate,
                        block_size=frame.samples_per_channel,
                        variable_block_size=False,
                    )
                )
                self._format = stream_format

            samples = _pcm16_to_float32(frame.data)
            expected_samples = frame.samples_per_channel * frame.num_channels
            if samples.size != expected_samples:
                raise ValueError(
                    f"AudioFrame contains {samples.size} samples, expected {expected_samples}"
                )

            channels = samples.reshape(frame.samples_per_channel, frame.num_channels)
            mono = (
                channels[:, 0].copy()
                if frame.num_channels == 1
                else channels.mean(axis=1, dtype=np.float32)
            )
            collector.buffer(mono)
            self._has_buffered_audio = True
        except Exception:
            logger.exception(
                "Collector: failed; passing audio through",
                extra=log_fields("collector", **self._stream_info),
            )

        return frame

    def _detach(self) -> None:
        self._closed = True
        self._enabled = False
        self._format = None
        self._stream_info = {}
        self._has_buffered_audio = False
        self._collector = None

    def _close(self) -> None:
        if self._closed:
            return
        self._detach()
        self._close_analyzer()


class Analyzer(rtc.EventEmitter[Literal["analysis_result"]]):
    """Periodically analyzes audio buffered by a transparent LiveKit Collector."""

    def __init__(
        self,
        *,
        model: aic_sdk.Model,
        license_key: str | None = None,
        analysis_interval: float = _DEFAULT_ANALYSIS_INTERVAL,
        enable_metrics: bool = True,
    ) -> None:
        super().__init__()
        if not math.isfinite(analysis_interval) or analysis_interval <= 0.0:
            raise ValueError("analysis_interval must be a finite value greater than zero")

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError(
                "ai-coustics Analyzer must be created while an event loop is running"
            ) from None

        resolved_license_key = _license_key(license_key)
        aic_sdk.set_sdk_id(8)  # type: ignore[attr-defined]
        try:
            model_id = model.get_id()
            native_collector, native_analyzer = aic_sdk.analyzer_pair(model, resolved_license_key)
        except Exception as error:
            raise RuntimeError(f"Failed to create ai-coustics Analyzer: {error}") from error

        self._native_analyzer: aic_sdk.Analyzer | None = native_analyzer
        self._model_id = model_id
        self._analysis_interval = analysis_interval
        self._enable_metrics = enable_metrics
        self._sequence = 0
        self._closed = False
        self._close_event = asyncio.Event()
        self._close_task: asyncio.Task[None] | None = None
        self.collector = Collector(
            native_collector,
            reset_analyzer=native_analyzer.reset,
            close_analyzer=self._request_close,
        )
        self._task = loop.create_task(self._analysis_loop(), name="ai-coustics-analyzer")

    async def _terminate_session(self) -> None:
        native_analyzer = self._native_analyzer
        self._native_analyzer = None
        if native_analyzer is None:
            return
        try:
            await asyncio.to_thread(native_analyzer.terminate_session)
        except Exception:
            logger.exception(
                "Analyzer: session termination failed",
                extra=log_fields("analyzer", model_name=self._model_id),
            )

    async def _analyze_once(self, native_analyzer: aic_sdk.Analyzer) -> None:
        started = time.perf_counter()
        try:
            result = await asyncio.to_thread(native_analyzer.analyze_buffered)
        except Exception:
            inference_duration = time.perf_counter() - started
            self._record_analysis_metrics(inference_duration, status="error")
            raise

        inference_duration = time.perf_counter() - started
        self._sequence += 1
        stream_info = self.collector.stream_info
        event = AnalysisEvent(
            result=result,
            timestamp=time.time(),
            inference_duration=inference_duration,
            sequence=self._sequence,
            model_id=self._model_id,
            room_name=stream_info.get("room_name"),
            participant_identity=stream_info.get("participant_identity"),
            publication_sid=stream_info.get("publication_sid"),
        )

        self._record_analysis_metrics(inference_duration, status="ok", result=result)

        try:
            self.emit("analysis_result", event)
        except Exception:
            logger.exception(
                "Analyzer: result event emission failed",
                extra=log_fields(
                    "analyzer",
                    model_name=self._model_id,
                    sequence=self._sequence,
                    **stream_info,
                ),
            )

    def _record_analysis_metrics(
        self,
        inference_duration: float,
        *,
        status: str,
        result: aic_sdk.AnalysisResult | None = None,
    ) -> None:
        if not self._enable_metrics:
            return

        try:
            attributes = {**_METRIC_BASE_ATTRIBUTES, "status": status}
            _ANALYSIS_COUNT.add(1, attributes=attributes)
            _INFERENCE_DURATION.record(inference_duration, attributes=attributes)
            if result is not None:
                for score_name in _RESULT_FIELDS:
                    _SCORE.record(
                        getattr(result, score_name),
                        attributes={**_METRIC_BASE_ATTRIBUTES, "score.name": score_name},
                    )
        except Exception:
            logger.exception(
                "Analyzer: metrics recording failed",
                extra=log_fields(
                    "analyzer",
                    model_name=self._model_id,
                    status=status,
                ),
            )

    async def _analysis_loop(self) -> None:
        native_analyzer = self._native_analyzer
        assert native_analyzer is not None
        try:
            while not self._close_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._close_event.wait(), timeout=self._analysis_interval
                    )
                    break
                except asyncio.TimeoutError:
                    pass
                if not self.collector.initialized:
                    continue
                try:
                    await self._analyze_once(native_analyzer)
                except Exception:
                    logger.exception(
                        "Analyzer: buffered audio analysis failed",
                        extra=log_fields(
                            "analyzer",
                            model_name=self._model_id,
                            **self.collector.stream_info,
                        ),
                    )
        finally:
            await self._terminate_session()

    async def _finish_close(self) -> None:
        await self._task
        # Keep an idempotent fallback in case the loop exits before entering its try/finally.
        await self._terminate_session()

    def _request_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.collector._detach()
        self._close_event.set()
        self._close_task = asyncio.create_task(
            self._finish_close(), name="ai-coustics-analyzer-close"
        )

    async def aclose(self) -> None:
        """Stop scheduled analysis and terminate the SDK session."""

        self._request_close()
        assert self._close_task is not None
        await self._close_task
