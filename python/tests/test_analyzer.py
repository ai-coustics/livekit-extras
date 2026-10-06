from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import aic_sdk
import numpy as np
import pytest
from livekit import rtc

from ai_coustics.livekit import AnalysisEvent, Analyzer, Collector
from ai_coustics.livekit.analyzer import _RESULT_FIELDS


class FakeModel:
    def get_id(self) -> str:
        return "analysis-test-model"


@dataclass
class FakeResult:
    risk_score: float = 0.1
    speaker_reverb: float = 0.2
    speaker_loudness: float = 0.3
    interfering_speech: float = 0.4
    noise: float = 0.5
    codec_degradation: float = 0.6
    packet_loss: float = 0.7


class FakeCollector:
    def __init__(self) -> None:
        self.inits: list[tuple[int, int, bool]] = []
        self.blocks: list[np.ndarray] = []
        self.error: Exception | None = None

    def initialize(self, config: aic_sdk.ProcessorConfig) -> None:
        self.inits.append((config.sample_rate, config.block_size, config.variable_block_size))

    def buffer(self, block: np.ndarray) -> None:
        if self.error is not None:
            raise self.error
        self.blocks.append(block.copy())


class FakeNativeAnalyzer:
    def __init__(self) -> None:
        self.analyze_calls = 0
        self.reset_calls = 0
        self.terminate_calls = 0
        self.result = FakeResult()
        self.error: Exception | None = None

    def analyze_buffered(self) -> FakeResult:
        self.analyze_calls += 1
        if self.error is not None:
            raise self.error
        return self.result

    def reset(self) -> None:
        self.reset_calls += 1

    def terminate_session(self) -> None:
        self.terminate_calls += 1


@pytest.fixture
def fake_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakeCollector, FakeNativeAnalyzer, list[int]]:
    collector = FakeCollector()
    analyzer = FakeNativeAnalyzer()
    sdk_ids: list[int] = []
    monkeypatch.setattr("ai_coustics.livekit.analyzer.aic_sdk.set_sdk_id", sdk_ids.append)
    monkeypatch.setattr(
        "ai_coustics.livekit.analyzer.aic_sdk.analyzer_pair",
        lambda model, license_key: (collector, analyzer),
    )

    async def run_immediately(function: Callable[..., object], /, *args: object) -> object:
        return function(*args)

    monkeypatch.setattr("ai_coustics.livekit.analyzer.asyncio.to_thread", run_immediately)
    return collector, analyzer, sdk_ids


class FakeInstrument:
    def __init__(self) -> None:
        self.measurements: list[tuple[float, dict[str, str]]] = []
        self.error: Exception | None = None

    def add(self, value: float, *, attributes: dict[str, str]) -> None:
        if self.error is not None:
            raise self.error
        self.measurements.append((value, attributes))

    def record(self, value: float, *, attributes: dict[str, str]) -> None:
        if self.error is not None:
            raise self.error
        self.measurements.append((value, attributes))


@pytest.fixture(autouse=True)
def metric_instruments(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeInstrument]:
    instruments = {
        "analysis": FakeInstrument(),
        "duration": FakeInstrument(),
        "score": FakeInstrument(),
    }
    monkeypatch.setattr("ai_coustics.livekit.analyzer._ANALYSIS_COUNT", instruments["analysis"])
    monkeypatch.setattr("ai_coustics.livekit.analyzer._INFERENCE_DURATION", instruments["duration"])
    monkeypatch.setattr("ai_coustics.livekit.analyzer._SCORE", instruments["score"])
    return instruments


def make_frame(*, channels: int = 1) -> rtc.AudioFrame:
    mono = np.array([32767, -32768, 16384, -16384], dtype=np.int16)
    data = np.repeat(mono, channels)
    return rtc.AudioFrame(
        data=data.tobytes(),
        sample_rate=16000,
        num_channels=channels,
        samples_per_channel=4,
    )


async def wait_until(condition: Callable[[], bool]) -> None:
    for _ in range(200):
        if condition():
            return
        await asyncio.sleep(0.005)
    pytest.fail("condition was not met within one second")


def test_metric_fields_exist_on_installed_sdk_analysis_result() -> None:
    missing_fields = [
        field for field in _RESULT_FIELDS if not hasattr(aic_sdk.AnalysisResult, field)
    ]

    assert not missing_fields, f"Analyzer metric fields missing from aic-sdk: {missing_fields}"


@pytest.mark.asyncio
async def test_collector_is_transparent_frame_processor_and_downmixes(
    fake_sdk: tuple[FakeCollector, FakeNativeAnalyzer, list[int]],
) -> None:
    native_collector, native_analyzer, sdk_ids = fake_sdk
    analyzer = Analyzer(
        model=cast(aic_sdk.Model, FakeModel()),
        license_key="test-license",
        analysis_interval=60.0,
    )

    assert isinstance(analyzer.collector, Collector)
    assert isinstance(analyzer.collector, rtc.FrameProcessor)
    assert sdk_ids == [8]

    frame = make_frame(channels=2)
    output = analyzer.collector._process(frame)

    assert output is frame
    assert native_collector.inits == [(16000, 4, False)]
    np.testing.assert_allclose(
        native_collector.blocks[0],
        np.array([32767, -32768, 16384, -16384], dtype=np.float32) / 32768.0,
    )

    analyzer.collector._on_stream_info_cleared()
    assert native_analyzer.reset_calls == 1
    await asyncio.wait_for(analyzer.aclose(), timeout=1.0)
    assert native_analyzer.terminate_calls == 1


@pytest.mark.asyncio
async def test_analyzes_on_interval_and_emits_without_logging_result(
    fake_sdk: tuple[FakeCollector, FakeNativeAnalyzer, list[int]],
    caplog: pytest.LogCaptureFixture,
    metric_instruments: dict[str, FakeInstrument],
) -> None:
    _, native_analyzer, _ = fake_sdk
    analyzer = Analyzer(
        model=cast(aic_sdk.Model, FakeModel()),
        license_key="test-license",
        analysis_interval=0.01,
    )
    events: list[AnalysisEvent] = []
    analyzer.on("analysis_result", events.append)
    analyzer.collector._on_stream_info_updated(
        room_name="test-room",
        participant_identity="test-participant",
        publication_sid="TR_test",
    )
    analyzer.collector._process(make_frame())

    with caplog.at_level(logging.INFO, logger="ai_coustics.livekit"):
        for _ in range(20):
            if native_analyzer.analyze_calls:
                break
            await asyncio.sleep(0.005)

    await asyncio.wait_for(analyzer.aclose(), timeout=1.0)
    assert native_analyzer.analyze_calls >= 1
    assert not any(record.message == "ai-coustics analysis result" for record in caplog.records)

    event = events[0]
    assert event.result is native_analyzer.result
    assert event.sequence == 1
    assert event.model_id == "analysis-test-model"
    assert event.room_name == "test-room"
    assert event.participant_identity == "test-participant"
    assert event.publication_sid == "TR_test"
    assert event.timestamp > 0
    assert event.inference_duration >= 0

    assert metric_instruments["analysis"].measurements[0] == (
        1,
        {"model_provider": "ai-coustics", "status": "ok"},
    )
    assert len(metric_instruments["duration"].measurements) >= 1
    assert len(metric_instruments["score"].measurements) >= 7
    assert metric_instruments["score"].measurements[0] == (
        0.1,
        {"model_provider": "ai-coustics", "score.name": "risk_score"},
    )


@pytest.mark.asyncio
async def test_can_disable_metrics(
    fake_sdk: tuple[FakeCollector, FakeNativeAnalyzer, list[int]],
    metric_instruments: dict[str, FakeInstrument],
) -> None:
    analyzer = Analyzer(
        model=cast(aic_sdk.Model, FakeModel()),
        license_key="test-license",
        analysis_interval=0.01,
        enable_metrics=False,
    )
    analyzer.collector._process(make_frame())

    for _ in range(20):
        if fake_sdk[1].analyze_calls:
            break
        await asyncio.sleep(0.005)

    await analyzer.aclose()
    assert all(not instrument.measurements for instrument in metric_instruments.values())


@pytest.mark.asyncio
async def test_records_failed_analysis_metrics(
    fake_sdk: tuple[FakeCollector, FakeNativeAnalyzer, list[int]],
    metric_instruments: dict[str, FakeInstrument],
) -> None:
    native_analyzer = fake_sdk[1]
    native_analyzer.error = RuntimeError("analysis failed")
    analyzer = Analyzer(
        model=cast(aic_sdk.Model, FakeModel()),
        license_key="test-license",
        analysis_interval=0.01,
    )
    analyzer.collector._process(make_frame())

    for _ in range(20):
        if native_analyzer.analyze_calls:
            break
        await asyncio.sleep(0.005)

    await analyzer.aclose()
    assert metric_instruments["analysis"].measurements[0] == (
        1,
        {"model_provider": "ai-coustics", "status": "error"},
    )
    assert not metric_instruments["score"].measurements


@pytest.mark.asyncio
async def test_room_closing_collector_stops_analyzer(
    fake_sdk: tuple[FakeCollector, FakeNativeAnalyzer, list[int]],
) -> None:
    _, native_analyzer, _ = fake_sdk
    analyzer = Analyzer(
        model=cast(aic_sdk.Model, FakeModel()),
        license_key="test-license",
        analysis_interval=60.0,
    )

    analyzer.collector._close()
    await analyzer.aclose()

    assert not analyzer.collector.enabled
    assert native_analyzer.terminate_calls == 1


@pytest.mark.asyncio
async def test_default_analysis_interval_is_five_seconds(
    fake_sdk: tuple[FakeCollector, FakeNativeAnalyzer, list[int]],
) -> None:
    analyzer = Analyzer(
        model=cast(aic_sdk.Model, FakeModel()),
        license_key="test-license",
    )

    assert analyzer._analysis_interval == 5.0
    await analyzer.aclose()


@pytest.mark.parametrize("interval", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_invalid_analysis_interval(interval: float) -> None:
    with pytest.raises(ValueError, match="analysis_interval"):
        Analyzer(
            model=cast(aic_sdk.Model, FakeModel()),
            license_key="test-license",
            analysis_interval=interval,
        )


def test_requires_running_event_loop() -> None:
    with pytest.raises(RuntimeError, match="event loop"):
        Analyzer(model=cast(aic_sdk.Model, FakeModel()), license_key="test-license")


@pytest.mark.asyncio
async def test_disabled_collector_passes_frames_without_buffering(
    fake_sdk: tuple[FakeCollector, FakeNativeAnalyzer, list[int]],
) -> None:
    native_collector, native_analyzer, _ = fake_sdk
    analyzer = Analyzer(
        model=cast(aic_sdk.Model, FakeModel()),
        license_key="test-license",
        analysis_interval=60.0,
    )

    analyzer.collector.enabled = False
    frame = make_frame()
    assert analyzer.collector._process(frame) is frame
    assert native_collector.blocks == []

    analyzer.collector.enabled = True
    assert native_analyzer.reset_calls == 1
    await analyzer.aclose()


@pytest.mark.asyncio
async def test_analyzes_only_after_audio_is_buffered(
    fake_sdk: tuple[FakeCollector, FakeNativeAnalyzer, list[int]],
) -> None:
    _, native_analyzer, _ = fake_sdk
    analyzer = Analyzer(
        model=cast(aic_sdk.Model, FakeModel()),
        license_key="test-license",
        analysis_interval=0.01,
    )

    await asyncio.sleep(0.05)
    assert native_analyzer.analyze_calls == 0

    analyzer.collector._process(make_frame())
    await wait_until(lambda: native_analyzer.analyze_calls >= 1)

    analyzer.collector._on_stream_info_updated(
        room_name="test-room",
        participant_identity="test-participant",
        publication_sid="TR_test",
    )
    calls = native_analyzer.analyze_calls
    await asyncio.sleep(0.05)
    assert native_analyzer.analyze_calls == calls

    analyzer.collector._process(make_frame())
    await wait_until(lambda: native_analyzer.analyze_calls > calls)
    await analyzer.aclose()


@pytest.mark.asyncio
async def test_collector_resets_on_every_stream_info_update(
    fake_sdk: tuple[FakeCollector, FakeNativeAnalyzer, list[int]],
) -> None:
    _, native_analyzer, _ = fake_sdk
    analyzer = Analyzer(
        model=cast(aic_sdk.Model, FakeModel()),
        license_key="test-license",
        analysis_interval=60.0,
    )

    for _ in range(2):
        analyzer.collector._on_stream_info_updated(
            room_name="test-room",
            participant_identity="test-participant",
            publication_sid="TR_test",
        )

    assert native_analyzer.reset_calls == 2
    await analyzer.aclose()


@pytest.mark.asyncio
async def test_reinitializes_collector_when_frame_format_changes(
    fake_sdk: tuple[FakeCollector, FakeNativeAnalyzer, list[int]],
) -> None:
    native_collector, _, _ = fake_sdk
    analyzer = Analyzer(
        model=cast(aic_sdk.Model, FakeModel()),
        license_key="test-license",
        analysis_interval=60.0,
    )

    formats = [
        (16000, 1, 4),
        (16000, 1, 4),
        (48000, 1, 4),
        (48000, 2, 4),
        (48000, 1, 8),
        (48000, 1, 8),
    ]
    for sample_rate, channels, samples_per_channel in formats:
        analyzer.collector._process(
            rtc.AudioFrame(
                data=bytes(2 * channels * samples_per_channel),
                sample_rate=sample_rate,
                num_channels=channels,
                samples_per_channel=samples_per_channel,
            )
        )

    assert native_collector.inits == [
        (16000, 4, False),
        (48000, 4, False),
        (48000, 4, False),
        (48000, 8, False),
    ]
    await analyzer.aclose()


@pytest.mark.asyncio
async def test_analysis_failure_is_logged_and_retried(
    fake_sdk: tuple[FakeCollector, FakeNativeAnalyzer, list[int]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, native_analyzer, _ = fake_sdk
    native_analyzer.error = RuntimeError("analysis failed")
    analyzer = Analyzer(
        model=cast(aic_sdk.Model, FakeModel()),
        license_key="test-license",
        analysis_interval=0.01,
    )
    events: list[AnalysisEvent] = []
    analyzer.on("analysis_result", events.append)
    analyzer.collector._process(make_frame())

    with caplog.at_level(logging.ERROR, logger="ai_coustics.livekit"):
        await wait_until(lambda: native_analyzer.analyze_calls >= 2)
        native_analyzer.error = None
        await wait_until(lambda: bool(events))
    await analyzer.aclose()

    assert any(
        record.message == "Analyzer: buffered audio analysis failed" for record in caplog.records
    )
    assert events[0].sequence == 1


@pytest.mark.asyncio
async def test_throwing_listener_does_not_stop_analysis(
    fake_sdk: tuple[FakeCollector, FakeNativeAnalyzer, list[int]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    analyzer = Analyzer(
        model=cast(aic_sdk.Model, FakeModel()),
        license_key="test-license",
        analysis_interval=0.01,
    )
    sequences: list[int] = []

    def failing_listener(event: AnalysisEvent) -> None:
        sequences.append(event.sequence)
        # Only TypeError escapes rtc.EventEmitter.emit; it logs and swallows everything else.
        raise TypeError("listener failed")

    analyzer.on("analysis_result", failing_listener)
    analyzer.collector._process(make_frame())

    with caplog.at_level(logging.ERROR, logger="ai_coustics.livekit"):
        await wait_until(lambda: len(sequences) >= 2)
    await analyzer.aclose()

    assert sequences[:2] == [1, 2]
    assert any(
        record.message == "Analyzer: result event emission failed" for record in caplog.records
    )


@pytest.mark.asyncio
async def test_metrics_failure_is_logged_and_event_still_emitted(
    fake_sdk: tuple[FakeCollector, FakeNativeAnalyzer, list[int]],
    caplog: pytest.LogCaptureFixture,
    metric_instruments: dict[str, FakeInstrument],
) -> None:
    metric_instruments["analysis"].error = RuntimeError("metrics failed")
    analyzer = Analyzer(
        model=cast(aic_sdk.Model, FakeModel()),
        license_key="test-license",
        analysis_interval=0.01,
    )
    events: list[AnalysisEvent] = []
    analyzer.on("analysis_result", events.append)
    analyzer.collector._process(make_frame())

    with caplog.at_level(logging.ERROR, logger="ai_coustics.livekit"):
        await wait_until(lambda: bool(events))
    await analyzer.aclose()

    assert any(record.message == "Analyzer: metrics recording failed" for record in caplog.records)
    assert events[0].sequence == 1


@pytest.mark.asyncio
async def test_close_is_idempotent_and_collector_passes_frames_after_close(
    fake_sdk: tuple[FakeCollector, FakeNativeAnalyzer, list[int]],
) -> None:
    native_collector, native_analyzer, _ = fake_sdk
    analyzer = Analyzer(
        model=cast(aic_sdk.Model, FakeModel()),
        license_key="test-license",
        analysis_interval=60.0,
    )

    await analyzer.aclose()
    await analyzer.aclose()
    assert native_analyzer.terminate_calls == 1

    frame = make_frame()
    assert analyzer.collector._process(frame) is frame
    assert native_collector.blocks == []


@pytest.mark.asyncio
async def test_aclose_waits_for_in_flight_analysis(
    fake_sdk: tuple[FakeCollector, FakeNativeAnalyzer, list[int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, native_analyzer, _ = fake_sdk
    in_flight = asyncio.Event()
    release = asyncio.Event()

    async def hold_analysis(function: Callable[..., object], /, *args: object) -> object:
        if function == native_analyzer.analyze_buffered:
            in_flight.set()
            await release.wait()
        return function(*args)

    monkeypatch.setattr("ai_coustics.livekit.analyzer.asyncio.to_thread", hold_analysis)
    analyzer = Analyzer(
        model=cast(aic_sdk.Model, FakeModel()),
        license_key="test-license",
        analysis_interval=0.01,
    )
    events: list[AnalysisEvent] = []
    analyzer.on("analysis_result", events.append)
    analyzer.collector._on_stream_info_updated(
        room_name="test-room",
        participant_identity="test-participant",
        publication_sid="TR_test",
    )
    analyzer.collector._process(make_frame())
    await wait_until(in_flight.is_set)

    close_task = asyncio.create_task(analyzer.aclose())
    await asyncio.sleep(0)
    assert not close_task.done()
    assert native_analyzer.terminate_calls == 0

    release.set()
    await asyncio.wait_for(close_task, timeout=1.0)
    assert native_analyzer.terminate_calls == 1
    assert len(events) == 1
    assert events[0].room_name is None
