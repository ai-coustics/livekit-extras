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

    def add(self, value: float, *, attributes: dict[str, str]) -> None:
        self.measurements.append((value, attributes))

    def record(self, value: float, *, attributes: dict[str, str]) -> None:
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
