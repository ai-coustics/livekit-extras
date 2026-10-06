from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import aic_sdk
import numpy as np
import pytest
from livekit import agents, rtc

from ai_coustics.livekit import VAD, FrameProcessorChain, VADParameters

native_calls: list[tuple[str, int | None]] = []
SCENARIOS: list[dict[str, Any]] = json.loads(
    (Path(__file__).resolve().parents[2] / "testdata" / "vad_scenarios.json").read_text()
)


class FakeModel:
    def __init__(self, *, sample_rate: int = 16000, block_size: int = 160) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.queried_sample_rates: list[int] = []

    def get_id(self) -> str:
        return "vad-test-model"

    def get_optimal_sample_rate(self) -> int:
        return self.sample_rate

    def get_optimal_block_size(self, sample_rate: int) -> int:
        self.queried_sample_rates.append(sample_rate)
        return round(self.block_size * sample_rate / self.sample_rate)


class FakeVadContext:
    parameter_errors: dict[str, Exception] = {}

    def __init__(self) -> None:
        self.parameters = {
            str(aic_sdk.VadParameter.Sensitivity): 0.5,
            str(aic_sdk.VadParameter.SpeechHoldDuration): 0.25,
            str(aic_sdk.VadParameter.MinimumSpeechDuration): 0.05,
        }
        self.probability = 0.0
        self.detected = False
        self.prediction_delay_samples = 0
        self.reset_count = 0

    def set_parameter(self, parameter: aic_sdk.VadParameter, value: float) -> None:
        if error := self.parameter_errors.get(str(parameter)):
            raise error
        self.parameters[str(parameter)] = value

    def get_parameter(self, parameter: aic_sdk.VadParameter) -> float:
        return self.parameters[str(parameter)]

    def raw_vad_probability(self) -> float:
        return self.probability

    def is_speech_detected(self) -> bool:
        return self.detected

    def get_prediction_delay(self) -> int:
        return self.prediction_delay_samples

    def reset(self) -> None:
        self.probability = 0.0
        self.detected = False
        self.reset_count += 1


class FakeVad:
    instances: list[FakeVad] = []

    def __init__(
        self,
        model: object,
        license_key: str,
        config: aic_sdk.ProcessorConfig | None = None,
    ) -> None:
        self.model = model
        self.license_key = license_key
        self.config = config
        self.context = FakeVadContext()
        self.predictions: deque[tuple[float, bool]] = deque()
        self.blocks: list[np.ndarray] = []
        self.initialized_configs: list[aic_sdk.ProcessorConfig] = []
        self.terminate_calls = 0
        self.initialize_error: Exception | None = None
        self.on_process: Callable[[], None] | None = None
        self.process_error: Exception | None = None
        self.instances.append(self)
        native_calls.append(("vad", None))

    def get_context(self) -> FakeVadContext:
        return self.context

    def initialize(self, config: aic_sdk.ProcessorConfig) -> None:
        self.config = config
        self.initialized_configs.append(config)
        if self.initialize_error is not None:
            raise self.initialize_error

    def process(self, block: np.ndarray) -> None:
        self.blocks.append(block.copy())
        if self.on_process is not None:
            self.on_process()
        if self.process_error is not None:
            raise self.process_error
        if self.predictions:
            self.context.probability, self.context.detected = self.predictions.popleft()

    def terminate_session(self) -> None:
        self.terminate_calls += 1


@pytest.fixture(autouse=True)
def fake_native_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeVad.instances.clear()
    FakeVadContext.parameter_errors.clear()
    native_calls.clear()
    monkeypatch.setattr(
        "ai_coustics.livekit.vad.aic_sdk.set_sdk_id",
        lambda sdk_id: native_calls.append(("sdk_id", sdk_id)),
    )
    monkeypatch.setattr(
        "ai_coustics.livekit.vad.aic_sdk.Vad",
        FakeVad,
    )


def create_vad(
    *,
    model: FakeModel | None = None,
    vad_parameters: VADParameters | None = None,
    **kwargs: object,
) -> VAD:
    return VAD(
        model=cast(aic_sdk.Model, model or FakeModel()),
        license_key="test-license",
        vad_parameters=vad_parameters,
        **kwargs,  # type: ignore[arg-type]
    )


def make_frame(
    data: np.ndarray,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=data.tobytes(),
        sample_rate=sample_rate,
        num_channels=channels,
        samples_per_channel=data.size // channels,
    )


def push_processed(vad: VAD, stream: agents.vad.VADStream, frame: rtc.AudioFrame) -> None:
    stream.push_frame(vad.processor._process(frame))


async def collect_events(stream: agents.vad.VADStream) -> list[agents.vad.VADEvent]:
    stream.end_input()
    return [event async for event in stream]


def test_constructs_first_native_vad_eagerly_without_an_audio_config() -> None:
    vad = create_vad(
        vad_parameters=VADParameters(
            sensitivity=0.7,
            speech_hold_duration=0.4,
            minimum_speech_duration=0.1,
        )
    )

    native = FakeVad.instances[0]
    assert isinstance(vad, agents.vad.VAD)
    assert native_calls[:2] == [("sdk_id", 8), ("vad", None)]
    assert native.config is None
    assert native.context.get_parameter(aic_sdk.VadParameter.Sensitivity) == 0.7
    assert native.context.get_parameter(aic_sdk.VadParameter.SpeechHoldDuration) == 0.4
    assert native.context.get_parameter(aic_sdk.VadParameter.MinimumSpeechDuration) == 0.1
    assert vad.capabilities.update_interval == pytest.approx(0.01)
    assert vad.model == "vad-test-model"
    assert vad.provider == "ai-coustics"
    assert vad.min_silence_duration == 0.4


def test_uses_livekit_compatible_duration_defaults() -> None:
    vad = create_vad()
    context = FakeVad.instances[0].context
    vad.set_parameters(VADParameters(sensitivity=0.7))

    assert context.get_parameter(aic_sdk.VadParameter.Sensitivity) == 0.7
    assert context.get_parameter(aic_sdk.VadParameter.SpeechHoldDuration) == 0.25
    assert context.get_parameter(aic_sdk.VadParameter.MinimumSpeechDuration) == 0.05
    assert vad.min_silence_duration == 0.25


def test_rejected_parameter_warns_and_does_not_block_other_updates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    vad = create_vad()
    context = FakeVad.instances[0].context
    FakeVadContext.parameter_errors[str(aic_sdk.VadParameter.Sensitivity)] = RuntimeError(
        "SDK rejected parameter"
    )

    vad.set_parameters(VADParameters(sensitivity=0.8, speech_hold_duration=0.6))

    assert context.get_parameter(aic_sdk.VadParameter.Sensitivity) == 0.5
    assert context.get_parameter(aic_sdk.VadParameter.SpeechHoldDuration) == 0.6
    assert vad.min_silence_duration == 0.6
    warning = next(
        record for record in caplog.records if "VAD: parameter rejected" in record.message
    )
    assert warning.plugin == "ai-coustics"  # type: ignore[attr-defined]
    assert warning.component == "vad"  # type: ignore[attr-defined]
    assert warning.model_provider == "ai-coustics"  # type: ignore[attr-defined]
    assert warning.model_name == "vad-test-model"  # type: ignore[attr-defined]
    assert warning.parameter == "sensitivity"  # type: ignore[attr-defined]
    assert warning.error_message == "SDK rejected parameter"  # type: ignore[attr-defined]


def test_rejected_constructor_parameter_keeps_vad_operational(
    caplog: pytest.LogCaptureFixture,
) -> None:
    FakeVadContext.parameter_errors[str(aic_sdk.VadParameter.Sensitivity)] = ValueError(
        "sensitivity out of range"
    )

    create_vad(vad_parameters=VADParameters(sensitivity=2.0, minimum_speech_duration=0.1))
    context = FakeVad.instances[0].context

    assert context.get_parameter(aic_sdk.VadParameter.Sensitivity) == 0.5
    assert context.get_parameter(aic_sdk.VadParameter.MinimumSpeechDuration) == 0.1
    assert any("VAD: parameter rejected" in record.message for record in caplog.records)


def test_wraps_native_vad_construction_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk_error = RuntimeError("not a VAD model")

    def fail_vad(*_args: object, **_kwargs: object) -> None:
        raise sdk_error

    monkeypatch.setattr("ai_coustics.livekit.vad.aic_sdk.Vad", fail_vad)

    with pytest.raises(RuntimeError, match="Failed to create ai-coustics VAD") as exc_info:
        create_vad()

    assert exc_info.value.__cause__ is sdk_error


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[scenario["name"] for scenario in SCENARIOS])
async def test_replays_shared_vad_scenario(scenario: dict[str, Any]) -> None:
    vad = create_vad(
        model=FakeModel(**scenario["model"]),
        vad_parameters=VADParameters(**scenario.get("vad_parameters", {})),
        **{
            option: scenario[option]
            for option in ("prefix_padding_duration", "max_buffered_speech")
            if option in scenario
        },
    )
    stream = vad.stream()
    native = FakeVad.instances[0]
    native.context.prediction_delay_samples = scenario["prediction_delay_samples"]
    native.predictions.extend(tuple(prediction) for prediction in scenario["predictions"])

    ramp = np.arange(scenario["input_samples"], dtype=np.int16)
    push_processed(vad, stream, make_frame(ramp, sample_rate=scenario["input_sample_rate"]))
    events = await collect_events(stream)

    for event, expected in zip(events, scenario["events"], strict=True):
        (frame,) = event.frames
        audio = np.frombuffer(frame.data, dtype=np.int16)
        assert np.array_equal(audio, ramp[audio[0] : audio[0] + audio.size])
        assert {
            "type": event.type.value,
            "samples_index": event.samples_index,
            "timestamp": event.timestamp,
            "speech_duration": event.speech_duration,
            "silence_duration": event.silence_duration,
            "probability": event.probability,
            "speaking": event.speaking,
            "raw_accumulated_speech": event.raw_accumulated_speech,
            "raw_accumulated_silence": event.raw_accumulated_silence,
            "audio_start": audio[0],
            "audio_samples": audio.size,
        } == pytest.approx(expected)


@pytest.mark.asyncio
async def test_warns_for_sustained_inference_backlog_with_structured_context(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = 100.0
    durations = deque([0.31, 0.31, 10.1])

    def perf_counter() -> float:
        return clock

    def advance_clock() -> None:
        nonlocal clock
        clock += durations.popleft()

    monkeypatch.setattr("ai_coustics.livekit.vad.time.perf_counter", perf_counter)
    vad = create_vad(model=FakeModel(sample_rate=100, block_size=10))
    stream = vad.stream()
    native = FakeVad.instances[0]
    native.on_process = advance_clock
    native.predictions.extend([(0.0, False)] * 3)

    push_processed(vad, stream, make_frame(np.arange(30, dtype=np.int16), sample_rate=100))
    with caplog.at_level("WARNING", logger="ai_coustics.livekit"):
        await collect_events(stream)

    warnings = [
        record for record in caplog.records if "falling behind realtime" in record.getMessage()
    ]
    assert len(warnings) == 2
    assert warnings[0].inference_duration == pytest.approx(0.31)
    assert warnings[0].block_duration == pytest.approx(0.1)
    assert warnings[0].realtime_factor == pytest.approx(3.1)
    assert warnings[0].processing_backlog == pytest.approx(0.21)
    assert warnings[0].sample_rate == 100
    assert warnings[0].block_size == 10
    assert warnings[0].model_name == "vad-test-model"
    assert warnings[0].model_provider == "ai-coustics"


@pytest.mark.asyncio
async def test_inference_error_includes_model_and_audio_format(
    caplog: pytest.LogCaptureFixture,
) -> None:
    vad = create_vad(model=FakeModel(sample_rate=48000, block_size=480))
    stream = vad.stream()
    native = FakeVad.instances[0]
    sdk_error = ValueError("native failure")
    native.process_error = sdk_error

    push_processed(vad, stream, make_frame(np.arange(480, dtype=np.int16), sample_rate=48000))

    with pytest.raises(
        RuntimeError,
        match=(
            r"ai-coustics VAD inference failed "
            r"\(model=vad-test-model, sample_rate=48000, block_size=480\): native failure"
        ),
    ) as exc_info:
        await collect_events(stream)

    assert exc_info.value.__cause__ is sdk_error
    assert native.terminate_calls == 0
    failure = next(record for record in caplog.records if record.message == "VAD: stream failed")
    assert failure.plugin == "ai-coustics"  # type: ignore[attr-defined]
    assert failure.component == "vad"  # type: ignore[attr-defined]
    assert failure.model_provider == "ai-coustics"  # type: ignore[attr-defined]
    assert failure.model_name == "vad-test-model"  # type: ignore[attr-defined]
    assert failure.sample_rate == 48000  # type: ignore[attr-defined]
    assert failure.block_size == 480  # type: ignore[attr-defined]
    assert failure.error_type == "RuntimeError"  # type: ignore[attr-defined]
    assert failure.error_message.endswith("native failure")  # type: ignore[attr-defined]
    assert failure.exc_info is not None


@pytest.mark.asyncio
async def test_initialization_error_includes_model_and_audio_format() -> None:
    vad = create_vad(model=FakeModel(sample_rate=16000, block_size=160))
    stream = vad.stream()
    native = FakeVad.instances[0]
    sdk_error = ValueError("unsupported configuration")
    native.initialize_error = sdk_error

    push_processed(vad, stream, make_frame(np.arange(480, dtype=np.int16), sample_rate=48000))

    with pytest.raises(
        RuntimeError,
        match=(
            r"ai-coustics VAD initialization failed "
            r"\(model=vad-test-model, sample_rate=48000, block_size=480\): "
            r"unsupported configuration"
        ),
    ) as exc_info:
        await collect_events(stream)

    assert exc_info.value.__cause__ is sdk_error
    assert native.terminate_calls == 0


@pytest.mark.asyncio
async def test_downmixes_stereo_and_reblocks_for_the_sdk() -> None:
    vad = create_vad(model=FakeModel(block_size=4))
    stream = vad.stream()
    native = FakeVad.instances[0]
    native.predictions.append((0.0, False))
    stereo = np.array([1000, 3000, 2000, 4000, 3000, 5000, 4000, 6000], dtype=np.int16)

    push_processed(vad, stream, make_frame(stereo, channels=2))
    events = await collect_events(stream)

    assert len(native.blocks) == 1
    assert np.array_equal(
        native.blocks[0],
        np.array([2000, 3000, 4000, 5000], dtype=np.float32) / 32768,
    )
    assert len(events) == 1
    assert events[0].frames[0].num_channels == 1


@pytest.mark.asyncio
async def test_uses_input_sample_rate_without_plugin_resampling() -> None:
    vad = create_vad(model=FakeModel(sample_rate=16000, block_size=160))
    stream = vad.stream()
    native = FakeVad.instances[0]
    native.predictions.append((0.0, False))
    input_samples = np.arange(480, dtype=np.int16)

    push_processed(vad, stream, make_frame(input_samples, sample_rate=48000))
    events = await collect_events(stream)

    assert len(native.initialized_configs) == 1
    config = native.initialized_configs[0]
    assert config.sample_rate == 48000
    assert config.block_size == 480
    assert config.variable_block_size is False
    assert len(native.blocks) == 1
    assert np.array_equal(native.blocks[0], input_samples.astype(np.float32) / 32768)
    assert events[0].frames[0].sample_rate == 48000
    assert np.array_equal(
        np.frombuffer(events[0].frames[0].data, dtype=np.int16),
        input_samples,
    )


def test_reinitializes_only_when_the_input_sample_rate_changes() -> None:
    model = FakeModel(block_size=4)
    vad = create_vad(model=model)
    native = FakeVad.instances[0]
    model.queried_sample_rates.clear()

    for samples, sample_rate in ((3, 16000), (7, 16000), (2, 16000), (12, 48000), (12, 48000)):
        vad.processor._process(
            make_frame(np.zeros(samples, dtype=np.int16), sample_rate=sample_rate)
        )

    assert model.queried_sample_rates == [16000, 48000]
    assert [(config.sample_rate, config.block_size) for config in native.initialized_configs] == [
        (16000, 4),
        (48000, 12),
    ]
    assert [block.size for block in native.blocks] == [4, 4, 4, 12, 12]


@pytest.mark.asyncio
async def test_flush_resets_stream_state_without_repeating_shared_inference() -> None:
    vad = create_vad(model=FakeModel(block_size=4))
    stream = vad.stream()
    native = FakeVad.instances[0]
    native.predictions.append((0.0, False))

    push_processed(vad, stream, make_frame(np.array([1, 2], dtype=np.int16)))
    stream.flush()
    push_processed(vad, stream, make_frame(np.array([3, 4, 5, 6], dtype=np.int16)))
    await collect_events(stream)

    assert len(native.blocks) == 1
    assert np.array_equal(native.blocks[0], np.array([1, 2, 3, 4]) / 32768)
    assert native.context.reset_count == 0


@pytest.mark.asyncio
async def test_parameter_updates_reach_active_and_future_streams() -> None:
    vad = create_vad()
    first_stream = vad.stream()
    native = FakeVad.instances[0]

    vad.set_parameters(VADParameters(sensitivity=0.8, speech_hold_duration=0.6))
    second_stream = vad.stream()

    assert len(FakeVad.instances) == 1
    assert native.context.get_parameter(aic_sdk.VadParameter.Sensitivity) == 0.8
    assert native.context.get_parameter(aic_sdk.VadParameter.SpeechHoldDuration) == 0.6
    assert vad.min_silence_duration == 0.6

    await collect_events(first_stream)
    await collect_events(second_stream)


@pytest.mark.asyncio
async def test_multiple_streams_share_one_inference_result() -> None:
    vad = create_vad()
    first_stream = vad.stream()
    second_stream = vad.stream()
    native = FakeVad.instances[0]
    native.predictions.extend([(0.9, True), (0.1, False)])
    processed = vad.processor._process(make_frame(np.arange(320, dtype=np.int16)))

    first_stream.push_frame(processed)
    second_stream.push_frame(processed)
    first_events = await collect_events(first_stream)
    second_events = await collect_events(second_stream)

    assert len(FakeVad.instances) == 1
    assert len(native.blocks) == 2
    assert [event.type for event in first_events] == [event.type for event in second_events]
    assert [event.probability for event in first_events] == [
        event.probability for event in second_events
    ]


@pytest.mark.asyncio
async def test_missing_metadata_log_uses_standard_vad_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    vad = create_vad()
    vad.processor._on_stream_info_updated(
        room_name="room",
        participant_identity="participant",
        publication_sid="TR_test",
    )
    stream = vad.stream()

    for _ in range(10):
        stream.push_frame(make_frame(np.zeros(160, dtype=np.int16)))
    await collect_events(stream)

    error = next(
        record for record in caplog.records if "no inference metadata found" in record.message
    )
    assert error.plugin == "ai-coustics"  # type: ignore[attr-defined]
    assert error.component == "vad"  # type: ignore[attr-defined]
    assert error.model_provider == "ai-coustics"  # type: ignore[attr-defined]
    assert error.model_name == "vad-test-model"  # type: ignore[attr-defined]
    assert error.room_name == "room"  # type: ignore[attr-defined]
    assert error.participant_identity == "participant"  # type: ignore[attr-defined]
    assert error.publication_sid == "TR_test"  # type: ignore[attr-defined]
    assert error.missing_metadata_frames == 10  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_vad_audio_survives_downstream_audio_replacement() -> None:
    class MutingProcessor(rtc.FrameProcessor[rtc.AudioFrame]):
        enabled = True

        def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
            return rtc.AudioFrame(
                data=np.zeros(frame.samples_per_channel, dtype=np.int16).tobytes(),
                sample_rate=frame.sample_rate,
                num_channels=1,
                samples_per_channel=frame.samples_per_channel,
                userdata=frame.userdata,
            )

        def _close(self) -> None:
            pass

    vad = create_vad(model=FakeModel(block_size=4))
    stream = vad.stream()
    native = FakeVad.instances[0]
    native.predictions.append((0.9, True))
    chain = FrameProcessorChain(vad.processor, MutingProcessor())
    raw = np.array([1000, 2000, 3000, 4000], dtype=np.int16)

    output = chain._process(make_frame(raw))
    stream.push_frame(output)
    events = await collect_events(stream)

    assert np.array_equal(native.blocks[0], raw.astype(np.float32) / 32768)
    assert np.count_nonzero(np.frombuffer(output.data, dtype=np.int16)) == 0
    assert np.array_equal(
        np.frombuffer(events[0].frames[0].data, dtype=np.int16),
        raw,
    )


@pytest.mark.asyncio
async def test_stream_close_does_not_terminate_shared_native_session() -> None:
    vad = create_vad()
    stream = vad.stream()
    native = FakeVad.instances[0]

    await stream.aclose()
    await stream.aclose()
    assert native.terminate_calls == 0

    vad.processor._close()
    vad.processor._close()
    assert native.terminate_calls == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prefix_padding_duration": -0.1}, "prefix_padding_duration"),
        ({"max_buffered_speech": 0.0}, "max_buffered_speech"),
    ],
)
def test_validates_buffer_options(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        create_vad(**kwargs)
