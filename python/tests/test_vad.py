from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import cast

import aic_sdk
import numpy as np
import pytest
from livekit import agents, rtc

from ai_coustics.livekit import VAD, VADParameters

native_calls: list[tuple[str, int | None]] = []


class FakeModel:
    def __init__(self, *, sample_rate: int = 16000, block_size: int = 160) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size

    def get_id(self) -> str:
        return "vad-test-model"

    def get_optimal_sample_rate(self) -> int:
        return self.sample_rate

    def get_optimal_block_size(self, sample_rate: int) -> int:
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
async def test_emits_inference_and_speech_transition_events() -> None:
    vad = create_vad(vad_parameters=VADParameters(speech_hold_duration=0.01))
    stream = vad.stream()
    native = FakeVad.instances[0]
    native.predictions.extend(
        [
            (0.1, False),
            (0.9, True),
            (0.8, True),
            (0.1, False),
        ]
    )

    push_processed(vad, stream, make_frame(np.arange(640, dtype=np.int16)))
    events = await collect_events(stream)

    assert [event.type for event in events] == [
        agents.vad.VADEventType.INFERENCE_DONE,
        agents.vad.VADEventType.INFERENCE_DONE,
        agents.vad.VADEventType.START_OF_SPEECH,
        agents.vad.VADEventType.INFERENCE_DONE,
        agents.vad.VADEventType.INFERENCE_DONE,
        agents.vad.VADEventType.END_OF_SPEECH,
    ]
    inference_events = [
        event for event in events if event.type == agents.vad.VADEventType.INFERENCE_DONE
    ]
    assert [event.probability for event in inference_events] == [0.1, 0.9, 0.8, 0.1]
    assert [event.samples_index for event in inference_events] == [160, 320, 480, 640]
    assert inference_events[1].raw_accumulated_speech == pytest.approx(0.01)
    assert inference_events[-1].raw_accumulated_silence == pytest.approx(0.01)

    start = events[2]
    end = events[-1]
    assert start.speaking is True
    assert end.speaking is False
    assert len(start.frames) == 1
    assert len(end.frames) == 1
    assert np.array_equal(
        np.frombuffer(start.frames[0].data, dtype=np.int16),
        np.arange(320, dtype=np.int16),
    )
    assert end.frames[0].samples_per_channel == 640
    assert np.array_equal(
        np.frombuffer(end.frames[0].data, dtype=np.int16),
        np.arange(640, dtype=np.int16),
    )
    assert native.terminate_calls == 0


@pytest.mark.asyncio
async def test_enforces_default_silence_floor_after_short_speech() -> None:
    vad = create_vad()
    stream = vad.stream()
    native = FakeVad.instances[0]
    native.predictions.extend([(0.9, True), *[(0.1, False)] * 25])

    push_processed(vad, stream, make_frame(np.arange(26 * 160, dtype=np.int16)))
    events = await collect_events(stream)

    inference_events = [
        event for event in events if event.type == agents.vad.VADEventType.INFERENCE_DONE
    ]
    end_events = [event for event in events if event.type == agents.vad.VADEventType.END_OF_SPEECH]
    assert len(end_events) == 1
    assert all(event.speaking for event in inference_events[1:])
    assert inference_events[-1].raw_accumulated_silence == pytest.approx(0.25)
    assert end_events[0].silence_duration == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_aligns_events_and_candidate_audio_with_sdk_prediction_delay() -> None:
    vad = create_vad(
        vad_parameters=VADParameters(
            speech_hold_duration=0.03,
            minimum_speech_duration=0.02,
        ),
        prefix_padding_duration=0.0,
    )
    stream = vad.stream()
    native = FakeVad.instances[0]
    native.context.prediction_delay_samples = 320
    native.predictions.extend(
        [
            (0.1, False),
            (0.1, False),
            (0.1, False),
            (0.1, False),
            (0.1, False),
            (0.9, False),
            (0.9, True),
            (0.1, False),
        ]
    )
    samples = np.concatenate([np.full(160, index, dtype=np.int16) for index in range(1, 9)])

    push_processed(vad, stream, make_frame(samples))
    events = await collect_events(stream)

    start = next(event for event in events if event.type == agents.vad.VADEventType.START_OF_SPEECH)
    end = next(event for event in events if event.type == agents.vad.VADEventType.END_OF_SPEECH)
    assert start.timestamp == pytest.approx(0.07)
    assert start.speech_duration == pytest.approx(0.04)
    assert start.raw_accumulated_speech == pytest.approx(0.04)
    assert start.timestamp - start.speech_duration == pytest.approx(0.03)
    assert len(start.frames) == 1
    start_samples = np.frombuffer(start.frames[0].data, dtype=np.int16)
    assert start_samples[::160].tolist() == [4, 5, 6, 7]
    assert end.timestamp == pytest.approx(0.08)
    assert end.silence_duration == pytest.approx(0.03)
    assert end.speech_duration == pytest.approx(0.02)
    assert end.raw_accumulated_silence == pytest.approx(0.03)


@pytest.mark.asyncio
async def test_caps_contiguous_speech_audio_and_keeps_rolling_prefix_current() -> None:
    vad = create_vad(
        model=FakeModel(sample_rate=10, block_size=4),
        vad_parameters=VADParameters(minimum_speech_duration=0.0),
        prefix_padding_duration=0.2,
        max_buffered_speech=0.5,
    )
    stream = vad.stream()
    native = FakeVad.instances[0]
    native.predictions.extend(
        [
            (0.9, True),
            (0.1, False),
            (0.1, False),
            (0.9, True),
            (0.1, False),
        ]
    )

    push_processed(vad, stream, make_frame(np.arange(1, 21, dtype=np.int16), sample_rate=10))
    events = await collect_events(stream)

    starts = [event for event in events if event.type == agents.vad.VADEventType.START_OF_SPEECH]
    ends = [event for event in events if event.type == agents.vad.VADEventType.END_OF_SPEECH]
    assert len(starts) == 2
    assert len(ends) == 2
    assert all(len(event.frames) == 1 for event in [*starts, *ends])
    assert np.frombuffer(ends[0].frames[0].data, dtype=np.int16).tolist() == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]
    assert np.frombuffer(starts[1].frames[0].data, dtype=np.int16).tolist() == [
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    ]


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
    raw = np.array([1000, 2000, 3000, 4000], dtype=np.int16)

    output = MutingProcessor()._process(vad.processor._process(make_frame(raw)))
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
