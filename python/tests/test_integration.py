from __future__ import annotations

import asyncio
import base64
import os
import zlib
from pathlib import Path

import numpy as np
import pytest
from livekit import agents, rtc

from ai_coustics import livekit as ai_coustics

pytestmark = pytest.mark.integration

LICENSE = os.getenv("AIC_SDK_LICENSE")
VAD_MODEL_ID = os.getenv("AIC_INTEGRATION_VAD_MODEL_ID", "vad-2.1-xxs-16khz")
ANALYSIS_MODEL_ID = os.getenv("AIC_INTEGRATION_ANALYSIS_MODEL_ID", "tyto-1.1-l-16khz")
MODEL_DIR = Path(os.getenv("AIC_INTEGRATION_MODEL_DIR", "~/.cache/aic-sdk/models")).expanduser()
SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = 800  # LiveKit Agents' current 50 ms default input frame.


def _recorded_speech_fixture() -> np.ndarray:
    encoded = (Path(__file__).parent / "data" / "yes_speech_i8.b85").read_bytes().strip()
    quantized = np.frombuffer(zlib.decompress(base64.b85decode(encoded)), dtype=np.int8)
    return quantized.astype(np.int16) * 256


def _frame(
    index: int,
    *,
    data: np.ndarray | None = None,
) -> rtc.AudioFrame:
    if data is None:
        rng = np.random.default_rng(index + 7)
        offset = index * SAMPLES_PER_FRAME
        t = (np.arange(SAMPLES_PER_FRAME) + offset) / SAMPLE_RATE
        voice = 0.3 * np.sin(2 * np.pi * 220 * t)
        noise = 0.1 * rng.standard_normal(SAMPLES_PER_FRAME)
        mono = np.clip((voice + noise) * 32768, -32768, 32767).astype(np.int16)
    else:
        mono = data
    return rtc.AudioFrame(
        data=mono.tobytes(),
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=SAMPLES_PER_FRAME,
    )


def _push_audio(stream: agents.vad.VADStream, audio: np.ndarray) -> None:
    for start in range(0, audio.size, SAMPLES_PER_FRAME):
        block = audio[start : start + SAMPLES_PER_FRAME]
        stream.push_frame(
            rtc.AudioFrame(
                data=block.tobytes(),
                sample_rate=SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=block.size,
            )
        )


@pytest.fixture(scope="module")
def vad_model() -> ai_coustics.Model:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ai_coustics.Model.download(VAD_MODEL_ID, MODEL_DIR)
    return ai_coustics.Model.from_file(model_path)


@pytest.fixture(scope="module")
def analysis_model() -> ai_coustics.Model:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ai_coustics.Model.download(ANALYSIS_MODEL_ID, MODEL_DIR)
    return ai_coustics.Model.from_file(model_path)


@pytest.mark.skipif(not LICENSE, reason="AIC_SDK_LICENSE is required")
@pytest.mark.asyncio
async def test_real_vad_stream_emits_livekit_inference_events(
    vad_model: ai_coustics.Model,
) -> None:
    detector = ai_coustics.VAD(
        model=vad_model,
        vad_parameters=ai_coustics.VADParameters(sensitivity=0.5),
    )
    stream = detector.stream()
    input_sample_rate = 48000
    input_samples_per_frame = input_sample_rate // 20
    silence = np.zeros(input_samples_per_frame, dtype=np.int16)
    for _ in range(10):
        stream.push_frame(
            rtc.AudioFrame(
                data=silence.tobytes(),
                sample_rate=input_sample_rate,
                num_channels=1,
                samples_per_channel=input_samples_per_frame,
            )
        )
    stream.end_input()

    events = [event async for event in stream]
    inference_events = [
        event for event in events if event.type == agents.vad.VADEventType.INFERENCE_DONE
    ]

    assert inference_events
    assert all(0.0 <= event.probability <= 1.0 for event in inference_events)
    assert all(event.frames[0].num_channels == 1 for event in inference_events)
    assert all(event.frames[0].sample_rate == input_sample_rate for event in inference_events)


@pytest.mark.skipif(not LICENSE, reason="AIC_SDK_LICENSE is required")
@pytest.mark.asyncio
async def test_real_vad_stream_detects_and_buffers_recorded_speech(
    vad_model: ai_coustics.Model,
) -> None:
    detector = ai_coustics.VAD(
        model=vad_model,
        vad_parameters=ai_coustics.VADParameters(
            sensitivity=0.5,
            speech_hold_duration=0.1,
            minimum_speech_duration=0.0,
        ),
        prefix_padding_duration=0.1,
    )
    stream = detector.stream()
    speech = _recorded_speech_fixture()
    audio = np.concatenate(
        [
            np.zeros(SAMPLE_RATE // 2, dtype=np.int16),
            speech,
            np.zeros(SAMPLE_RATE, dtype=np.int16),
        ]
    )
    _push_audio(stream, audio)
    stream.end_input()

    events = [event async for event in stream]
    starts = [event for event in events if event.type == agents.vad.VADEventType.START_OF_SPEECH]
    ends = [event for event in events if event.type == agents.vad.VADEventType.END_OF_SPEECH]

    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0].speech_duration > 0.0
    assert ends[0].silence_duration >= 0.1
    assert len(starts[0].frames) == 1
    assert len(ends[0].frames) == 1
    assert all(frame.sample_rate == SAMPLE_RATE for frame in ends[0].frames)
    buffered_speech = b"".join(bytes(frame.data) for frame in ends[0].frames)
    assert speech.tobytes() in buffered_speech


@pytest.mark.skipif(not LICENSE, reason="AIC_SDK_LICENSE is required")
@pytest.mark.asyncio
async def test_real_vad_sequential_streams_share_one_native_vad(
    vad_model: ai_coustics.Model,
) -> None:
    detector = ai_coustics.VAD(
        model=vad_model,
        vad_parameters=ai_coustics.VADParameters(
            sensitivity=0.5,
            speech_hold_duration=0.1,
            minimum_speech_duration=0.0,
        ),
    )
    native = detector._idle[0]
    audio = np.concatenate(
        [
            np.zeros(SAMPLE_RATE // 2, dtype=np.int16),
            _recorded_speech_fixture(),
            np.zeros(SAMPLE_RATE, dtype=np.int16),
        ]
    )

    for _ in range(2):
        stream = detector.stream()
        _push_audio(stream, audio)
        stream.end_input()
        events = [event async for event in stream]
        assert any(event.type == agents.vad.VADEventType.START_OF_SPEECH for event in events)

    assert detector._idle == [native]
    assert not detector._active


@pytest.mark.skipif(not LICENSE, reason="AIC_SDK_LICENSE is required")
@pytest.mark.asyncio
async def test_real_analyzer_emits_current_sdk_result_schema(
    analysis_model: ai_coustics.Model,
) -> None:
    analyzer = ai_coustics.Analyzer(model=analysis_model, analysis_interval=0.01)
    result_future: asyncio.Future[ai_coustics.AnalysisEvent] = (
        asyncio.get_running_loop().create_future()
    )

    @analyzer.on("analysis_result")
    def on_analysis(event: ai_coustics.AnalysisEvent) -> None:
        if not result_future.done():
            result_future.set_result(event)

    try:
        for index in range(120):  # Six seconds; the current model analyzes five-second windows.
            analyzer.collector._process(_frame(index))

        event = await asyncio.wait_for(result_future, timeout=30.0)
    finally:
        await analyzer.aclose()

    scores = {
        "risk_score": event.result.risk_score,
        "speaker_reverb": event.result.speaker_reverb,
        "speaker_loudness": event.result.speaker_loudness,
        "interfering_speech": event.result.interfering_speech,
        "noise": event.result.noise,
        "codec_degradation": event.result.codec_degradation,
        "packet_loss": event.result.packet_loss,
    }
    assert all(0.0 <= score <= 1.0 for score in scores.values())
