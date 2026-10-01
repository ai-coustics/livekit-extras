from __future__ import annotations

import asyncio
import math
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from livekit import api, rtc
from livekit.agents import Agent, AgentSession, room_io

from ai_coustics import livekit as ai_coustics

pytestmark = [pytest.mark.integration, pytest.mark.e2e]

LICENSE = os.getenv("AIC_SDK_LICENSE")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://127.0.0.1:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")
VAD_MODEL_ID = os.getenv("AIC_INTEGRATION_VAD_MODEL_ID", "vad-2.1-xxs-16khz")
ANALYSIS_MODEL_ID = os.getenv("AIC_INTEGRATION_ANALYSIS_MODEL_ID", "tyto-1.1-l-16khz")
MODEL_DIR = Path(os.getenv("AIC_INTEGRATION_MODEL_DIR", "~/.cache/aic-sdk/models")).expanduser()

INPUT_SAMPLE_RATE = 16_000
PUBLISH_SAMPLE_RATE = 48_000
PUBLISH_FRAME_MS = 20
LICENSE_GRACE_PERIOD_SECONDS = 10.0


@pytest.fixture(scope="module")
def vad_model() -> ai_coustics.Model:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return ai_coustics.Model.from_file(ai_coustics.Model.download(VAD_MODEL_ID, MODEL_DIR))


@pytest.fixture(scope="module")
def analysis_model() -> ai_coustics.Model:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return ai_coustics.Model.from_file(ai_coustics.Model.download(ANALYSIS_MODEL_ID, MODEL_DIR))


def _token(*, identity: str, room_name: str, agent_participant: bool = False) -> str:
    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                agent=agent_participant,
            )
        )
    )
    if agent_participant:
        token = token.with_kind("agent")
    return token.to_jwt()


async def _connect(room: rtc.Room, *, identity: str, room_name: str, agent: bool = False) -> None:
    await room.connect(
        LIVEKIT_URL,
        _token(identity=identity, room_name=room_name, agent_participant=agent),
    )


async def _publish_test_signal(source: rtc.AudioSource, *, until: float) -> None:
    samples_per_frame = PUBLISH_SAMPLE_RATE * PUBLISH_FRAME_MS // 1000
    frame_index = 0
    next_tick = asyncio.get_running_loop().time()

    while time.monotonic() < until:
        offset = frame_index * samples_per_frame
        positions = (np.arange(samples_per_frame, dtype=np.float64) + offset) / PUBLISH_SAMPLE_RATE
        signal = 0.28 * np.sin(2.0 * math.pi * 220.0 * positions)
        noise = 0.10 * np.sin(2.0 * math.pi * 997.0 * positions)
        samples = np.clip((signal + noise) * 32768.0, -32768, 32767).astype(np.int16)
        await source.capture_frame(
            rtc.AudioFrame(
                data=samples.tobytes(),
                sample_rate=PUBLISH_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=samples_per_frame,
            )
        )

        frame_index += 1
        next_tick += PUBLISH_FRAME_MS / 1000.0
        delay = next_tick - asyncio.get_running_loop().time()
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            next_tick = asyncio.get_running_loop().time()


@pytest.mark.skipif(not LICENSE, reason="AIC_SDK_LICENSE is required")
@pytest.mark.asyncio
async def test_vad_runs_in_a_real_agent_room_after_license_grace_period(
    vad_model: ai_coustics.Model,
    analysis_model: ai_coustics.Model,
) -> None:
    room_name = f"ai-coustics-e2e-{uuid.uuid4().hex[:12]}"
    agent_room = rtc.Room()
    publisher_room = rtc.Room()
    created_at = time.monotonic()
    detector = ai_coustics.VAD(model=vad_model)
    analyzer = ai_coustics.Analyzer(model=analysis_model)
    # LiveKit emits VAD metrics only after successful inference events.
    metrics_times: list[float] = []
    detector.on("metrics_collected", lambda _: metrics_times.append(time.monotonic()))
    session = AgentSession(
        vad=detector,
        turn_handling={"turn_detection": "manual"},
        user_away_timeout=None,
    )
    source: rtc.AudioSource | None = None
    session_started = False

    try:
        await _connect(agent_room, identity="agent", room_name=room_name, agent=True)
        await _connect(publisher_room, identity="publisher", room_name=room_name)

        await session.start(
            agent=Agent(instructions="E2E audio probe"),
            room=agent_room,
            record=False,
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(
                    sample_rate=INPUT_SAMPLE_RATE,
                    num_channels=1,
                    frame_size_ms=50,
                    noise_cancellation=analyzer.collector,
                    auto_gain_control=False,
                ),
                audio_output=False,
                text_input=False,
                text_output=False,
                participant_identity="publisher",
            ),
        )
        session_started = True

        source = rtc.AudioSource(PUBLISH_SAMPLE_RATE, 1)
        track = rtc.LocalAudioTrack.create_audio_track("e2e-microphone", source)
        await publisher_room.local_participant.publish_track(
            track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )

        # Continue beyond the SDK's authentication grace period, while guaranteeing at least
        # three seconds of actual room audio if room setup itself took unusually long.
        stream_until = max(time.monotonic() + 3.0, created_at + 12.0)
        await _publish_test_signal(source, until=stream_until)

        assert metrics_times, "AgentSession did not consume the ai-coustics VAD stream"
        assert any(
            observed_at >= created_at + LICENSE_GRACE_PERIOD_SECONDS
            for observed_at in metrics_times
        ), "no VAD inference was observed after the license grace period"
    finally:
        if session_started:
            await session.aclose()
        await analyzer.aclose()
        if source is not None:
            await source.aclose()
        await publisher_room.disconnect()
        await agent_room.disconnect()

    assert not detector._active, "a VAD stream did not release its native VAD"
