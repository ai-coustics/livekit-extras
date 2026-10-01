from __future__ import annotations

import os

import numpy as np
from livekit import rtc


def _license_key(value: str | None) -> str:
    key = value or os.getenv("AIC_SDK_LICENSE")
    if not key:
        raise ValueError(
            "An ai-coustics SDK license is required. Pass license_key or set AIC_SDK_LICENSE."
        )
    return key


def _mono_pcm16(frame: rtc.AudioFrame) -> np.ndarray:
    samples = np.frombuffer(frame.data, dtype=np.int16)
    expected_samples = frame.samples_per_channel * frame.num_channels
    if samples.size != expected_samples:
        raise ValueError(f"AudioFrame contains {samples.size} samples, expected {expected_samples}")
    if frame.num_channels == 1:
        return samples
    channels = samples.reshape(frame.samples_per_channel, frame.num_channels)
    # numpy < 2.5 stubs type ndarray.mean() with an axis as Any.
    mono: np.ndarray = np.rint(channels.mean(axis=1, dtype=np.float32)).astype(np.int16)
    return mono


def _pcm16_to_float32(data: memoryview) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
