from __future__ import annotations

from typing import TypeVar

from livekit import rtc

Frame = TypeVar("Frame", bound=rtc.AudioFrame | rtc.VideoFrame)


class FrameProcessorChain(rtc.FrameProcessor[Frame]):
    """Run frame processors in sequence and close them with the chain."""

    def __init__(self, *processors: rtc.FrameProcessor[Frame]) -> None:
        self._processors = processors
        self._enabled = True
        self._closed = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if self._closed:
            return
        self._enabled = value

    def _on_stream_info_updated(
        self,
        *,
        room_name: str,
        participant_identity: str,
        publication_sid: str,
    ) -> None:
        for processor in self._processors:
            processor._on_stream_info_updated(
                room_name=room_name,
                participant_identity=participant_identity,
                publication_sid=publication_sid,
            )

    def _on_stream_info_cleared(self) -> None:
        for processor in self._processors:
            processor._on_stream_info_cleared()

    def _process(self, frame: Frame) -> Frame:
        if not self._enabled or self._closed:
            return frame

        output = frame
        for processor in self._processors:
            if processor.enabled:
                output = processor._process(output)
        return output

    def _close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._enabled = False
        first_error: Exception | None = None
        for processor in self._processors:
            try:
                processor._close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
