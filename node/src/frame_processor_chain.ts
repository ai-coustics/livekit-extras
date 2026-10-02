import {
  FrameProcessor,
  type AudioFrame,
  type FrameProcessorStreamInfo,
  type VideoFrame,
} from "@livekit/rtc-node";

/** Runs frame processors in sequence and closes them with the chain. */
export class FrameProcessorChain<
  Frame extends AudioFrame | VideoFrame,
> extends FrameProcessor<Frame> {
  private readonly processors: readonly FrameProcessor<Frame>[];
  private chainEnabled = true;
  private closed = false;

  constructor(...processors: FrameProcessor<Frame>[]) {
    super();
    this.processors = processors;
  }

  isEnabled(): boolean {
    return this.chainEnabled;
  }

  setEnabled(enabled: boolean): void {
    if (this.closed) return;
    this.chainEnabled = enabled;
  }

  override onStreamInfoUpdated(info: FrameProcessorStreamInfo): void {
    for (const processor of this.processors) {
      processor.onStreamInfoUpdated(info);
    }
  }

  override onStreamInfoCleared(): void {
    for (const processor of this.processors) {
      processor.onStreamInfoCleared();
    }
  }

  process(frame: Frame): Frame {
    if (!this.chainEnabled || this.closed) return frame;

    let output = frame;
    for (const processor of this.processors) {
      if (processor.isEnabled()) {
        output = processor.process(output);
      }
    }
    return output;
  }

  close(): void {
    if (this.closed) return;

    this.closed = true;
    this.chainEnabled = false;
    let firstError: unknown;
    for (const processor of this.processors) {
      try {
        processor.close();
      } catch (error) {
        firstError ??= error;
      }
    }
    if (firstError !== undefined) throw firstError;
  }
}
