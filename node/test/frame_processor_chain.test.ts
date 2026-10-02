import { AudioFrame, FrameProcessor } from "@livekit/rtc-node";
import { describe, expect, it } from "vitest";

import { FrameProcessorChain } from "../src/index.js";

type Call = [name: string, value: unknown];

class RecordingProcessor extends FrameProcessor<AudioFrame> {
  private enabled = true;

  constructor(
    private readonly name: string,
    private readonly calls: Call[],
    private readonly output?: AudioFrame,
    private readonly closeError?: Error,
  ) {
    super();
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  override onStreamInfoUpdated(info: {
    roomName: string;
    participantIdentity: string;
    publicationSid: string;
  }): void {
    this.calls.push([this.name, ["stream", info]]);
  }

  override onStreamInfoCleared(): void {
    this.calls.push([this.name, "streamCleared"]);
  }

  process(frame: AudioFrame): AudioFrame {
    this.calls.push([this.name, frame]);
    return this.output ?? frame;
  }

  close(): void {
    this.calls.push([this.name, "close"]);
    if (this.closeError) throw this.closeError;
  }
}

function makeFrame(value: number): AudioFrame {
  return new AudioFrame(Int16Array.of(value), 16000, 1, 1);
}

describe("FrameProcessorChain", () => {
  it("processes enabled children in order", () => {
    const calls: Call[] = [];
    const input = makeFrame(1);
    const intermediate = makeFrame(2);
    const output = makeFrame(3);
    const first = new RecordingProcessor("first", calls, intermediate);
    const second = new RecordingProcessor("second", calls, output);
    const chain = new FrameProcessorChain(first, second);

    expect(chain.process(input)).toBe(output);
    expect(calls).toEqual([
      ["first", input],
      ["second", intermediate],
    ]);

    calls.length = 0;
    second.setEnabled(false);
    expect(chain.process(input)).toBe(intermediate);
    expect(calls).toEqual([["first", input]]);

    calls.length = 0;
    chain.setEnabled(false);
    expect(chain.process(input)).toBe(input);
    expect(calls).toEqual([]);
  });

  it("processes an arbitrary number of children", () => {
    const calls: Call[] = [];
    const frames = Array.from({ length: 4 }, (_, value) => makeFrame(value));
    const chain = new FrameProcessorChain(
      new RecordingProcessor("first", calls, frames[1]),
      new RecordingProcessor("second", calls, frames[2]),
      new RecordingProcessor("third", calls, frames[3]),
    );

    expect(chain.process(frames[0]!)).toBe(frames[3]);
    expect(calls).toEqual([
      ["first", frames[0]],
      ["second", frames[1]],
      ["third", frames[2]],
    ]);
  });

  it("forwards stream lifecycle to all children in order", () => {
    const calls: Call[] = [];
    const chain = new FrameProcessorChain(
      new RecordingProcessor("first", calls),
      new RecordingProcessor("second", calls),
      new RecordingProcessor("third", calls),
    );
    const streamInfo = {
      roomName: "room",
      participantIdentity: "participant",
      publicationSid: "TR_test",
    };

    chain.onStreamInfoUpdated(streamInfo);
    chain.onStreamInfoCleared();
    chain.onCredentialsUpdated({ token: "token", url: "wss://example.test" });
    chain.onCredentialsCleared();

    expect(calls).toEqual([
      ["first", ["stream", streamInfo]],
      ["second", ["stream", streamInfo]],
      ["third", ["stream", streamInfo]],
      ["first", "streamCleared"],
      ["second", "streamCleared"],
      ["third", "streamCleared"],
    ]);
  });

  it("is idempotent and closes all children after an error", () => {
    const calls: Call[] = [];
    const chain = new FrameProcessorChain(
      new RecordingProcessor("first", calls, undefined, new Error("failed")),
      new RecordingProcessor("second", calls),
      new RecordingProcessor("third", calls),
    );

    expect(() => chain.close()).toThrow("failed");
    chain.close();

    expect(calls).toEqual([
      ["first", "close"],
      ["second", "close"],
      ["third", "close"],
    ]);
    expect(chain.isEnabled()).toBe(false);
  });
});
