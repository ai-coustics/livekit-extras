import { AudioFrame, FrameProcessor } from "@livekit/rtc-node";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const logging = vi.hoisted(() => {
  const calls: Array<{
    level: "debug" | "info" | "warn" | "error";
    fields: Record<string, unknown>;
    message: string;
  }> = [];
  const write = (level: "debug" | "info" | "warn" | "error") =>
    (fields: Record<string, unknown>, message: string) =>
      calls.push({ level, fields, message });
  return {
    calls,
    logger: {
      debug: write("debug"),
      info: write("info"),
      warn: write("warn"),
      error: write("error"),
    },
  };
});

const telemetry = vi.hoisted(() => {
  const measurements = {
    analysis: [] as Array<{ value: number; attributes?: Record<string, unknown> }>,
    duration: [] as Array<{ value: number; attributes?: Record<string, unknown> }>,
    score: [] as Array<{ value: number; attributes?: Record<string, unknown> }>,
  };
  return { measurements };
});

const sdk = vi.hoisted(() => {
  const analyzers: FakeAnalyzer[] = [];
  const sdkIds: number[] = [];

  class FakeAnalyzer {
    initializations: Array<[number, number, boolean]> = [];
    blocks: number[][] = [];
    analyzeCalls = 0;
    resetCalls = 0;
    terminateCalls = 0;
    error: Error | null = null;
    hold: Promise<void> | null = null;

    constructor() {
      analyzers.push(this);
    }

    initialize(sampleRate: number, blockSize: number, variable: boolean): void {
      this.initializations.push([sampleRate, blockSize, variable]);
    }

    buffer(samples: Float32Array): void {
      this.blocks.push(Array.from(samples));
    }

    async analyzeAsync() {
      this.analyzeCalls += 1;
      if (this.hold) await this.hold;
      if (this.error) throw this.error;
      return {
        riskScore: 0.1,
        speakerReverb: 0.2,
        speakerLoudness: 0.3,
        interferingSpeech: 0.4,
        noise: 0.5,
        codecDegradation: 0.6,
        packetLoss: 0.7,
      };
    }

    reset(): void {
      this.resetCalls += 1;
    }

    terminateSession(): void {
      this.terminateCalls += 1;
    }

    updateBearerToken(): void {}
  }

  return { FakeAnalyzer, analyzers, sdkIds };
});

vi.mock("@ai-coustics/aic-sdk", () => ({
  Analyzer: sdk.FakeAnalyzer,
  Model: class {},
  Vad: class {},
  VadParameter: {
    SpeechHoldDuration: 2,
    Sensitivity: 3,
    MinimumSpeechDuration: 4,
  },
  _setSdkId: (id: number) => sdk.sdkIds.push(id),
}));

vi.mock("@livekit/agents", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@livekit/agents")>();
  return { ...actual, log: () => logging.logger };
});

vi.mock("@opentelemetry/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@opentelemetry/api")>();
  const instrument = (name: keyof typeof telemetry.measurements) => ({
    add: (value: number, attributes?: Record<string, unknown>) => {
      telemetry.measurements[name].push({ value, attributes });
    },
    record: (value: number, attributes?: Record<string, unknown>) => {
      telemetry.measurements[name].push({ value, attributes });
    },
  });
  return {
    ...actual,
    metrics: {
      ...actual.metrics,
      getMeter: () => ({
        createCounter: () => instrument("analysis"),
        createHistogram: (name: string) =>
          instrument(name.endsWith("score") ? "score" : "duration"),
      }),
    },
  };
});

import { type AnalysisEvent, Analyzer, Collector } from "../src/index.js";

function makeFrame(channels = 1): AudioFrame {
  const mono = [32767, -32768, 16384, -16384];
  const data = Int16Array.from(
    channels === 1 ? mono : mono.flatMap((sample) => Array(channels).fill(sample)),
  );
  return new AudioFrame(data, 16000, channels, 4);
}

describe("Analyzer", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    logging.calls.length = 0;
    sdk.analyzers.length = 0;
    sdk.sdkIds.length = 0;
    telemetry.measurements.analysis.length = 0;
    telemetry.measurements.duration.length = 0;
    telemetry.measurements.score.length = 0;
  });

  afterEach(() => vi.useRealTimers());

  it("exposes a transparent FrameProcessor collector that downmixes audio", () => {
    const analyzer = new Analyzer({
      model: { getId: () => "analysis-test-model" } as never,
      licenseKey: "test-license",
      analysisInterval: 60,
    });

    expect(analyzer.collector).toBeInstanceOf(Collector);
    expect(analyzer.collector).toBeInstanceOf(FrameProcessor);
    expect(sdk.sdkIds).toEqual([9]);

    const frame = makeFrame(2);
    expect(analyzer.collector.process(frame)).toBe(frame);
    expect(sdk.analyzers[0]!.initializations).toEqual([[16000, 4, false]]);
    expect(sdk.analyzers[0]!.blocks[0]).toEqual([
      32767 / 32768,
      -1,
      0.5,
      -0.5,
    ]);

    analyzer.collector.onStreamInfoCleared();
    expect(sdk.analyzers[0]!.resetCalls).toBe(1);
    analyzer.close();
    expect(sdk.analyzers[0]!.terminateCalls).toBe(1);
  });

  it("analyzes on the configured interval and emits without logging the result", async () => {
    const analyzer = new Analyzer({
      model: { getId: () => "analysis-test-model" } as never,
      licenseKey: "test-license",
      analysisInterval: 0.01,
    });
    const events: AnalysisEvent[] = [];
    analyzer.on("analysisResult", (event) => events.push(event));
    analyzer.collector.onStreamInfoUpdated({
      roomName: "test-room",
      participantIdentity: "test-participant",
      publicationSid: "TR_test",
    });
    analyzer.collector.process(makeFrame());

    await vi.advanceTimersByTimeAsync(10);

    expect(sdk.analyzers[0]!.analyzeCalls).toBe(1);
    expect(logging.calls).toEqual([]);
    expect(events).toHaveLength(1);
    expect(events[0]).toEqual(
      expect.objectContaining({
        sequence: 1,
        modelId: "analysis-test-model",
        roomName: "test-room",
        participantIdentity: "test-participant",
        publicationSid: "TR_test",
      }),
    );
    expect(Object.isFrozen(events[0])).toBe(true);
    expect(Object.isFrozen(events[0]!.result)).toBe(true);
    expect(telemetry.measurements.analysis).toEqual([
      {
        value: 1,
        attributes: { model_provider: "ai-coustics", status: "ok" },
      },
    ]);
    expect(telemetry.measurements.score).toHaveLength(7);
    expect(telemetry.measurements.score[0]).toEqual({
      value: 0.1,
      attributes: { model_provider: "ai-coustics", "score.name": "risk_score" },
    });
    analyzer.close();
  });

  it("stops the analyzer when RoomIO closes its collector", () => {
    const analyzer = new Analyzer({
      model: { getId: () => "analysis-test-model" } as never,
      licenseKey: "test-license",
      analysisInterval: 1,
    });

    analyzer.collector.close();
    vi.advanceTimersByTime(1000);

    expect(analyzer.collector.isEnabled()).toBe(false);
    expect(sdk.analyzers[0]!.terminateCalls).toBe(1);
    expect(sdk.analyzers[0]!.analyzeCalls).toBe(0);
  });

  it("records failed analyses without score measurements", async () => {
    const analyzer = new Analyzer({
      model: { getId: () => "analysis-test-model" } as never,
      licenseKey: "test-license",
      analysisInterval: 0.01,
    });
    sdk.analyzers[0]!.error = new Error("analysis failed");
    analyzer.collector.process(makeFrame());

    await vi.advanceTimersByTimeAsync(10);

    expect(telemetry.measurements.analysis).toEqual([
      {
        value: 1,
        attributes: { model_provider: "ai-coustics", status: "error" },
      },
    ]);
    expect(telemetry.measurements.score).toEqual([]);
    expect(logging.calls).toContainEqual(
      expect.objectContaining({
        level: "error",
        message: "Analyzer: buffered audio analysis failed",
        fields: expect.objectContaining({
          plugin: "ai-coustics",
          component: "analyzer",
          modelName: "analysis-test-model",
          error: expect.any(Error),
        }),
      }),
    );
    analyzer.close();
  });

  it("defaults to analyzing every five seconds", () => {
    const analyzer = new Analyzer({
      model: { getId: () => "analysis-test-model" } as never,
      licenseKey: "test-license",
    });
    analyzer.collector.process(makeFrame());

    vi.advanceTimersByTime(4999);
    expect(sdk.analyzers[0]!.analyzeCalls).toBe(0);
    vi.advanceTimersByTime(1);
    expect(sdk.analyzers[0]!.analyzeCalls).toBe(1);

    analyzer.close();
  });

  it.each([0, -1, Number.POSITIVE_INFINITY, Number.NaN])(
    "rejects invalid interval %s",
    (analysisInterval) => {
      expect(
        () =>
          new Analyzer({
            model: { getId: () => "analysis-test-model" } as never,
            licenseKey: "test-license",
            analysisInterval,
          }),
      ).toThrow(/analysisInterval/);
    },
  );

  it("disabled collector passes frames without buffering", () => {
    const analyzer = new Analyzer({
      model: { getId: () => "analysis-test-model" } as never,
      licenseKey: "test-license",
      analysisInterval: 60,
    });

    analyzer.collector.setEnabled(false);
    const frame = makeFrame();
    expect(analyzer.collector.process(frame)).toBe(frame);
    expect(sdk.analyzers[0]!.blocks).toEqual([]);

    analyzer.collector.setEnabled(true);
    expect(sdk.analyzers[0]!.resetCalls).toBe(1);
    analyzer.close();
  });

  it("analyzes only after audio is buffered", async () => {
    const analyzer = new Analyzer({
      model: { getId: () => "analysis-test-model" } as never,
      licenseKey: "test-license",
      analysisInterval: 0.01,
    });

    await vi.advanceTimersByTimeAsync(50);
    expect(sdk.analyzers[0]!.analyzeCalls).toBe(0);

    analyzer.collector.process(makeFrame());
    await vi.advanceTimersByTimeAsync(10);
    expect(sdk.analyzers[0]!.analyzeCalls).toBe(1);

    analyzer.collector.onStreamInfoUpdated({
      roomName: "test-room",
      participantIdentity: "test-participant",
      publicationSid: "TR_test",
    });
    await vi.advanceTimersByTimeAsync(50);
    expect(sdk.analyzers[0]!.analyzeCalls).toBe(1);

    analyzer.collector.process(makeFrame());
    await vi.advanceTimersByTimeAsync(10);
    expect(sdk.analyzers[0]!.analyzeCalls).toBe(2);
    analyzer.close();
  });

  it("collector resets on every stream info update", () => {
    const analyzer = new Analyzer({
      model: { getId: () => "analysis-test-model" } as never,
      licenseKey: "test-license",
      analysisInterval: 60,
    });
    const info = {
      roomName: "test-room",
      participantIdentity: "test-participant",
      publicationSid: "TR_test",
    };

    analyzer.collector.onStreamInfoUpdated(info);
    analyzer.collector.onStreamInfoUpdated(info);

    expect(sdk.analyzers[0]!.resetCalls).toBe(2);
    analyzer.close();
  });

  it("defers stream boundary resets until in-flight analysis settles", async () => {
    const analyzer = new Analyzer({
      model: { getId: () => "analysis-test-model" } as never,
      licenseKey: "test-license",
      analysisInterval: 0.01,
    });
    let release!: () => void;
    sdk.analyzers[0]!.hold = new Promise((resolve) => {
      release = resolve;
    });
    analyzer.collector.process(makeFrame());
    await vi.advanceTimersByTimeAsync(10);
    expect(sdk.analyzers[0]!.analyzeCalls).toBe(1);

    analyzer.collector.onStreamInfoUpdated({
      roomName: "test-room",
      participantIdentity: "test-participant",
      publicationSid: "TR_test",
    });
    analyzer.collector.onStreamInfoCleared();
    expect(sdk.analyzers[0]!.resetCalls).toBe(0);

    release();
    await vi.advanceTimersByTimeAsync(0);
    expect(sdk.analyzers[0]!.resetCalls).toBe(1);
    analyzer.close();
  });

  it("deferred reset clears frames buffered during analysis", async () => {
    const analyzer = new Analyzer({
      model: { getId: () => "analysis-test-model" } as never,
      licenseKey: "test-license",
      analysisInterval: 0.01,
    });
    let release!: () => void;
    sdk.analyzers[0]!.hold = new Promise((resolve) => {
      release = resolve;
    });
    analyzer.collector.process(makeFrame());
    await vi.advanceTimersByTimeAsync(10);
    expect(sdk.analyzers[0]!.analyzeCalls).toBe(1);

    analyzer.collector.onStreamInfoUpdated({
      roomName: "test-room",
      participantIdentity: "test-participant",
      publicationSid: "TR_test",
    });
    analyzer.collector.process(makeFrame());
    expect(analyzer.collector.initialized).toBe(true);

    release();
    await vi.advanceTimersByTimeAsync(0);
    expect(sdk.analyzers[0]!.resetCalls).toBe(1);
    expect(analyzer.collector.initialized).toBe(false);
    await vi.advanceTimersByTimeAsync(50);
    expect(sdk.analyzers[0]!.analyzeCalls).toBe(1);
    analyzer.close();
  });

  it("reinitializes collector when frame format changes", () => {
    const analyzer = new Analyzer({
      model: { getId: () => "analysis-test-model" } as never,
      licenseKey: "test-license",
      analysisInterval: 60,
    });
    const formats: Array<[number, number, number]> = [
      [16000, 1, 4],
      [16000, 1, 4],
      [48000, 1, 4],
      [48000, 2, 4],
      [48000, 1, 8],
      [48000, 1, 8],
    ];

    for (const [sampleRate, channels, samplesPerChannel] of formats) {
      analyzer.collector.process(
        new AudioFrame(
          new Int16Array(channels * samplesPerChannel),
          sampleRate,
          channels,
          samplesPerChannel,
        ),
      );
    }

    expect(sdk.analyzers[0]!.initializations).toEqual([
      [16000, 4, false],
      [48000, 4, false],
      [48000, 4, false],
      [48000, 8, false],
    ]);
    analyzer.close();
  });

  it("analysis failure is logged and retried", async () => {
    const analyzer = new Analyzer({
      model: { getId: () => "analysis-test-model" } as never,
      licenseKey: "test-license",
      analysisInterval: 0.01,
    });
    const events: AnalysisEvent[] = [];
    analyzer.on("analysisResult", (event) => events.push(event));
    sdk.analyzers[0]!.error = new Error("analysis failed");
    analyzer.collector.process(makeFrame());

    await vi.advanceTimersByTimeAsync(20);
    expect(sdk.analyzers[0]!.analyzeCalls).toBe(2);
    expect(logging.calls).toContainEqual(
      expect.objectContaining({
        level: "error",
        message: "Analyzer: buffered audio analysis failed",
      }),
    );

    sdk.analyzers[0]!.error = null;
    await vi.advanceTimersByTimeAsync(10);
    expect(events).toHaveLength(1);
    expect(events[0]!.sequence).toBe(1);
    analyzer.close();
  });

  it("throwing listener does not stop analysis", async () => {
    const analyzer = new Analyzer({
      model: { getId: () => "analysis-test-model" } as never,
      licenseKey: "test-license",
      analysisInterval: 0.01,
    });
    const sequences: number[] = [];
    analyzer.on("analysisResult", (event) => {
      sequences.push(event.sequence);
      throw new Error("listener failed");
    });
    analyzer.collector.process(makeFrame());

    await vi.advanceTimersByTimeAsync(20);

    expect(sequences).toEqual([1, 2]);
    expect(logging.calls).toContainEqual(
      expect.objectContaining({
        level: "error",
        message: "Analyzer: result event emission failed",
      }),
    );
    analyzer.close();
  });

  it("close is idempotent and collector passes frames after close", () => {
    const analyzer = new Analyzer({
      model: { getId: () => "analysis-test-model" } as never,
      licenseKey: "test-license",
      analysisInterval: 60,
    });

    analyzer.close();
    analyzer.close();
    expect(sdk.analyzers[0]!.terminateCalls).toBe(1);

    const frame = makeFrame();
    expect(analyzer.collector.process(frame)).toBe(frame);
    expect(sdk.analyzers[0]!.blocks).toEqual([]);
  });
});
