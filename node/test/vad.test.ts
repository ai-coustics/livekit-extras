import {
  type JobContext,
  VAD as LiveKitVAD,
  VADEventType,
  initializeLogger,
  log,
  runWithJobContext,
  type VADStream,
} from "@livekit/agents";
import { AudioFrame } from "@livekit/rtc-node";
import { beforeEach, describe, expect, it, vi } from "vitest";

const sdk = vi.hoisted(() => {
  const instances: FakeVad[] = [];
  const nativeCalls: Array<[string, number?]> = [];

  const VadParameter = {
    SpeechHoldDuration: 10,
    Sensitivity: 11,
    MinimumSpeechDuration: 12,
  } as const;

  class FakeContext {
    static parameterErrors = new Map<number, Error>();
    readonly parameters = new Map<number, number>([
      [VadParameter.SpeechHoldDuration, 0.25],
      [VadParameter.Sensitivity, 0.5],
      [VadParameter.MinimumSpeechDuration, 0.05],
    ]);
    probability = 0;
    detected = false;
    predictionDelay = 0;
    resetCount = 0;

    reset(): void {
      this.probability = 0;
      this.detected = false;
      this.resetCount += 1;
    }

    isSpeechDetected(): boolean {
      return this.detected;
    }

    getRawVadProbability(): number {
      return this.probability;
    }

    setParameter(parameter: number, value: number): void {
      const error = FakeContext.parameterErrors.get(parameter);
      if (error) throw error;
      this.parameters.set(parameter, value);
    }

    getParameter(parameter: number): number {
      return this.parameters.get(parameter)!;
    }

    getPredictionDelay(): number {
      return this.predictionDelay;
    }
  }

  class FakeModel {
    constructor(
      readonly sampleRate = 16_000,
      readonly blockSize = 160,
    ) {}

    getId(): string {
      return "vad-test-model";
    }

    getOptimalSampleRate(): number {
      return this.sampleRate;
    }

    getOptimalBlockSize(sampleRate: number): number {
      return Math.round((this.blockSize * sampleRate) / this.sampleRate);
    }

    dispose(): void {}
  }

  class FakeVad {
    static constructorError: Error | null = null;
    readonly context = new FakeContext();
    readonly initializations: Array<[number, number, boolean]> = [];
    readonly blocks: number[][] = [];
    readonly predictions: Array<[number, boolean]> = [];
    initializeError: Error | null = null;
    processError: Error | null = null;
    terminateCalls = 0;

    constructor() {
      if (FakeVad.constructorError) throw FakeVad.constructorError;
      instances.push(this);
      nativeCalls.push(["vad"]);
    }

    initialize(...config: [number, number, boolean]): void {
      if (this.initializeError) throw this.initializeError;
      this.initializations.push(config);
    }

    process(block: Float32Array): void {
      if (this.processError) throw this.processError;
      this.blocks.push(Array.from(block));
      const prediction = this.predictions.shift();
      if (prediction) {
        [this.context.probability, this.context.detected] = prediction;
      }
    }

    getContext(): FakeContext {
      return this.context;
    }

    terminateSession(): void {
      this.terminateCalls += 1;
    }
  }

  return {
    FakeContext,
    FakeModel,
    FakeVad,
    VadParameter,
    instances,
    nativeCalls,
  };
});

vi.mock("@ai-coustics/aic-sdk", () => ({
  Model: sdk.FakeModel,
  Vad: sdk.FakeVad,
  VadParameter: sdk.VadParameter,
  _setSdkId: (id: number) => sdk.nativeCalls.push(["sdk_id", id]),
}));

import { VAD, VADParameters } from "../src/index.js";

initializeLogger({ pretty: false, level: "silent" });

function makeFrame(
  data: Int16Array,
  sampleRate = 16_000,
  channels = 1,
): AudioFrame {
  return new AudioFrame(data, sampleRate, channels, data.length / channels);
}

function fakeJob(): { context: JobContext; shutdownCallbacks: Array<() => Promise<void>> } {
  const shutdownCallbacks: Array<() => Promise<void>> = [];
  const context = {
    addShutdownCallback: (callback: () => Promise<void>) => shutdownCallbacks.push(callback),
  } as unknown as JobContext;
  return { context, shutdownCallbacks };
}

async function collectEvents(stream: VADStream) {
  stream.endInput();
  const events = [];
  for await (const event of stream) events.push(event);
  return events;
}

describe("VAD", () => {
  beforeEach(() => {
    sdk.instances.length = 0;
    sdk.nativeCalls.length = 0;
    sdk.FakeVad.constructorError = null;
    sdk.FakeContext.parameterErrors.clear();
  });

  it("constructs the first native VAD eagerly with LiveKit-compatible defaults", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(),
      licenseKey: "test-license",
      vadParameters: { sensitivity: 0.7 },
    });
    const native = sdk.instances[0]!;

    expect(vad).toBeInstanceOf(LiveKitVAD);
    expect(sdk.nativeCalls.slice(0, 2)).toEqual([
      ["sdk_id", 9],
      ["vad"],
    ]);
    expect(native.initializations).toEqual([]);
    expect(native.context.getParameter(sdk.VadParameter.Sensitivity)).toBe(0.7);
    expect(native.context.getParameter(sdk.VadParameter.SpeechHoldDuration)).toBe(0.25);
    expect(native.context.getParameter(sdk.VadParameter.MinimumSpeechDuration)).toBe(0.05);
    expect(vad.capabilities.updateInterval).toBe(10);
    expect(vad.minSilenceDuration).toBe(250);
    expect(vad.model).toBe("vad-test-model");
    expect(vad.provider).toBe("ai-coustics");

    await vad.close();
    expect(native.terminateCalls).toBe(1);
  });

  it("wraps native construction errors", () => {
    const sdkError = new Error("not a VAD model");
    sdk.FakeVad.constructorError = sdkError;

    expect(
      () =>
        new VAD({
          model: new sdk.FakeModel(),
          licenseKey: "test-license",
        }),
    ).toThrow("Failed to create ai-coustics VAD");

    try {
      new VAD({ model: new sdk.FakeModel(), licenseKey: "test-license" });
    } catch (error) {
      expect((error as Error & { cause?: unknown }).cause).toBe(sdkError);
    }
  });

  it("warns and remains usable when a constructor parameter is rejected", () => {
    sdk.FakeContext.parameterErrors.set(
      sdk.VadParameter.Sensitivity,
      new Error("SDK rejected parameter"),
    );
    const warning = vi.spyOn(log(), "warn");

    const vad = new VAD({
      model: new sdk.FakeModel(),
      licenseKey: "test-license",
      vadParameters: { sensitivity: 2, minimumSpeechDuration: 0.1 },
    });
    const context = sdk.instances[0]!.context;

    expect(context.getParameter(sdk.VadParameter.Sensitivity)).toBe(0.5);
    expect(context.getParameter(sdk.VadParameter.MinimumSpeechDuration)).toBe(0.1);
    expect(warning).toHaveBeenCalledWith(
      expect.objectContaining({
        plugin: "ai-coustics",
        component: "vad",
        modelProvider: "ai-coustics",
        modelName: "vad-test-model",
        parameter: "sensitivity",
        parameterValue: 2,
        errorMessage: "SDK rejected parameter",
      }),
      "VAD: parameter rejected; keeping the current value",
    );
    warning.mockRestore();
  });

  it("emits inference events and contiguous speech audio", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(),
      licenseKey: "test-license",
      vadParameters: { speechHoldDuration: 0.01 },
    });
    const stream = vad.stream();
    const native = sdk.instances[0]!;
    native.predictions.push(
      [0.1, false],
      [0.9, true],
      [0.8, true],
      [0.1, false],
    );

    stream.pushFrame(makeFrame(Int16Array.from({ length: 640 }, (_, index) => index)));
    const events = await collectEvents(stream);

    expect(events.map((event) => event.type)).toEqual([
      VADEventType.INFERENCE_DONE,
      VADEventType.INFERENCE_DONE,
      VADEventType.START_OF_SPEECH,
      VADEventType.INFERENCE_DONE,
      VADEventType.INFERENCE_DONE,
      VADEventType.END_OF_SPEECH,
    ]);
    const inferenceEvents = events.filter(
      (event) => event.type === VADEventType.INFERENCE_DONE,
    );
    expect(inferenceEvents.map((event) => event.probability)).toEqual([
      0.1, 0.9, 0.8, 0.1,
    ]);
    expect(inferenceEvents.map((event) => event.samplesIndex)).toEqual([
      160, 320, 480, 640,
    ]);
    expect(events[2]!.frames[0]!.data).toEqual(
      Int16Array.from({ length: 320 }, (_, index) => index),
    );
    expect(events.at(-1)!.frames[0]!.data).toEqual(
      Int16Array.from({ length: 640 }, (_, index) => index),
    );
    expect(native.terminateCalls).toBe(0);
  });

  it("enforces the default silence floor after short speech", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(),
      licenseKey: "test-license",
    });
    const stream = vad.stream();
    const native = sdk.instances[0]!;
    native.predictions.push([0.9, true], ...Array(25).fill([0.1, false]));

    stream.pushFrame(makeFrame(new Int16Array(26 * 160)));
    const events = await collectEvents(stream);

    const inferenceEvents = events.filter(
      (event) => event.type === VADEventType.INFERENCE_DONE,
    );
    const endEvents = events.filter(
      (event) => event.type === VADEventType.END_OF_SPEECH,
    );
    expect(endEvents).toHaveLength(1);
    expect(inferenceEvents.slice(1).every((event) => event.speaking)).toBe(true);
    expect(inferenceEvents.at(-1)!.rawAccumulatedSilence).toBeCloseTo(250);
    expect(endEvents[0]!.silenceDuration).toBeCloseTo(250);
  });

  it("uses the input sample rate and accounts for the SDK prediction delay", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(),
      licenseKey: "test-license",
      vadParameters: {
        speechHoldDuration: 0.03,
        minimumSpeechDuration: 0.02,
      },
      prefixPaddingDuration: 0,
    });
    const stream = vad.stream();
    const native = sdk.instances[0]!;
    native.context.predictionDelay = 960;
    native.predictions.push(
      [0.1, false],
      [0.1, false],
      [0.1, false],
      [0.1, false],
      [0.1, false],
      [0.9, false],
      [0.9, true],
      [0.1, false],
    );
    const samples = new Int16Array(8 * 480);
    for (let block = 0; block < 8; block += 1) {
      samples.fill(block + 1, block * 480, (block + 1) * 480);
    }

    stream.pushFrame(makeFrame(samples, 48_000));
    const events = await collectEvents(stream);
    const start = events.find(
      (event) => event.type === VADEventType.START_OF_SPEECH,
    )!;
    const end = events.find((event) => event.type === VADEventType.END_OF_SPEECH)!;

    expect(native.initializations).toEqual([[48_000, 480, false]]);
    expect(native.blocks).toHaveLength(8);
    expect(start.timestamp).toBeCloseTo(70);
    expect(start.speechDuration).toBeCloseTo(40);
    expect(start.rawAccumulatedSpeech).toBeCloseTo(40);
    expect(Array.from(start.frames[0]!.data).filter((_, index) => index % 480 === 0)).toEqual([
      4, 5, 6, 7,
    ]);
    expect(end.timestamp).toBeCloseTo(80);
    expect(end.silenceDuration).toBeCloseTo(30);
  });

  it("downmixes stereo and reblocks without resampling", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(16_000, 4),
      licenseKey: "test-license",
    });
    const stream = vad.stream();
    const native = sdk.instances[0]!;
    native.predictions.push([0, false]);

    stream.pushFrame(
      makeFrame(new Int16Array([1000, 3000, 2000, 4000, 3000, 5000, 4000, 6000]), 16_000, 2),
    );
    const events = await collectEvents(stream);

    expect(native.blocks).toHaveLength(1);
    expect(native.blocks[0]).toEqual([2000, 3000, 4000, 5000].map((sample) => sample / 32768));
    expect(events[0]!.frames[0]!.channels).toBe(1);
  });

  it("caps contiguous speech and keeps the rolling prefix current", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(10, 4),
      licenseKey: "test-license",
      vadParameters: { minimumSpeechDuration: 0 },
      prefixPaddingDuration: 200,
      maxBufferedSpeech: 500,
    });
    const stream = vad.stream();
    const native = sdk.instances[0]!;
    native.predictions.push(
      [0.9, true],
      [0.1, false],
      [0.1, false],
      [0.9, true],
      [0.1, false],
    );

    stream.pushFrame(makeFrame(Int16Array.from({ length: 20 }, (_, index) => index + 1), 10));
    const events = await collectEvents(stream);
    const starts = events.filter(
      (event) => event.type === VADEventType.START_OF_SPEECH,
    );
    const ends = events.filter((event) => event.type === VADEventType.END_OF_SPEECH);

    expect(starts).toHaveLength(2);
    expect(ends).toHaveLength(2);
    expect(Array.from(ends[0]!.frames[0]!.data)).toEqual([1, 2, 3, 4, 5, 6, 7]);
    expect(Array.from(starts[1]!.frames[0]!.data)).toEqual([
      9, 10, 11, 12, 13, 14, 15,
    ]);
  });

  it("resets stream state on flush but keeps model state", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(16_000, 4),
      licenseKey: "test-license",
    });
    const stream = vad.stream();
    const native = sdk.instances[0]!;
    native.predictions.push([0.9, true], [0.9, true]);

    stream.pushFrame(makeFrame(new Int16Array([1, 2, 3, 4, 5, 6])));
    stream.flush();
    stream.pushFrame(makeFrame(new Int16Array([7, 8])));
    const events = await collectEvents(stream);

    expect(native.blocks).toHaveLength(2);
    expect(native.blocks[1]).toEqual([5, 6, 7, 8].map((sample) => sample / 32768));
    expect(native.context.resetCount).toBe(0);
    expect(events.map((event) => event.type)).toEqual([
      VADEventType.INFERENCE_DONE,
      VADEventType.START_OF_SPEECH,
      VADEventType.INFERENCE_DONE,
      VADEventType.START_OF_SPEECH,
    ]);
  });

  it("updates idle, active and new native VADs", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(),
      licenseKey: "test-license",
    });
    const firstStream = vad.stream();
    vad.stream().close();

    vad.setParameters({ sensitivity: 0.8, speechHoldDuration: 0.6 });
    const thirdStream = vad.stream();
    const fourthStream = vad.stream();

    expect(sdk.instances).toHaveLength(3);
    for (const native of sdk.instances) {
      expect(native.context.getParameter(sdk.VadParameter.Sensitivity)).toBe(0.8);
      expect(native.context.getParameter(sdk.VadParameter.SpeechHoldDuration)).toBe(0.6);
    }
    expect(vad.minSilenceDuration).toBe(600);

    await Promise.all([
      collectEvents(firstStream),
      collectEvents(thirdStream),
      collectEvents(fourthStream),
    ]);
  });

  it("reuses the native VAD for a replacement stream without a reset", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(16_000, 4),
      licenseKey: "test-license",
    });
    const native = sdk.instances[0]!;
    native.predictions.push([0, false], [0, false]);

    const firstStream = vad.stream();
    firstStream.pushFrame(makeFrame(new Int16Array([1, 2, 3, 4, 5, 6])));
    await collectEvents(firstStream);
    const secondStream = vad.stream();
    secondStream.pushFrame(makeFrame(new Int16Array([7, 8])));
    await collectEvents(secondStream);

    expect(sdk.instances).toHaveLength(1);
    expect(native.context.resetCount).toBe(0);
    expect(native.initializations).toHaveLength(1);
    expect(native.blocks[1]).toEqual([5, 6, 7, 8].map((sample) => sample / 32768));
    expect(native.terminateCalls).toBe(0);
  });

  it("uses independent native VADs for concurrent streams", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(),
      licenseKey: "test-license",
    });
    const firstStream = vad.stream();
    const secondStream = vad.stream();

    firstStream.pushFrame(makeFrame(new Int16Array(160)));
    secondStream.pushFrame(makeFrame(new Int16Array(160)));
    await Promise.all([collectEvents(firstStream), collectEvents(secondStream)]);

    expect(sdk.instances).toHaveLength(2);
    expect(sdk.instances.map((native) => native.blocks.length)).toEqual([1, 1]);
  });

  it("returns the native VAD to the pool synchronously on close", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(),
      licenseKey: "test-license",
    });

    vad.stream().close();
    const stream = vad.stream();

    expect(sdk.instances).toHaveLength(1);
    await collectEvents(stream);
  });

  it("terminates every native VAD on close", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(),
      licenseKey: "test-license",
    });
    vad.stream().close();
    vad.stream();
    vad.stream();

    await vad.close();

    expect(sdk.instances.map((native) => native.terminateCalls)).toEqual([1, 1]);
  });

  it("terminates idle and later released native VADs on job shutdown", async () => {
    const job = fakeJob();
    const vad = new VAD({
      model: new sdk.FakeModel(),
      licenseKey: "test-license",
    });
    vad.stream().close();

    const [firstStream, secondStream] = runWithJobContext(job.context, () => [
      vad.stream(),
      vad.stream(),
    ]);
    firstStream!.close();

    await job.shutdownCallbacks[0]!();
    expect(sdk.instances.map((native) => native.terminateCalls)).toEqual([1, 0]);
    secondStream!.close();

    expect(job.shutdownCallbacks).toHaveLength(1);
    expect(sdk.instances.map((native) => native.terminateCalls)).toEqual([1, 1]);
  });

  it("rejects instead of throwing when terminating fails on job shutdown", async () => {
    const job = fakeJob();
    const vad = new VAD({
      model: new sdk.FakeModel(),
      licenseKey: "test-license",
    });
    runWithJobContext(job.context, () => vad.stream()).close();
    sdk.instances[0]!.terminateSession = () => {
      throw new Error("terminate failed");
    };

    let shutdown: Promise<void> | undefined;
    expect(() => {
      shutdown = job.shutdownCallbacks[0]!();
    }).not.toThrow();
    await expect(shutdown).rejects.toThrow("terminate failed");
  });

  it("adds model and audio-format context to initialization errors", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(),
      licenseKey: "test-license",
    });
    const stream = vad.stream();
    const native = sdk.instances[0]!;
    native.initializeError = new Error("unsupported configuration");
    stream.pushFrame(makeFrame(new Int16Array(480), 48_000));

    await expect(collectEvents(stream)).rejects.toThrow(
      "ai-coustics VAD initialization failed (model=vad-test-model, sampleRate=48000, blockSize=480): unsupported configuration",
    );
    expect(native.terminateCalls).toBe(1);

    await collectEvents(vad.stream());
    expect(sdk.instances).toHaveLength(2);
  });

  it("adds model and audio-format context to inference errors", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(),
      licenseKey: "test-license",
    });
    const stream = vad.stream();
    const native = sdk.instances[0]!;
    native.processError = new Error("native failure");
    const errorLog = vi.spyOn(log(), "error");
    stream.pushFrame(makeFrame(new Int16Array(480), 48_000));

    await expect(collectEvents(stream)).rejects.toThrow(
      "ai-coustics VAD inference failed (model=vad-test-model, sampleRate=48000, blockSize=480): native failure",
    );
    expect(native.terminateCalls).toBe(1);
    expect(errorLog).toHaveBeenCalledWith(
      expect.objectContaining({
        plugin: "ai-coustics",
        component: "vad",
        modelProvider: "ai-coustics",
        modelName: "vad-test-model",
        sampleRate: 48_000,
        blockSize: 480,
        errorType: "Error",
        errorMessage: expect.stringContaining("native failure"),
        error: expect.any(Error),
      }),
      "VAD: stream failed",
    );
    errorLog.mockRestore();

    await collectEvents(vad.stream());
    expect(sdk.instances).toHaveLength(2);
  });

  it.each([
    [{ prefixPaddingDuration: -1 }, "prefixPaddingDuration"],
    [{ maxBufferedSpeech: 0 }, "maxBufferedSpeech"],
  ])("validates buffering option %j", (options, message) => {
    expect(
      () =>
        new VAD({
          model: new sdk.FakeModel(),
          licenseKey: "test-license",
          ...options,
        }),
    ).toThrow(message);
  });
});
