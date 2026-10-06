import {
  VAD as LiveKitVAD,
  VADEventType,
  initializeLogger,
  log,
  type VADStream,
} from "@livekit/agents";
import { AudioFrame, FrameProcessor } from "@livekit/rtc-node";
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
    readonly queriedSampleRates: number[] = [];

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
      this.queriedSampleRates.push(sampleRate);
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

import { FrameProcessorChain, VAD, VADParameters } from "../src/index.js";

initializeLogger({ pretty: false, level: "silent" });

function makeFrame(
  data: Int16Array,
  sampleRate = 16_000,
  channels = 1,
): AudioFrame {
  return new AudioFrame(data, sampleRate, channels, data.length / channels);
}

function pushProcessed(vad: VAD, stream: VADStream, frame: AudioFrame): void {
  stream.pushFrame(vad.processor.process(frame));
}

class MutingProcessor extends FrameProcessor<AudioFrame> {
  isEnabled(): boolean {
    return true;
  }

  setEnabled(): void {}

  process(frame: AudioFrame): AudioFrame {
    return new AudioFrame(
      new Int16Array(frame.samplesPerChannel),
      frame.sampleRate,
      1,
      frame.samplesPerChannel,
      frame.userdata,
    );
  }

  close(): void {}
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

    pushProcessed(
      vad,
      stream,
      makeFrame(Int16Array.from({ length: 640 }, (_, index) => index)),
    );
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

    pushProcessed(vad, stream, makeFrame(new Int16Array(26 * 160)));
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

    pushProcessed(vad, stream, makeFrame(samples, 48_000));
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

    pushProcessed(
      vad,
      stream,
      makeFrame(new Int16Array([1000, 3000, 2000, 4000, 3000, 5000, 4000, 6000]), 16_000, 2),
    );
    const events = await collectEvents(stream);

    expect(native.blocks).toHaveLength(1);
    expect(native.blocks[0]).toEqual([2000, 3000, 4000, 5000].map((sample) => sample / 32768));
    expect(events[0]!.frames[0]!.channels).toBe(1);
  });

  it("reinitializes only when the input sample rate changes", () => {
    const model = new sdk.FakeModel(16_000, 4);
    const vad = new VAD({ model, licenseKey: "test-license" });
    const native = sdk.instances[0]!;
    model.queriedSampleRates.length = 0;

    for (const [samples, sampleRate] of [
      [3, 16_000],
      [7, 16_000],
      [2, 16_000],
      [12, 48_000],
      [12, 48_000],
    ] as const) {
      vad.processor.process(makeFrame(new Int16Array(samples), sampleRate));
    }

    expect(model.queriedSampleRates).toEqual([16_000, 48_000]);
    expect(native.initializations).toEqual([
      [16_000, 4, false],
      [48_000, 12, false],
    ]);
    expect(native.blocks.map((block) => block.length)).toEqual([4, 4, 4, 12, 12]);
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

    pushProcessed(
      vad,
      stream,
      makeFrame(Int16Array.from({ length: 20 }, (_, index) => index + 1), 10),
    );
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

  it("resets stream state on flush without repeating shared inference", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(16_000, 4),
      licenseKey: "test-license",
    });
    const stream = vad.stream();
    const native = sdk.instances[0]!;
    native.predictions.push([0, false]);

    pushProcessed(vad, stream, makeFrame(new Int16Array([1, 2])));
    stream.flush();
    pushProcessed(vad, stream, makeFrame(new Int16Array([3, 4, 5, 6])));
    await collectEvents(stream);

    expect(native.blocks).toHaveLength(1);
    expect(native.blocks[0]).toEqual([1, 2, 3, 4].map((sample) => sample / 32768));
    expect(native.context.resetCount).toBe(0);
  });

  it("updates active and future streams", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(),
      licenseKey: "test-license",
    });
    const firstStream = vad.stream();
    const native = sdk.instances[0]!;

    vad.setParameters({ sensitivity: 0.8, speechHoldDuration: 0.6 });
    const secondStream = vad.stream();

    expect(sdk.instances).toHaveLength(1);
    expect(native.context.getParameter(sdk.VadParameter.Sensitivity)).toBe(0.8);
    expect(native.context.getParameter(sdk.VadParameter.SpeechHoldDuration)).toBe(0.6);
    expect(vad.minSilenceDuration).toBe(600);

    await Promise.all([collectEvents(firstStream), collectEvents(secondStream)]);
  });

  it("shares one inference result across multiple streams", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(),
      licenseKey: "test-license",
    });
    const firstStream = vad.stream();
    const secondStream = vad.stream();
    const native = sdk.instances[0]!;
    native.predictions.push([0.9, true], [0.1, false]);
    const processed = vad.processor.process(makeFrame(new Int16Array(320)));
    firstStream.pushFrame(processed);
    secondStream.pushFrame(processed);

    const [firstEvents, secondEvents] = await Promise.all([
      collectEvents(firstStream),
      collectEvents(secondStream),
    ]);
    expect(sdk.instances).toHaveLength(1);
    expect(native.blocks).toHaveLength(2);
    expect(firstEvents.map((event) => event.type)).toEqual(
      secondEvents.map((event) => event.type),
    );
    expect(firstEvents.map((event) => event.probability)).toEqual(
      secondEvents.map((event) => event.probability),
    );
  });

  it("logs missing metadata with the standard VAD context", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(),
      licenseKey: "test-license",
    });
    vad.processor.onStreamInfoUpdated({
      roomName: "room",
      participantIdentity: "participant",
      publicationSid: "TR_test",
    });
    const stream = vad.stream();
    const errorLog = vi.spyOn(log(), "error");

    for (let index = 0; index < 10; index += 1) {
      stream.pushFrame(makeFrame(new Int16Array(160)));
    }
    await collectEvents(stream);

    expect(errorLog).toHaveBeenCalledWith(
      expect.objectContaining({
        plugin: "ai-coustics",
        component: "vad",
        modelProvider: "ai-coustics",
        modelName: "vad-test-model",
        roomName: "room",
        participantIdentity: "participant",
        publicationSid: "TR_test",
        missingMetadataFrames: 10,
      }),
      expect.stringContaining("VAD: no inference metadata found"),
    );
    errorLog.mockRestore();
  });

  it("keeps VAD audio when a downstream processor replaces the frame", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(16_000, 4),
      licenseKey: "test-license",
    });
    const stream = vad.stream();
    const native = sdk.instances[0]!;
    native.predictions.push([0.9, true]);
    const chain = new FrameProcessorChain(vad.processor, new MutingProcessor());
    const raw = new Int16Array([1000, 2000, 3000, 4000]);

    const output = chain.process(makeFrame(raw));
    stream.pushFrame(output);
    const events = await collectEvents(stream);

    expect(native.blocks[0]).toEqual(Array.from(raw, (sample) => sample / 32768));
    expect(Array.from(output.data)).toEqual([0, 0, 0, 0]);
    expect(Array.from(events[0]!.frames[0]!.data)).toEqual(Array.from(raw));
  });

  it("keeps the shared native session alive until the processor closes", () => {
    const vad = new VAD({
      model: new sdk.FakeModel(),
      licenseKey: "test-license",
    });
    const stream = vad.stream();
    const native = sdk.instances[0]!;

    stream.close();
    stream.close();

    expect(native.terminateCalls).toBe(0);
    vad.processor.close();
    vad.processor.close();
    expect(native.terminateCalls).toBe(1);
  });

  it("adds model and audio-format context to initialization errors", async () => {
    const vad = new VAD({
      model: new sdk.FakeModel(),
      licenseKey: "test-license",
    });
    const stream = vad.stream();
    const native = sdk.instances[0]!;
    native.initializeError = new Error("unsupported configuration");
    pushProcessed(vad, stream, makeFrame(new Int16Array(480), 48_000));

    await expect(collectEvents(stream)).rejects.toThrow(
      "ai-coustics VAD initialization failed (model=vad-test-model, sampleRate=48000, blockSize=480): unsupported configuration",
    );
    expect(native.terminateCalls).toBe(0);
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
    pushProcessed(vad, stream, makeFrame(new Int16Array(480), 48_000));

    await expect(collectEvents(stream)).rejects.toThrow(
      "ai-coustics VAD inference failed (model=vad-test-model, sampleRate=48000, blockSize=480): native failure",
    );
    expect(native.terminateCalls).toBe(0);
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
