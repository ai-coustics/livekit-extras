import {
  VAD as LiveKitVAD,
  VADStream as LiveKitVADStream,
  VADEventType,
  getJobContext,
} from "@livekit/agents";
import {
  type Model,
  Vad as AicVad,
  type VadContext,
  VadParameter as AicVadParameter,
  _setSdkId as setSdkId,
} from "@ai-coustics/aic-sdk";
import { AudioFrame } from "@livekit/rtc-node";

import { errorDetail, pcm16ToFloat32, resolveLicenseKey } from "./common.js";
import { writeLog } from "./log.js";

const DEFAULT_SPEECH_HOLD_DURATION_SECONDS = 0.25;
const DEFAULT_MINIMUM_SPEECH_DURATION_SECONDS = 0.05;
const DEFAULT_PREFIX_PADDING_DURATION_MS = 500;
const DEFAULT_MAX_BUFFERED_SPEECH_MS = 60_000;
const SLOW_INFERENCE_BACKLOG_THRESHOLD_MS = 200;
const SLOW_WARNING_INTERVAL_MS = 10_000;

export interface VADParameters {
  sensitivity?: number;
  speechHoldDuration?: number;
  minimumSpeechDuration?: number;
}

export interface VADOptions {
  model: Model;
  licenseKey?: string;
  vadParameters?: VADParameters;
  prefixPaddingDuration?: number;
  maxBufferedSpeech?: number;
}

interface InferenceResult {
  pcm: Int16Array;
  sampleRate: number;
  probability: number;
  detected: boolean;
  sensitivity: number;
  speechHoldDuration: number;
  minimumSpeechDuration: number;
  inferenceDurationMs: number;
  predictionDelaySamples: number;
}

/** One SDK VAD and its inference state, handed between streams without a reset. */
class NativeVad {
  readonly context: VadContext;
  private readonly model: Model;
  private readonly modelId: string;
  private readonly nativeVad: AicVad;
  private format: [number, number] | null = null;
  private inferenceBuffer = new Int16Array(0);
  private predictionDelaySamples = 0;
  private processingBacklogMs = 0;
  private slowWarningActive = false;
  private lastSlowWarning = 0;

  constructor(model: Model, licenseKey: string) {
    setSdkId(9);
    let nativeVad: AicVad;
    try {
      nativeVad = new AicVad(model, licenseKey);
    } catch (error) {
      throw new Error(`Failed to create ai-coustics VAD: ${errorDetail(error)}`, {
        cause: error,
      });
    }
    this.model = model;
    this.modelId = model.getId();
    this.nativeVad = nativeVad;
    this.context = nativeVad.getContext();
  }

  terminate(): void {
    this.nativeVad.terminateSession();
  }

  processFrame(frame: AudioFrame): readonly InferenceResult[] {
    const nativeVad = this.nativeVad;
    const context = this.context;

    const expectedSamples = frame.samplesPerChannel * frame.channels;
    if (frame.data.length !== expectedSamples) {
      throw new Error(
        `AudioFrame contains ${frame.data.length} samples, expected ${expectedSamples}`,
      );
    }
    const mono = new Int16Array(frame.samplesPerChannel);
    if (frame.channels === 1) {
      mono.set(frame.data);
    } else {
      for (let sample = 0; sample < frame.samplesPerChannel; sample += 1) {
        let sum = 0;
        for (let channel = 0; channel < frame.channels; channel += 1) {
          sum += frame.data[sample * frame.channels + channel]!;
        }
        mono[sample] = Math.round(sum / frame.channels);
      }
    }

    const inferenceBlockSize = this.model.getOptimalBlockSize(frame.sampleRate);
    const nextFormat: [number, number] = [frame.sampleRate, inferenceBlockSize];
    if (
      !this.format ||
      this.format[0] !== nextFormat[0] ||
      this.format[1] !== nextFormat[1]
    ) {
      try {
        nativeVad.initialize(frame.sampleRate, inferenceBlockSize, false);
      } catch (error) {
        throw new Error(
          `ai-coustics VAD initialization failed (model=${this.modelId}, ` +
            `sampleRate=${frame.sampleRate}, blockSize=${inferenceBlockSize}): ` +
            errorDetail(error),
          { cause: error },
        );
      }
      this.format = nextFormat;
      this.inferenceBuffer = new Int16Array(0);
      this.predictionDelaySamples = context.getPredictionDelay();
    }

    const buffered = new Int16Array(this.inferenceBuffer.length + mono.length);
    buffered.set(this.inferenceBuffer);
    buffered.set(mono, this.inferenceBuffer.length);
    this.inferenceBuffer = buffered;

    const results: InferenceResult[] = [];
    while (this.inferenceBuffer.length >= inferenceBlockSize) {
      const pcm = this.inferenceBuffer.slice(0, inferenceBlockSize);
      this.inferenceBuffer = this.inferenceBuffer.slice(inferenceBlockSize);
      const started = performance.now();
      try {
        nativeVad.process(pcm16ToFloat32(pcm));
      } catch (error) {
        throw new Error(
          `ai-coustics VAD inference failed (model=${this.modelId}, ` +
            `sampleRate=${frame.sampleRate}, blockSize=${inferenceBlockSize}): ` +
            errorDetail(error),
          { cause: error },
        );
      }
      const completed = performance.now();
      const inferenceDurationMs = completed - started;
      const blockDurationMs = (inferenceBlockSize / frame.sampleRate) * 1000;
      this.processingBacklogMs = Math.max(
        0,
        this.processingBacklogMs + inferenceDurationMs - blockDurationMs,
      );
      results.push({
        pcm,
        sampleRate: frame.sampleRate,
        probability: context.getRawVadProbability(),
        detected: context.isSpeechDetected(),
        sensitivity: context.getParameter(AicVadParameter.Sensitivity),
        speechHoldDuration: context.getParameter(
          AicVadParameter.SpeechHoldDuration,
        ),
        minimumSpeechDuration: context.getParameter(
          AicVadParameter.MinimumSpeechDuration,
        ),
        inferenceDurationMs,
        predictionDelaySamples: this.predictionDelaySamples,
      });

      if (this.processingBacklogMs === 0) {
        this.slowWarningActive = false;
      } else if (
        this.processingBacklogMs >= SLOW_INFERENCE_BACKLOG_THRESHOLD_MS &&
        (!this.slowWarningActive ||
          completed - this.lastSlowWarning >= SLOW_WARNING_INTERVAL_MS)
      ) {
        this.slowWarningActive = true;
        this.lastSlowWarning = completed;
        writeLog(
          "warn",
          "vad",
          "inference falling behind realtime",
          this.diagnosticFields({
            inferenceDurationMs,
            blockDurationMs,
            realtimeFactor: inferenceDurationMs / blockDurationMs,
            processingBacklogMs: this.processingBacklogMs,
            sampleRate: frame.sampleRate,
            blockSize: inferenceBlockSize,
          }),
        );
      }
    }
    return results;
  }

  diagnosticFields(fields: Record<string, unknown> = {}): Record<string, unknown> {
    const diagnostics: Record<string, unknown> = {
      modelProvider: "ai-coustics",
      modelName: this.modelId,
    };
    if (this.format) {
      Object.assign(diagnostics, {
        sampleRate: this.format[0],
        blockSize: this.format[1],
      });
    }
    return Object.assign(diagnostics, fields);
  }
}

/**
 * LiveKit VAD whose streams run SDK inference on a pool of native VADs. Use one VAD per
 * AgentSession: the pool keeps model history, so all its streams must see the same audio.
 */
export class VAD extends LiveKitVAD {
  label = "ai-coustics.VAD";

  private readonly modelId: string;
  private readonly sdkModel: Model;
  private readonly licenseKey: string;
  private readonly prefixPaddingDuration: number;
  private readonly maxBufferedSpeech: number;
  private readonly parameters: VADParameters = {
    speechHoldDuration: DEFAULT_SPEECH_HOLD_DURATION_SECONDS,
    minimumSpeechDuration: DEFAULT_MINIMUM_SPEECH_DURATION_SECONDS,
  };
  private readonly streams = new Set<WeakRef<AicVADStream>>();
  private idle: NativeVad[];
  private readonly active = new Set<NativeVad>();
  private draining = false;
  private shutdownRegistered = false;

  constructor(options: VADOptions) {
    const prefixPaddingDuration =
      options.prefixPaddingDuration ?? DEFAULT_PREFIX_PADDING_DURATION_MS;
    const maxBufferedSpeech =
      options.maxBufferedSpeech ?? DEFAULT_MAX_BUFFERED_SPEECH_MS;
    if (prefixPaddingDuration < 0) {
      throw new Error("prefixPaddingDuration must be greater than or equal to zero");
    }
    if (maxBufferedSpeech <= 0) {
      throw new Error("maxBufferedSpeech must be greater than zero");
    }

    const modelSampleRate = options.model.getOptimalSampleRate();
    const modelBlockSize = options.model.getOptimalBlockSize(modelSampleRate);
    super({ updateInterval: (modelBlockSize / modelSampleRate) * 1000 });

    this.modelId = options.model.getId();
    this.sdkModel = options.model;
    this.licenseKey = resolveLicenseKey(options.licenseKey);
    this.prefixPaddingDuration = prefixPaddingDuration;
    this.maxBufferedSpeech = maxBufferedSpeech;
    this.idle = [new NativeVad(options.model, this.licenseKey)];
    this.setParameters(this.parameters);
    if (options.vadParameters) this.setParameters(options.vadParameters);
  }

  get model(): string {
    return this.modelId;
  }

  get provider(): string {
    return "ai-coustics";
  }

  override get minSilenceDuration(): number {
    return (
      (this.parameters.speechHoldDuration ?? DEFAULT_SPEECH_HOLD_DURATION_SECONDS) * 1000
    );
  }

  override stream(): LiveKitVADStream {
    const stream = new AicVADStream(this, {
      native: this.acquire(),
      release: (native, discard) => this.release(native, discard),
      prefixPaddingDuration: this.prefixPaddingDuration,
      maxBufferedSpeech: this.maxBufferedSpeech,
    });
    this.streams.add(new WeakRef(stream));
    return stream;
  }

  setParameters(parameters: VADParameters): void {
    if (
      parameters.sensitivity !== undefined &&
      this.applyParameter(
        AicVadParameter.Sensitivity,
        "sensitivity",
        parameters.sensitivity,
      )
    ) {
      this.parameters.sensitivity = parameters.sensitivity;
    }
    if (
      parameters.speechHoldDuration !== undefined &&
      this.applyParameter(
        AicVadParameter.SpeechHoldDuration,
        "speechHoldDuration",
        parameters.speechHoldDuration,
      )
    ) {
      this.parameters.speechHoldDuration = parameters.speechHoldDuration;
    }
    if (
      parameters.minimumSpeechDuration !== undefined &&
      this.applyParameter(
        AicVadParameter.MinimumSpeechDuration,
        "minimumSpeechDuration",
        parameters.minimumSpeechDuration,
      )
    ) {
      this.parameters.minimumSpeechDuration = parameters.minimumSpeechDuration;
    }
  }

  override async close(): Promise<void> {
    for (const reference of this.streams) reference.deref()?.close();
    this.streams.clear();
    await this.drain();
  }

  private applyParameter(parameter: number, name: string, value: number): boolean {
    // All native VADs share one model, so they accept and reject the same values.
    for (const native of [...this.idle, ...this.active]) {
      try {
        native.context.setParameter(parameter, value);
      } catch (error) {
        writeLog(
          "warn",
          "vad",
          "parameter rejected; keeping the current value",
          native.diagnosticFields({
            parameter: name,
            parameterValue: value,
            errorType: error instanceof Error ? error.name : typeof error,
            errorMessage: errorDetail(error),
          }),
          error,
        );
        return false;
      }
    }
    return true;
  }

  private acquire(): NativeVad {
    if (!this.shutdownRegistered) {
      const jobContext = getJobContext(false);
      if (jobContext) {
        jobContext.addShutdownCallback(() => this.drain());
        this.shutdownRegistered = true;
      }
    }

    const native = this.idle.pop() ?? this.createNative();
    this.active.add(native);
    return native;
  }

  private createNative(): NativeVad {
    const native = new NativeVad(this.sdkModel, this.licenseKey);
    const values: [AicVadParameter, number | undefined][] = [
      [AicVadParameter.Sensitivity, this.parameters.sensitivity],
      [AicVadParameter.SpeechHoldDuration, this.parameters.speechHoldDuration],
      [AicVadParameter.MinimumSpeechDuration, this.parameters.minimumSpeechDuration],
    ];
    for (const [parameter, value] of values) {
      if (value !== undefined) native.context.setParameter(parameter, value);
    }
    return native;
  }

  private release(native: NativeVad, discard: boolean): void {
    this.active.delete(native);
    if (discard || this.draining) native.terminate();
    else this.idle.push(native);
  }

  // Async so that a terminate failure rejects instead of throwing synchronously into
  // LiveKit's shutdown loop, which would skip its remaining shutdown steps.
  private async drain(): Promise<void> {
    this.draining = true;
    const idle = this.idle;
    this.idle = [];
    for (const native of idle) native.terminate();
  }
}

interface StreamOptions {
  native: NativeVad;
  release: (native: NativeVad, discard: boolean) => void;
  prefixPaddingDuration: number;
  maxBufferedSpeech: number;
}

class AicVADStream extends LiveKitVADStream {
  private readonly native: NativeVad;
  private readonly onRelease: (native: NativeVad, discard: boolean) => void;
  private readonly prefixPaddingDuration: number;
  private readonly maxBufferedSpeech: number;
  private released = false;
  private outputFinished = false;
  private pumpError: unknown;

  constructor(vad: VAD, options: StreamOptions) {
    super(vad);
    this.native = options.native;
    this.onRelease = options.release;
    this.prefixPaddingDuration = options.prefixPaddingDuration;
    this.maxBufferedSpeech = options.maxBufferedSpeech;
    void this.pump()
      .then(async () => {
        this.releaseNative(false);
        await this.finishOutput();
      })
      .catch(async (error: unknown) => {
        this.pumpError = error;
        writeLog(
          "error",
          "vad",
          "stream failed",
          this.native.diagnosticFields({
            errorType: error instanceof Error ? error.name : typeof error,
            errorMessage: errorDetail(error),
          }),
          error,
        );
        this.releaseNative(true);
        await this.finishOutput();
      });
  }

  override endInput(): void {
    if (this.inputClosed) throw new Error("Input is closed");
    if (this.closed) throw new Error("Stream is closed");
    this.inputClosed = true;
    void this.inputWriter.close();
  }

  // LiveKit starts the next stream right after close() returns, so the native VAD must be back
  // in the pool before then.
  override close(): void {
    if (this.closed) return;
    this.closed = true;
    this.releaseNative(false);
    void this.detachInputStream();
    void this.inputReader.cancel();
    void this.finishOutput();
  }

  override async next(): Promise<IteratorResult<import("@livekit/agents").VADEvent>> {
    const result = await super.next();
    if (result.done && this.pumpError !== undefined) {
      const error = this.pumpError;
      this.pumpError = undefined;
      throw error;
    }
    return result;
  }

  private releaseNative(discard: boolean): void {
    if (this.released) return;
    this.released = true;
    this.onRelease(this.native, discard);
  }

  private async finishOutput(): Promise<void> {
    if (this.outputFinished) return;
    this.outputFinished = true;
    try {
      await this.outputWriter.close();
    } catch {
      // The output can already be closed during AgentSession teardown.
    }
  }

  private async pump(): Promise<void> {
    let inputSampleRate = 0;
    const prefixFrames: AudioFrame[] = [];
    let prefixSamples = 0;
    let speechBuffer: Int16Array | null = null;
    let speechSamples = 0;
    let speechBufferFull = false;
    let speaking = false;
    let speechDurationMs = 0;
    let silenceDurationMs = 0;
    let rawSpeechDurationMs = 0;
    let rawSilenceDurationMs = 0;
    let currentSample = 0;
    let timestampMs = 0;

    const resetState = () => {
      inputSampleRate = 0;
      prefixFrames.length = 0;
      prefixSamples = 0;
      speechBuffer = null;
      speechSamples = 0;
      speechBufferFull = false;
      speaking = false;
      speechDurationMs = 0;
      silenceDurationMs = 0;
      rawSpeechDurationMs = 0;
      rawSilenceDurationMs = 0;
      currentSample = 0;
      timestampMs = 0;
    };

    const updatePrefixBuffer = (frame: AudioFrame, result: InferenceResult) => {
      prefixFrames.push(frame);
      prefixSamples += frame.samplesPerChannel;
      const activationSamples = Math.max(
        frame.samplesPerChannel,
        Math.trunc(result.minimumSpeechDuration * frame.sampleRate),
      );
      const targetSamples =
        Math.trunc((this.prefixPaddingDuration * frame.sampleRate) / 1000) +
        result.predictionDelaySamples +
        activationSamples;
      while (
        prefixFrames.length > 1 &&
        prefixSamples - prefixFrames[0]!.samplesPerChannel >= targetSamples
      ) {
        prefixSamples -= prefixFrames.shift()!.samplesPerChannel;
      }
    };

    const appendSpeechAudio = (frame: AudioFrame, predictionDelaySamples: number) => {
      if (!speechBuffer) {
        const maxSamples =
          Math.trunc(
            ((this.prefixPaddingDuration + this.maxBufferedSpeech) * frame.sampleRate) /
              1000,
          ) + predictionDelaySamples;
        speechBuffer = new Int16Array(maxSamples);
      }
      const remainingSamples = Math.max(0, speechBuffer.length - speechSamples);
      const copiedSamples = Math.min(frame.samplesPerChannel, remainingSamples);
      if (copiedSamples > 0) {
        speechBuffer.set(frame.data.subarray(0, copiedSamples), speechSamples);
        speechSamples += copiedSamples;
      }
      if (copiedSamples < frame.samplesPerChannel && !speechBufferFull) {
        speechBufferFull = true;
        writeLog(
          "warn",
          "vad",
          "maximum buffered speech reached; ignoring further audio",
          this.native.diagnosticFields({
            maxBufferedSpeechMs: this.maxBufferedSpeech,
          }),
        );
      }
    };

    const speechEventFrames = (): AudioFrame[] => {
      if (!speechBuffer || speechSamples === 0) return [];
      return [
        new AudioFrame(
          speechBuffer.slice(0, speechSamples),
          inputSampleRate,
          1,
          speechSamples,
        ),
      ];
    };

    while (!this.closed) {
      const { done, value } = await this.inputReader.read();
      // A read can resolve just before close(); by then the native VAD may belong to the next
      // stream.
      if (done || this.released) break;
      if (typeof value === "symbol") {
        resetState();
        continue;
      }

      for (const result of this.native.processFrame(value)) {
        if (inputSampleRate === 0) inputSampleRate = result.sampleRate;
        else if (result.sampleRate !== inputSampleRate) {
          writeLog(
            "error",
            "vad",
            "received frame with a different sample rate",
            this.native.diagnosticFields({
              sampleRate: inputSampleRate,
              receivedSampleRate: result.sampleRate,
            }),
          );
          continue;
        }

        const inferenceAudio = new AudioFrame(
          result.pcm,
          result.sampleRate,
          1,
          result.pcm.length,
        );
        const blockDurationMs =
          (inferenceAudio.samplesPerChannel / inputSampleRate) * 1000;
        const predictionDelayMs =
          (result.predictionDelaySamples / inputSampleRate) * 1000;
        currentSample += inferenceAudio.samplesPerChannel;
        timestampMs += blockDurationMs;

        if (result.probability >= result.sensitivity) {
          rawSpeechDurationMs += blockDurationMs;
          rawSilenceDurationMs = 0;
        } else {
          rawSilenceDurationMs += blockDurationMs;
          rawSpeechDurationMs = 0;
        }
        const alignedRawSpeechDurationMs =
          rawSpeechDurationMs > 0 ? rawSpeechDurationMs + predictionDelayMs : 0;
        const alignedRawSilenceDurationMs =
          rawSilenceDurationMs > 0 ? rawSilenceDurationMs + predictionDelayMs : 0;
        if (speaking) speechDurationMs += blockDurationMs;
        else silenceDurationMs += blockDurationMs;
        updatePrefixBuffer(inferenceAudio, result);
        if (speaking) appendSpeechAudio(inferenceAudio, result.predictionDelaySamples);

        this.sendVADEvent({
          type: VADEventType.INFERENCE_DONE,
          samplesIndex: currentSample,
          timestamp: timestampMs,
          speechDuration: speechDurationMs,
          silenceDuration: silenceDurationMs,
          frames: [inferenceAudio],
          probability: result.probability,
          inferenceDuration: result.inferenceDurationMs,
          speaking,
          rawAccumulatedSilence: alignedRawSilenceDurationMs,
          rawAccumulatedSpeech: alignedRawSpeechDurationMs,
        });

        if (result.detected && !speaking) {
          speaking = true;
          silenceDurationMs = 0;
          speechDurationMs = Math.max(blockDurationMs, alignedRawSpeechDurationMs);
          speechBuffer = null;
          speechSamples = 0;
          speechBufferFull = false;
          for (const prefixFrame of prefixFrames) {
            appendSpeechAudio(prefixFrame, result.predictionDelaySamples);
          }
          this.sendVADEvent({
            type: VADEventType.START_OF_SPEECH,
            samplesIndex: currentSample,
            timestamp: timestampMs,
            speechDuration: speechDurationMs,
            silenceDuration: 0,
            frames: speechEventFrames(),
            probability: result.probability,
            inferenceDuration: result.inferenceDurationMs,
            speaking: true,
            rawAccumulatedSilence: 0,
            rawAccumulatedSpeech: alignedRawSpeechDurationMs,
          });
        // The SDK hold is a rolling-majority window, so short utterances can make
        // detected drop before a full hold interval of raw silence has elapsed.
        } else if (
          !result.detected &&
          speaking &&
          alignedRawSilenceDurationMs >= result.speechHoldDuration * 1000
        ) {
          speaking = false;
          silenceDurationMs = alignedRawSilenceDurationMs;
          const completedSpeechDurationMs = Math.max(
            0,
            speechDurationMs - silenceDurationMs,
          );
          this.sendVADEvent({
            type: VADEventType.END_OF_SPEECH,
            samplesIndex: currentSample,
            timestamp: timestampMs,
            speechDuration: completedSpeechDurationMs,
            silenceDuration: silenceDurationMs,
            frames: speechEventFrames(),
            probability: result.probability,
            inferenceDuration: result.inferenceDurationMs,
            speaking: false,
            rawAccumulatedSilence: alignedRawSilenceDurationMs,
            rawAccumulatedSpeech: 0,
          });
          speechDurationMs = 0;
          speechBuffer = null;
          speechSamples = 0;
          speechBufferFull = false;
        }
      }
    }
  }
}
