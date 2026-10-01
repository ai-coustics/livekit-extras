import {
  VAD as LiveKitVAD,
  VADStream as LiveKitVADStream,
  VADEventType,
} from "@livekit/agents";
import {
  type Model,
  Vad as AicVad,
  type VadContext,
  VadParameter as AicVadParameter,
  _setSdkId as setSdkId,
} from "@ai-coustics/aic-sdk";
import {
  AudioFrame,
  FrameProcessor,
  type FrameProcessorStreamInfo,
} from "@livekit/rtc-node";

import { errorDetail, monoPcm16, pcm16ToFloat32, resolveLicenseKey } from "./common.js";
import { writeLog } from "./log.js";

const FRAME_USERDATA_VAD_ATTRIBUTE = "ai_coustics.vad";
const DEFAULT_SPEECH_HOLD_DURATION_SECONDS = 0.25;
const DEFAULT_MINIMUM_SPEECH_DURATION_SECONDS = 0.05;
const DEFAULT_PREFIX_PADDING_DURATION_MS = 500;
const DEFAULT_MAX_BUFFERED_SPEECH_MS = 60_000;
const SLOW_INFERENCE_BACKLOG_THRESHOLD_MS = 200;
const SLOW_WARNING_INTERVAL_MS = 10_000;
const MISSING_METADATA_WARNING_FRAMES = 10;

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

interface InferenceError {
  message: string;
  cause: unknown;
}

interface FrameMetadata {
  results: readonly InferenceResult[];
  error?: InferenceError;
}

/** Pass-through processor that runs one SDK VAD and annotates input frames. */
export class VADProcessor extends FrameProcessor<AudioFrame> {
  private readonly model: Model;
  private readonly modelId: string;
  private nativeVad: AicVad | null;
  private context: VadContext | null;
  private processorEnabled = true;
  private closed = false;
  private format: [number, number] | null = null;
  private inferenceBuffer = new Int16Array(0);
  private predictionDelaySamples = 0;
  private processingBacklogMs = 0;
  private slowWarningActive = false;
  private lastSlowWarning = 0;
  private streamInfo: FrameProcessorStreamInfo | null = null;

  constructor(model: Model, licenseKey: string) {
    super();
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

  get sdkContext(): VadContext {
    if (!this.context) {
      throw new Error("Cannot get context from a closed ai-coustics VAD processor");
    }
    return this.context;
  }

  isEnabled(): boolean {
    return this.processorEnabled;
  }

  setEnabled(enabled: boolean): void {
    if (this.closed || enabled === this.processorEnabled) return;
    if (enabled) this.reset();
    this.processorEnabled = enabled;
  }

  override onStreamInfoUpdated(info: FrameProcessorStreamInfo): void {
    if (
      this.streamInfo &&
      (this.streamInfo.roomName !== info.roomName ||
        this.streamInfo.participantIdentity !== info.participantIdentity ||
        this.streamInfo.publicationSid !== info.publicationSid)
    ) {
      this.reset();
    }
    this.streamInfo = info;
  }

  override onStreamInfoCleared(): void {
    if (this.streamInfo) this.reset();
    this.streamInfo = null;
  }

  process(frame: AudioFrame): AudioFrame {
    if (!this.processorEnabled || !this.nativeVad || !this.context) return frame;

    let metadata: FrameMetadata;
    try {
      metadata = { results: this.processFrame(frame) };
    } catch (error) {
      metadata = {
        results: [],
        error: { message: errorDetail(error), cause: error },
      };
    }
    frame.userdata[FRAME_USERDATA_VAD_ATTRIBUTE] = metadata;
    return frame;
  }

  reset(): void {
    this.context?.reset();
    this.format = null;
    this.inferenceBuffer = new Int16Array(0);
    this.predictionDelaySamples = 0;
    this.processingBacklogMs = 0;
    this.slowWarningActive = false;
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.processorEnabled = false;
    const nativeVad = this.nativeVad;
    this.nativeVad = null;
    this.context = null;
    this.inferenceBuffer = new Int16Array(0);
    nativeVad?.terminateSession();
  }

  private processFrame(frame: AudioFrame): readonly InferenceResult[] {
    const nativeVad = this.nativeVad;
    const context = this.context;
    if (!nativeVad || !context) return [];
    const mono = monoPcm16(frame);

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
    if (this.streamInfo) Object.assign(diagnostics, this.streamInfo);
    if (this.format) {
      Object.assign(diagnostics, {
        sampleRate: this.format[0],
        blockSize: this.format[1],
      });
    }
    return Object.assign(diagnostics, fields);
  }
}

/** LiveKit VAD whose shared `processor` performs the SDK inference. */
export class VAD extends LiveKitVAD {
  label = "ai-coustics.VAD";

  private readonly modelId: string;
  private readonly prefixPaddingDuration: number;
  private readonly maxBufferedSpeech: number;
  private readonly parameters: VADParameters = {
    speechHoldDuration: DEFAULT_SPEECH_HOLD_DURATION_SECONDS,
    minimumSpeechDuration: DEFAULT_MINIMUM_SPEECH_DURATION_SECONDS,
  };
  private readonly streams = new Set<WeakRef<AicVADStream>>();
  private readonly sharedProcessor: VADProcessor;

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
    this.prefixPaddingDuration = prefixPaddingDuration;
    this.maxBufferedSpeech = maxBufferedSpeech;
    this.sharedProcessor = new VADProcessor(
      options.model,
      resolveLicenseKey(options.licenseKey),
    );
    this.setParameters(this.parameters);
    if (options.vadParameters) this.setParameters(options.vadParameters);
  }

  get processor(): VADProcessor {
    return this.sharedProcessor;
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
      processor: this.sharedProcessor,
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
    this.sharedProcessor.close();
  }

  private applyParameter(parameter: number, name: string, value: number): boolean {
    try {
      this.sharedProcessor.sdkContext.setParameter(parameter, value);
    } catch (error) {
      writeLog(
        "warn",
        "vad",
        "parameter rejected; keeping the current value",
        this.sharedProcessor.diagnosticFields({
          parameter: name,
          parameterValue: value,
          errorType: error instanceof Error ? error.name : typeof error,
          errorMessage: errorDetail(error),
        }),
        error,
      );
      return false;
    }
    return true;
  }
}

interface StreamOptions {
  processor: VADProcessor;
  prefixPaddingDuration: number;
  maxBufferedSpeech: number;
}

class AicVADStream extends LiveKitVADStream {
  private readonly processor: VADProcessor;
  private readonly prefixPaddingDuration: number;
  private readonly maxBufferedSpeech: number;
  private outputFinished = false;
  private pumpError: unknown;
  private missingMetadataFrames = 0;

  constructor(vad: VAD, options: StreamOptions) {
    super(vad);
    this.processor = options.processor;
    this.prefixPaddingDuration = options.prefixPaddingDuration;
    this.maxBufferedSpeech = options.maxBufferedSpeech;
    void this.pump()
      .then(() => this.finishOutput())
      .catch(async (error: unknown) => {
        this.pumpError = error;
        writeLog(
          "error",
          "vad",
          "stream failed",
          this.processor.diagnosticFields({
            errorType: error instanceof Error ? error.name : typeof error,
            errorMessage: errorDetail(error),
          }),
          error,
        );
        await this.finishOutput();
      });
  }

  override endInput(): void {
    if (this.inputClosed) throw new Error("Input is closed");
    if (this.closed) throw new Error("Stream is closed");
    this.inputClosed = true;
    void this.inputWriter.close();
  }

  override close(): void {
    if (this.closed) return;
    this.closed = true;
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
          this.processor.diagnosticFields({
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
      if (done) break;
      if (typeof value === "symbol") {
        resetState();
        continue;
      }

      const metadata = value.userdata[FRAME_USERDATA_VAD_ATTRIBUTE] as
        | FrameMetadata
        | undefined;
      if (!metadata || !Array.isArray(metadata.results)) {
        this.missingMetadataFrames += 1;
        if (this.missingMetadataFrames === MISSING_METADATA_WARNING_FRAMES) {
          writeLog(
            "error",
            "vad",
            "no inference metadata found; pass vad.processor as RoomIO noiseCancellation",
            this.processor.diagnosticFields({
              missingMetadataFrames: this.missingMetadataFrames,
            }),
          );
        }
        continue;
      }
      this.missingMetadataFrames = 0;
      if (metadata.error) {
        throw new Error(metadata.error.message, { cause: metadata.error.cause });
      }

      for (const result of metadata.results) {
        if (inputSampleRate === 0) inputSampleRate = result.sampleRate;
        else if (result.sampleRate !== inputSampleRate) {
          writeLog(
            "error",
            "vad",
            "received frame with a different sample rate",
            this.processor.diagnosticFields({
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
