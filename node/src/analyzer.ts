import type { TypedEventEmitter as TypedEmitter } from "@livekit/typed-emitter";
import { metrics } from "@opentelemetry/api";
import { EventEmitter } from "node:events";
import {
  AudioFrame,
  FrameProcessor,
  type FrameProcessorStreamInfo,
} from "@livekit/rtc-node";

import { writeLog } from "./log.js";
import { pcm16ToFloat32 } from "./processor.js";
import {
  type AnalysisResult,
  type AnalyzerInstance,
  type CollectorInstance,
  type Model,
  analyzerPair,
  setSdkId,
} from "./sdk.js";

const DEFAULT_ANALYSIS_INTERVAL_SECONDS = 5;
const meter = metrics.getMeter("ai-coustics-livekit-extras");
const analysisCount = meter.createCounter("ai_coustics.analyzer.analysis", {
  description: "Number of ai-coustics buffered audio analyses",
});
const inferenceDuration = meter.createHistogram(
  "ai_coustics.analyzer.inference_duration",
  {
    unit: "s",
    description: "Duration of ai-coustics buffered audio analysis",
  },
);
const score = meter.createHistogram("ai_coustics.analyzer.score", {
  description: "Audio analysis score produced by ai-coustics",
});
const metricBaseAttributes = { model_provider: "ai-coustics" } as const;
const resultFields = [
  ["risk_score", "riskScore"],
  ["speaker_reverb", "speakerReverb"],
  ["speaker_loudness", "speakerLoudness"],
  ["interfering_speech", "interferingSpeech"],
  ["noise", "noise"],
  ["codec_degradation", "codecDegradation"],
  ["packet_loss", "packetLoss"],
] as const satisfies ReadonlyArray<readonly [string, keyof AnalysisResult]>;

export interface AnalyzerOptions {
  /** Loaded ai-coustics SDK analysis model. */
  model: Model;
  licenseKey?: string;
  /** Seconds between analyses. Defaults to 5. */
  analysisInterval?: number;
  /** Record aggregate OpenTelemetry metrics. Defaults to true. */
  enableMetrics?: boolean;
}

export interface AnalysisEvent {
  readonly result: Readonly<AnalysisResult>;
  /** Unix timestamp in milliseconds recorded after inference completed. */
  readonly timestamp: number;
  /** Elapsed inference time in milliseconds. */
  readonly inferenceDuration: number;
  readonly sequence: number;
  readonly modelId: string;
  readonly roomName?: string;
  readonly participantIdentity?: string;
  readonly publicationSid?: string;
}

export type AnalyzerCallbacks = {
  analysisResult: (event: AnalysisEvent) => void;
};

function resolveLicenseKey(value?: string): string {
  const key = value || process.env.AIC_SDK_LICENSE;
  if (!key) {
    throw new Error(
      "An ai-coustics SDK license is required. Pass licenseKey or set AIC_SDK_LICENSE.",
    );
  }
  return key;
}

function errorDetail(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Transparent LiveKit frame processor that collects audio for an Analyzer. */
export class Collector extends FrameProcessor<AudioFrame> {
  private nativeCollector: CollectorInstance | null;
  private readonly resetAnalyzer: () => void;
  private readonly closeAnalyzer: () => void;
  private streamFormat: [number, number, number] | null = null;
  private streamInfo: FrameProcessorStreamInfo | null = null;
  private hasBufferedAudio = false;
  private collectingEnabled = true;
  private closed = false;

  constructor(
    nativeCollector: CollectorInstance,
    resetAnalyzer: () => void,
    closeAnalyzer: () => void,
  ) {
    super();
    this.nativeCollector = nativeCollector;
    this.resetAnalyzer = resetAnalyzer;
    this.closeAnalyzer = closeAnalyzer;
  }

  isEnabled(): boolean {
    return this.collectingEnabled;
  }

  setEnabled(enabled: boolean): void {
    if (this.closed || enabled === this.collectingEnabled) return;
    if (enabled) this.reset();
    this.collectingEnabled = enabled;
  }

  get initialized(): boolean {
    return this.hasBufferedAudio && this.nativeCollector !== null;
  }

  get currentStreamInfo(): FrameProcessorStreamInfo | null {
    return this.streamInfo ? { ...this.streamInfo } : null;
  }

  override onStreamInfoUpdated(info: FrameProcessorStreamInfo): void {
    this.streamInfo = { ...info };
    this.reset();
  }

  override onStreamInfoCleared(): void {
    this.streamInfo = null;
    this.reset();
  }

  process(frame: AudioFrame): AudioFrame {
    const collector = this.nativeCollector;
    if (!this.collectingEnabled || !collector) return frame;

    try {
      const streamFormat: [number, number, number] = [
        frame.sampleRate,
        frame.channels,
        frame.samplesPerChannel,
      ];
      if (
        !this.streamFormat ||
        this.streamFormat[0] !== streamFormat[0] ||
        this.streamFormat[1] !== streamFormat[1] ||
        this.streamFormat[2] !== streamFormat[2]
      ) {
        collector.initialize(frame.sampleRate, frame.samplesPerChannel, false);
        this.streamFormat = streamFormat;
      }

      const expectedSamples = frame.samplesPerChannel * frame.channels;
      if (frame.data.length !== expectedSamples) {
        throw new Error(
          `AudioFrame contains ${frame.data.length} samples, expected ${expectedSamples}`,
        );
      }

      const samples = pcm16ToFloat32(frame.data);
      const mono = new Float32Array(frame.samplesPerChannel);
      if (frame.channels === 1) {
        mono.set(samples);
      } else {
        for (let sample = 0; sample < frame.samplesPerChannel; sample += 1) {
          let sum = 0;
          for (let channel = 0; channel < frame.channels; channel += 1) {
            sum += samples[sample * frame.channels + channel]!;
          }
          mono[sample] = sum / frame.channels;
        }
      }
      collector.buffer(mono);
      this.hasBufferedAudio = true;
    } catch (error) {
      writeLog(
        "error",
        "collector",
        "failed; passing audio through",
        this.streamInfo ?? {},
        error,
      );
    }

    return frame;
  }

  private reset(): void {
    if (this.closed) return;
    this.hasBufferedAudio = false;
    try {
      this.resetAnalyzer();
    } catch (error) {
      writeLog(
        "error",
        "analyzer",
        "reset failed",
        { errorMessage: errorDetail(error) },
        error,
      );
    }
  }

  detach(): void {
    this.closed = true;
    this.collectingEnabled = false;
    this.streamFormat = null;
    this.streamInfo = null;
    this.hasBufferedAudio = false;
    this.nativeCollector = null;
  }

  close(): void {
    if (this.closed) return;
    this.detach();
    this.closeAnalyzer();
  }
}

/** Owns an SDK analyzer pair and periodically reports analysis of collected room audio. */
export class Analyzer extends (EventEmitter as new () => TypedEmitter<AnalyzerCallbacks>) {
  readonly collector: Collector;

  private nativeAnalyzer: AnalyzerInstance | null;
  private readonly timer: ReturnType<typeof setInterval>;
  private readonly modelId: string;
  private readonly enableMetrics: boolean;
  private sequence = 0;
  private closed = false;

  constructor(options: AnalyzerOptions) {
    super();
    const analysisInterval =
      options.analysisInterval ?? DEFAULT_ANALYSIS_INTERVAL_SECONDS;
    if (!Number.isFinite(analysisInterval) || analysisInterval <= 0) {
      throw new Error("analysisInterval must be a finite value greater than zero");
    }

    setSdkId(9);
    let pair: ReturnType<typeof analyzerPair>;
    try {
      this.modelId = options.model.getId();
      pair = analyzerPair(options.model, resolveLicenseKey(options.licenseKey));
    } catch (error) {
      throw new Error(`Failed to create ai-coustics Analyzer: ${errorDetail(error)}`, {
        cause: error,
      });
    }

    this.nativeAnalyzer = pair.analyzer;
    this.enableMetrics = options.enableMetrics ?? true;
    this.collector = new Collector(
      pair.collector,
      () => pair.analyzer.reset(),
      () => this.close(),
    );
    this.timer = setInterval(() => this.analyze(), analysisInterval * 1000);
    this.timer.unref?.();
  }

  private analyze(): void {
    const analyzer = this.nativeAnalyzer;
    if (!analyzer || !this.collector.initialized) return;

    const started = performance.now();
    try {
      const nativeResult = analyzer.analyzeBuffered();
      const elapsed = performance.now() - started;
      const result = Object.freeze({ ...nativeResult });
      this.sequence += 1;
      const streamInfo = this.collector.currentStreamInfo;
      const event = Object.freeze({
        result,
        timestamp: Date.now(),
        inferenceDuration: elapsed,
        sequence: this.sequence,
        modelId: this.modelId,
        ...(streamInfo ?? {}),
      }) satisfies AnalysisEvent;

      this.recordMetrics(elapsed, "ok", result);
      try {
        this.emit("analysisResult", event);
      } catch (error) {
        writeLog(
          "error",
          "analyzer",
          "result event emission failed",
          { modelName: this.modelId, sequence: this.sequence, ...(streamInfo ?? {}) },
          error,
        );
      }
    } catch (error) {
      this.recordMetrics(performance.now() - started, "error");
      writeLog(
        "error",
        "analyzer",
        "buffered audio analysis failed",
        { modelName: this.modelId, ...(this.collector.currentStreamInfo ?? {}) },
        error,
      );
    }
  }

  private recordMetrics(
    inferenceDurationMs: number,
    status: "ok" | "error",
    result?: Readonly<AnalysisResult>,
  ): void {
    if (!this.enableMetrics) return;

    try {
      const attributes = { ...metricBaseAttributes, status };
      analysisCount.add(1, attributes);
      inferenceDuration.record(inferenceDurationMs / 1000, attributes);
      if (result) {
        for (const [scoreName, property] of resultFields) {
          score.record(result[property], {
            ...metricBaseAttributes,
            "score.name": scoreName,
          });
        }
      }
    } catch (error) {
      writeLog(
        "error",
        "analyzer",
        "metrics recording failed",
        { modelName: this.modelId, status },
        error,
      );
    }
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    clearInterval(this.timer);
    this.collector.detach();
    const analyzer = this.nativeAnalyzer;
    this.nativeAnalyzer = null;
    if (!analyzer) return;
    try {
      analyzer.terminateSession();
    } catch (error) {
      writeLog(
        "error",
        "analyzer",
        "session termination failed",
        { modelName: this.modelId, errorMessage: errorDetail(error) },
        error,
      );
    }
  }
}
