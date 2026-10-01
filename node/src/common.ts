import type { AudioFrame } from "@livekit/rtc-node";

export function resolveLicenseKey(value?: string): string {
  const key = value || process.env.AIC_SDK_LICENSE;
  if (!key) {
    throw new Error(
      "An ai-coustics SDK license is required. Pass licenseKey or set AIC_SDK_LICENSE.",
    );
  }
  return key;
}

export function errorDetail(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function monoPcm16(frame: AudioFrame): Int16Array {
  const expectedSamples = frame.samplesPerChannel * frame.channels;
  if (frame.data.length !== expectedSamples) {
    throw new Error(
      `AudioFrame contains ${frame.data.length} samples, expected ${expectedSamples}`,
    );
  }
  if (frame.channels === 1) return frame.data;
  const mono = new Int16Array(frame.samplesPerChannel);
  for (let sample = 0; sample < frame.samplesPerChannel; sample += 1) {
    let sum = 0;
    for (let channel = 0; channel < frame.channels; channel += 1) {
      sum += frame.data[sample * frame.channels + channel]!;
    }
    mono[sample] = Math.round(sum / frame.channels);
  }
  return mono;
}

export function pcm16ToFloat32(data: Int16Array): Float32Array {
  return Float32Array.from(data, (sample) => sample / 32768);
}
