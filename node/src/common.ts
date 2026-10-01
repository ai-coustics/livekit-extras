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

export function pcm16ToFloat32(data: Int16Array): Float32Array {
  return Float32Array.from(data, (sample) => sample / 32768);
}
