import { log } from "@livekit/agents";

export type LogComponent = "analyzer" | "collector" | "logger" | "vad";
export type LogLevel = "debug" | "info" | "warn" | "error";

const componentLabels: Record<LogComponent, string> = {
  analyzer: "Analyzer",
  collector: "Collector",
  logger: "Logger",
  vad: "VAD",
};

/** Write a structured diagnostic through LiveKit, with a console fallback during early startup. */
export function writeLog(
  level: LogLevel,
  component: LogComponent,
  event: string,
  fields: Record<string, unknown> = {},
  error?: unknown,
): void {
  const message = `${componentLabels[component]}: ${event}`;
  const diagnostics = {
    plugin: "ai-coustics",
    component,
    ...fields,
    ...(error === undefined ? {} : { error }),
  };

  try {
    log()[level](diagnostics, message);
    return;
  } catch (loggingError) {
    const loggerIsUninitialized =
      loggingError instanceof TypeError &&
      loggingError.message.includes("logger not initialized");
    if (!loggerIsUninitialized) {
      console.error(`${componentLabels.logger}: failed to write diagnostic through LiveKit`, {
        plugin: "ai-coustics",
        component: "logger",
        error: loggingError,
      });
    }
  }

  // Objects can be constructed before LiveKit initializes its process-wide logger. Preserve
  // operational diagnostics, but do not turn normally-hidden debug events into console noise.
  if (level === "error") console.error(message, diagnostics);
  else if (level === "warn") console.warn(message, diagnostics);
  else if (level === "info") console.info(message, diagnostics);
}
