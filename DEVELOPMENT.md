# Development

This repository contains separate Python and Node.js packages backed by the corresponding public
ai-coustics SDK. Develop and test each package from its own directory.

## Architecture

`Processor` accepts an already-loaded SDK `Model`, never a model ID or file path. Both packages
expose the SDK's `Model.from_file` / `Model.fromFile` and `Model.download` APIs so applications can
choose when and where models are downloaded and loaded.

Model loading and native Processor construction happen when the filter is constructed. A
synchronous SDK construction error is raised with its original error attached. Backend
authentication continues asynchronously during the SDK's grace period, so the plugin does not
process a throwaway frame to probe the license.

Processor format initialization is lazy because LiveKit supplies the complete stream geometry
with the first frame. Each LiveKit frame is processed in one fixed-size SDK call, avoiding the
additional latency of the SDK's variable-block-size mode. aic-sdk 3.1 for Python and 0.23 for Node
process mono audio only, so multichannel LiveKit frames are downmixed before processing and the
enhanced signal is duplicated across the original channel count. This preserves the LiveKit frame
geometry and metadata.

Processing errors are logged and the original LiveKit frame is returned. This fail-open behavior
keeps room audio flowing if the SDK rejects a frame or encounters a runtime error. Processor logs
use LiveKit's logging path and include the model, stream identity when available, audio format,
failure stage, and timing context. Repeated failures and slower-than-realtime warnings are
rate-limited; recovery and close events summarize the affected frames, processing time, real-time
factor, and maximum accumulated processing backlog. The Node implementation falls back to the
console for operational messages when LiveKit has not initialized its logger.

Each `VAD` owns one synchronous native SDK VAD and exposes it through a pass-through
`vad.processor`. That processor must be installed at the start of RoomIO's `noise_cancellation`
path. It downmixes and reblocks original input audio, runs inference once, and attaches immutable
per-block results to the frame's userdata. Every LiveKit `VADStream` consumes the same results, so
creating additional streams does not create native VADs or repeat inference. A missing-metadata
diagnostic identifies configurations that forgot to install `vad.processor`.

Streams independently derive LiveKit inference and speech transition events from the shared
snapshots. Event durations and speech lookback account for the SDK's prediction delay, while raw
mono PCM stored in each snapshot keeps candidate audio aligned even when a later processor in the
chain replaces the frame's audio. Closing a stream does not terminate the shared SDK session;
RoomIO closes `vad.processor` directly or through `FrameProcessorChain`.

The wrapper overrides the model-specific SDK duration defaults with LiveKit-compatible values:
50 ms minimum speech and a 250 ms speech hold. The latter satisfies the minimum silence required
by LiveKit's streaming turn detector. Because the SDK hold uses a rolling-majority window, the
wrapper also keeps an active LiveKit speech segment open until that much continuous raw silence
has accumulated. Explicit `VADParameters` values still take precedence.

Each `Analyzer` owns one SDK collector/analyzer pair. Its public `collector` is a transparent
`FrameProcessor` installed in RoomIO's `noise_cancellation` slot: it lazily initializes from the
first frame, downmixes PCM16 input to mono float32, buffers it, and returns the original frame
unchanged. Stream boundaries reset the analyzer. Closing either the analyzer or its collector
stops scheduling and terminates the SDK telemetry session.

`FrameProcessorChain` forwards stream-info lifecycle hooks and applies any number of enabled
processors in constructor order. It lets a `Processor`, VAD processor, and Collector share
RoomIO's single `noise_cancellation` slot. Placing the Collector before the enhancement Processor
is recommended: analyzing original input audio helps explain how its quality affects the rest of
the pipeline.

Python schedules inference with an asyncio task and runs each blocking `analyze_buffered()` call
through `asyncio.to_thread()`. Shutdown waits for an active inference before terminating the SDK
session. Node uses a timer around the SDK's synchronous `analyzeBuffered()` API. Both runtimes emit
a plugin-level result event after every successful scheduled call without logging the result by
default. They also record aggregate score, inference-duration, and success/error count instruments
through the process-wide OpenTelemetry metrics API; operational errors remain logged and fail-open
for the room audio path.

## Logging convention

Plugin diagnostics follow LiveKit's structured logging conventions in both runtimes. Python uses
the `livekit.plugins.ai_coustics` logger and passes context through `extra`; Node writes an object
before the message through LiveKit's Pino logger. Do not embed changing values or serialized
objects in the message when they can be represented as fields.

Human-readable messages use `Component: event`, for example `Processor: initialized` or
`VAD: inference falling behind realtime`. Every plugin record also includes the stable fields
`plugin=ai-coustics` and `component=processor|vad|analyzer|collector` (`logger` is reserved for a
failure in the logging bridge itself). Python field names use
snake_case and durations are in seconds; Node field names use camelCase and durations are suffixed
with `Ms`. Exceptions should retain the original exception/traceback in the logging API and add
stable error fields where callers need to filter or aggregate failures.

Provider identity is deliberately not repeated as a bracketed message prefix. LiveKit's Python
formatter already displays the hierarchical logger name, while the structured `plugin` field
keeps Node, JSON, and OpenTelemetry records queryable. Node falls back to the same formatted
message and fields on the console when objects are created before LiveKit initializes its global
logger; debug records remain silent in that case.

## Future work

### LiveKit `download-files` integration

The plugins do not currently participate in LiveKit Agents' `download-files` commands. Model
provisioning must remain an explicit application or container-build step for now.

Python already discovers and imports `livekit.plugins.ai_coustics`, and the package registers a
LiveKit `Plugin`, but its inherited `download_files()` implementation is intentionally a no-op.
The standalone command does not load the user's agent configuration or pass arguments to the
plugin, so it cannot determine which arbitrary enhancement and VAD model IDs the application
intends to use. Downloading a fixed model enum would conflict with this plugin's model-ID and
file-path API.

Node has the same model-selection problem plus a discovery limitation. The standalone LiveKit CLI
only scans `node_modules/@livekit/agents-plugin-*`, so it does not import
`@ai-coustics/livekit-extras`; registering a `Plugin.downloadFiles()` hook in our package would not
make the modern command discover it.

A future implementation needs both an explicit, build-time source of model IDs and a stable way
to resolve the downloaded versioned files without contacting the model manifest again at runtime.
Node additionally needs an upstream LiveKit discovery mechanism for third-party package scopes,
such as package metadata or an explicit package list. Until those pieces exist, use
`Model.download` during provisioning and construct runtime models from the returned files with
`Model.from_file` / `Model.fromFile`.

### First-class streaming Analyzer integration

The aic-sdk streaming analysis API is split into a `Collector` and an `Analyzer`. The collector
accepts mono float32 audio synchronously and is safe to feed from the audio path, while
`analyze_buffered()` / `analyzeBuffered()` runs an expensive model inference and must execute away
from that path. The result contains risk, speaker reverb, speaker loudness, interfering speech,
noise, codec-degradation, and packet-loss scores. `FileAnalyzer` is intended for complete in-memory
signals and is not appropriate for a live agent stream.

An Analyzer is a side-channel consumer rather than an audio transform. The current workaround is
a pass-through `FrameProcessor` in RoomIO's single `noise_cancellation` slot.
`FrameProcessorChain` allows it to run sequentially with the enhancement `Processor`, but this
still couples observation to the transform path and makes the observation point depend on
Collector placement. The examples put the Collector before the Processor to capture raw input.
Making `Analyzer` itself delegate the FrameProcessor interface would permit the minor
`noise_cancellation=analyzer` shortcut, but would duplicate the collector's public role and would
not solve composition with `Processor`; keep the explicit collector unless an upstream interface
makes the ownership model clearer.

The preferred upstream solution is a generic audio observer or tap interface in LiveKit Agents.
RoomIO or AgentSession should fan frames out to registered observers without allowing observer
backpressure or failures to affect the main audio pipeline. Its lifecycle should cover stream
metadata, audio-format initialization, hard-boundary reset or flush, track replacement and detach,
and asynchronous close. Placement should be explicit:

- A `raw` tap before noise cancellation and automatic gain control measures the caller's original
  environment. This is the recommended placement for understanding how input quality affects the
  rest of the pipeline.
- A `processed` tap observes exactly what downstream STT, VAD, turn detection, or speech-to-speech
  models receive and remains useful when that is the desired observation point.

Supporting a truly raw tap may also require an RTC `AudioStream` hook or a branching processing
graph because the current RTC frame processor runs before Agents receives the frame.

Once that upstream interface exists, the plugin should create one SDK collector/analyzer pair per
LiveKit observer stream. The current plugin-level `analysis_result` / `analysisResult` events can
then be attached to that observer without changing their payload shape. Events include the seven
scores, model and stream metadata, result time, inference duration, and a sequence number; the
upstream observer could additionally provide an authoritative accumulated audio position.
Collection requires only PCM16-to-float32 conversion and mono downmixing; the SDK handles sample
rate adaptation after the collector is initialized with the input rate. Analysis should be
scheduled from accumulated audio duration, allow only one inference at a time, and replace stale
pending work instead of building an unbounded queue. Track and format discontinuities must reset
the SDK state, and close must terminate the analyzer telemetry session.

As a further upstream option, `AgentSession` could forward observer results through a dedicated
`audio_analysis` event, with a new framework-owned event type and reporting hook. That would make
analysis discoverable alongside other session observability without forcing it into LiveKit's
deprecated `metrics_collected` event or its closed `AgentMetrics` union. LiveKit could also define
stable analyzer metric names and Cloud dashboard treatment; until then the plugin-owned
OpenTelemetry instruments are intentionally aggregate and backend-agnostic.

Adding an `AudioAnalysisMetrics` model to LiveKit's metrics package would be useful as an upstream
data contract, but is not sufficient by itself. RoomIO does not subscribe to events from a
`FrameProcessor`, AgentActivity only forwards metrics from its known model interfaces, the
session-level `metrics_collected` event is deprecated, and session reports discard those events.
LiveKit's OpenTelemetry exporter also maps each supported metric type explicitly. A complete
integration would therefore need the observer transport, Python and Node metric types and union
exports, the dedicated session event or another non-deprecated delivery path, explicit
OpenTelemetry instruments, and a decision about Cloud persistence and dashboard rendering.

If LiveKit chooses such a type, its fields should match the public aic-sdk result rather than
inventing translations that the plugin cannot populate:

```python
class AudioAnalysisMetrics(_BaseMetrics):
    type: Literal["audio_analysis_metrics"] = "audio_analysis_metrics"
    label: str
    timestamp: float
    inference_duration: float
    risk_score: float
    speaker_reverb: float
    speaker_loudness: float
    interfering_speech: float
    noise: float
    codec_degradation: float
    packet_loss: float
    window_duration: float | None = None
    metadata: Metadata | None = None
```

The current SDK has `codec_degradation` and `noise`, but no `media_speech` result. A Tyto analysis
window is continuous room audio and is not naturally associated with an AgentSession `speech_id`.
The streaming SDK determines its window length from the model but does not expose that duration,
so `window_duration` must remain optional until aic-sdk provides it. If future analysis models use
different context windows, that API will also avoid hard-coding the current five-second window.

Python runs `analyze_buffered()` through `asyncio.to_thread()` because the binding releases the
GIL during inference. Node aic-sdk 0.23 exposes only synchronous `analyzeBuffered()` and
`terminateSession()`, so calling them from a timer would still block the agent's JavaScript event
loop. A production Node integration first needs native asynchronous APIs such as
`analyzeBufferedAsync()` and `terminateSessionAsync()` that execute on a worker pool.

### First-class Processor metrics

The plugins currently expose Processor health through structured LiveKit logs. This covers events
that are immediately actionable—initialization and format changes, failures and recovery,
slower-than-realtime processing, and a lifetime summary—but it does not produce LiveKit metrics.

LiveKit's `FrameProcessor` interface has lifecycle and processing hooks but no metrics event, and
the Agents metrics model has no `FrameProcessorMetrics` equivalent. Adding private plugin metric
objects would therefore bypass the normal `AgentSession` collection, `log_metrics()`, usage
collection, and OpenTelemetry export paths.

A future upstream implementation should let RoomIO or `AgentSession` receive and emit frame
processor metrics. Useful fields include processed and failed frame counts, input-audio duration,
total and maximum processing duration, average and maximum real-time factor, maximum accumulated
processing backlog, initialization count, and the model's fixed audio delay. The fixed audio
delay describes signal alignment and must remain distinct from processing backlog, which is
runtime scheduling and compute debt. Once LiveKit provides that surface, these plugins can map the
same counters used by their structured logs into the standard metrics pipeline.

### Richer upstream VAD metrics

The plugin reports each native call's elapsed time through `VADEvent.inference_duration`, which
LiveKit aggregates into `VADMetrics.inference_duration_total` and `inference_count`. The plugin also
logs structured warnings when cumulative processing time falls behind the incoming audio timeline.

LiveKit's metrics model cannot currently expose the worst inference or processing backlog observed
inside an aggregation period. A future upstream improvement could add processed audio duration (or
real-time factor), maximum inference duration, and maximum processing backlog to `VADMetrics`.
LiveKit's `log_metrics()` and OpenTelemetry exporter could then render and export those fields. The
processing backlog is runtime scheduling/compute debt and must remain distinct from a model's fixed
VAD prediction delay.

## Setup

Python development uses `uv`:

```bash
cd python
uv sync --dev
```

Node.js development uses npm:

```bash
cd node
npm ci
```

Set `AIC_SDK_LICENSE` before running tests that construct a real SDK Processor or VAD:

```bash
export AIC_SDK_LICENSE=...
```

The integration and end-to-end tests download
`quail-vf-2.2-s-16khz` and `vad-2.1-xxs-16khz` by default. Override their IDs or cache directory
with `AIC_INTEGRATION_MODEL_ID`, `AIC_INTEGRATION_VAD_MODEL_ID`, and
`AIC_INTEGRATION_MODEL_DIR`.

## Tests and checks

The test suite has three layers:

1. Unit tests mock the native SDK boundary and require no license or network access.
2. Integration tests use downloaded models and the real SDK Processor and VAD directly.
3. End-to-end tests send microphone audio through a real LiveKit room and an `AgentSession`
   configured with the real Processor and VAD integrations.

Run the Python suite and checks with:

```bash
cd python
uv run pytest tests/test_analyzer.py -q
uv run pytest tests/test_processor.py -q
uv run pytest tests/test_vad.py -q
uv run pytest tests/test_integration.py -q
uv run pytest tests/test_e2e_room.py -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Run the Node.js suite and checks with:

```bash
cd node
npm test
npm run test:integration
npm run test:e2e
npm run check
npm run build
```

The integration and end-to-end commands require `AIC_SDK_LICENSE`. The end-to-end commands also
require a LiveKit server.

## Local end-to-end environment

The end-to-end tests connect two RTC participants to a unique room:

```text
synthetic microphone publisher
  -> LiveKit server
  -> agent RoomIO
  -> rtc.AudioStream
  -> ai-coustics Processor and VAD
  -> AgentSession
```

They verify that RoomIO invokes the real SDK-backed Processor, the `AgentSession` consumes the
SDK-backed VAD, processing remains successful after the SDK authentication grace period, frame
geometry is preserved, and RoomIO closes the Processor during session teardown.

Start LiveKit in a separate terminal. A native server is the simplest option on macOS; Linux can
use either the native server or the container:

```bash
livekit-server --dev

# Linux container alternative. Keep the version aligned with CI.
docker run --rm --network host livekit/livekit-server:v1.13.1 --dev
```

LiveKit dev mode uses the following values, which are also the test defaults:

```text
LIVEKIT_URL=ws://127.0.0.1:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
```

Set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` to target a different test server.
Do not run the end-to-end tests against a production LiveKit project.

## Continuous integration

Unit tests run across the supported Python and Node.js version matrices. The protected integration
jobs run one representative version of each runtime, require the repository's `AIC_SDK_LICENSE`
secret, cache downloaded models, and execute both the native integration and room end-to-end
tests.

CI starts the pinned upstream LiveKit server image with host networking and dev credentials. The
server is readiness-checked before the tests, its logs are printed even after a failure, and the
container is always removed. Integration jobs do not run for pull requests from forks so the SDK
license is not exposed.

## Releases

Python and Node.js packages are released together and must always have the same version. To prepare
a release:

1. Update `project.version` in `python/pyproject.toml` and run `uv lock` from `python/`.
2. Run `npm version <version> --no-git-tag-version` from `node/` to update `package.json` and
   `package-lock.json`.
3. Commit the version changes, merge them to `main`, and wait for CI to pass.
4. Create and push an unprefixed semantic version tag:

   ```bash
   git tag 0.1.0
   git push origin 0.1.0
   ```

The release workflow verifies that the tag points to a commit on `main` and matches both package
versions. It then builds both distributions, publishes `ai-coustics-livekit-extras` to PyPI and
`@ai-coustics/livekit-extras` to npm, and creates a GitHub release containing all distribution
artifacts. The GitHub release is created only after both registry publications succeed.

Repository and registry configuration required by the publish jobs:

- A GitHub `publish` environment, restricted to `*.*.*` tags, optionally with required reviewers.
- A PyPI trusted publisher for `ai-coustics-livekit-extras`, restricted to this repository,
  `release.yml`, and the `publish` environment. The publish job authenticates over OIDC and
  stores no PyPI token.
- An npm trusted publisher for `@ai-coustics/livekit-extras`, restricted to this repository,
  `.github/workflows/release.yml`, and the `publish` environment. npm only allows a trusted
  publisher on a package that already exists, so the first release authenticates with an
  `NPM_TOKEN` secret in the `publish` environment. Configure the trusted publisher and delete
  that secret once the package is on the registry.

## Dependency scanning

Both lockfiles are committed and are the input our dependency scanning reads: `python/uv.lock`
and `node/package-lock.json`. Keep them in step with their manifests. `npm ci` fails on a stale
`package-lock.json`, and CI runs `uv lock --check` for the same reason on the Python side. Do not
add either lockfile to `.gitignore`.
