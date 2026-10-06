# Development

This repository contains separate Python and Node.js packages backed by the corresponding public
ai-coustics SDK. Develop and test each package from its own directory.

## Architecture

`VAD` and `Analyzer` accept an already-loaded SDK `Model`, never a model ID or file path. Both
packages expose the SDK's `Model.from_file` / `Model.fromFile` and `Model.download` APIs so
applications can choose when and where models are downloaded and loaded.

Native SDK objects are created when the plugin object is constructed. A synchronous SDK
construction error is raised with its original error attached. Backend authentication continues
asynchronously during the SDK's grace period, so the plugin does not process a throwaway frame to
probe the license.

Audio format initialization is lazy because LiveKit supplies the complete stream geometry with the
first frame. aic-sdk 3.3 for Python and 0.25 for Node process mono audio only, so multichannel
LiveKit frames are downmixed before they reach the SDK.

Both frame processors return the original LiveKit frame, so SDK errors never stop room audio. The
VAD processor attaches an inference error to the frame's metadata and the VAD stream reports it; the
Collector logs the error and keeps passing audio through. Logs use LiveKit's logging path and
include the model, stream identity when available, and audio format. The Node implementation falls
back to the console for operational messages when LiveKit has not initialized its logger.

Each `VAD` owns one synchronous native SDK VAD and exposes it through a pass-through
`vad.processor`. That processor must be installed in RoomIO's `noise_cancellation` slot. It
downmixes and reblocks original input audio, runs inference once, and attaches immutable
per-block results to the frame's userdata. Every LiveKit `VADStream` consumes the same results, so
creating additional streams does not create native VADs or repeat inference. A missing-metadata
diagnostic identifies configurations that forgot to install `vad.processor`.

Streams independently derive LiveKit inference and speech transition events from the shared
snapshots. Event durations and speech lookback account for the SDK's prediction delay, while raw
mono PCM stored in each snapshot keeps candidate audio aligned even when a later frame processor
replaces the frame's audio. Closing a stream does not terminate the shared SDK session; RoomIO
closes `vad.processor` directly or through `FrameProcessorChain`.

The wrapper overrides the model-specific SDK duration defaults with LiveKit-compatible values:
50 ms minimum speech and a 250 ms speech hold. The latter satisfies the minimum silence required
by LiveKit's streaming turn detector. Because the SDK hold uses a rolling-majority window, the
wrapper also keeps an active LiveKit speech segment open until that much continuous raw silence
has accumulated. Explicit `VADParameters` values still take precedence.

Each `Analyzer` owns one SDK analyzer: a collector/analyzer pair in Python, a single `Analyzer` in
Node. Its public `collector` is a transparent `FrameProcessor` installed in RoomIO's
`noise_cancellation` slot: it lazily initializes from the first frame, downmixes PCM16 input to
mono float32, buffers it, and returns the original frame unchanged. Stream boundaries reset the
analyzer. Closing either the analyzer or its collector stops scheduling and terminates the SDK
telemetry session.

RoomIO accepts one `noise_cancellation` frame processor. `FrameProcessorChain` lets
`vad.processor`, `analyzer.collector`, and other frame processors share it. It applies its enabled
processors in constructor order and forwards the stream-info lifecycle hooks to all of them. It
does not forward the credential hooks. Closing the chain is idempotent and closes every child, then
raises the first child error. Place `vad.processor` and `analyzer.collector` before any processor
that changes the audio: analyzing original input audio helps explain how its quality affects the
rest of the pipeline.

Python schedules inference with an asyncio task and runs each blocking `analyze_buffered()` call
through `asyncio.to_thread()`. Shutdown waits for an active inference before terminating the SDK
session. Node uses a timer around the SDK's synchronous `analyze()` API. Both runtimes emit
a plugin-level result event after every successful scheduled call without logging the result by
default. They also record aggregate score, inference-duration, and success/error count instruments
through the process-wide OpenTelemetry metrics API; operational errors remain logged and fail-open
for the room audio path.

## Logging convention

Plugin diagnostics follow LiveKit's structured logging conventions in both runtimes. Python uses
the `ai_coustics.livekit` logger and passes context through `extra`; Node writes an object before
the message through LiveKit's Pino logger. Do not embed changing values or serialized objects in
the message when they can be represented as fields.

Human-readable messages use `Component: event`, for example `VAD: inference falling behind
realtime` or `Collector: failed; passing audio through`. Every plugin record also includes the
stable fields `plugin=ai-coustics` and `component=vad|analyzer|collector` (`logger` is reserved for
a failure in the logging bridge itself). Python field names use
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

The standalone command does not load the user's agent configuration or pass arguments to the
plugin, so it cannot determine which arbitrary VAD and analysis model IDs the application intends
to use. Downloading a fixed model enum would conflict with this plugin's model-ID and file-path
API.

Both runtimes also have a discovery limitation. The Python package registers a LiveKit `Plugin`
whose inherited `download_files()` implementation is a no-op, but it lives under
`ai_coustics.livekit` rather than the `livekit.plugins` namespace, so the standalone command does
not import it. The standalone Node CLI only scans `node_modules/@livekit/agents-plugin-*`, so it
does not import `@ai-coustics/livekit-extras`; registering a `Plugin.downloadFiles()` hook in our
package would not make the modern command discover it.

A future implementation needs both an explicit, build-time source of model IDs and a stable way
to resolve the downloaded versioned files without contacting the model manifest again at runtime.
Both runtimes additionally need an upstream LiveKit discovery mechanism for third-party packages,
such as package metadata or an explicit package list. Until those pieces exist, use
`Model.download` during provisioning and construct runtime models from the returned files with
`Model.from_file` / `Model.fromFile`.

### First-class streaming Analyzer integration

The Python aic-sdk streaming analysis API is split into a `Collector` and an `Analyzer`; Node 0.25
combines both roles in one `Analyzer`. Collection accepts mono float32 audio synchronously and is
safe to feed from the audio path, while `analyze_buffered()` / `analyze()` runs an expensive model
inference and must execute away from that path. The result contains risk, speaker reverb, speaker
loudness, interfering speech, noise, codec-degradation, and packet-loss scores. Python's
`FileAnalyzer` is intended for complete in-memory signals and is not appropriate for a live agent
stream.

An Analyzer is a side-channel consumer rather than an audio transform. The current workaround is
a pass-through `FrameProcessor` in RoomIO's single `noise_cancellation` slot.
`FrameProcessorChain` allows it to run sequentially with other processors, but this still couples
observation to the transform path and makes the observation point depend on Collector placement.
Making `Analyzer` itself delegate the FrameProcessor interface would permit the minor
`noise_cancellation=analyzer` shortcut, but would duplicate the collector's public role; keep the
explicit collector unless an upstream interface makes the ownership model clearer.

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
GIL during inference. Node calls the synchronous `analyze()` from a timer, which blocks the agent's
JavaScript event loop during inference. Node aic-sdk 0.25 also offers `analyzeAsync()`, which runs
on the libuv thread pool; moving to it requires a one-in-flight guard and terminating the session
only after an in-flight analysis settles.

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

Set `AIC_SDK_LICENSE` before running tests that construct a real SDK VAD or Analyzer:

```bash
export AIC_SDK_LICENSE=...
```

The integration and end-to-end tests download `vad-2.1-xxs-16khz` and `tyto-1.1-l-16khz` by
default. Override their IDs or cache directory with `AIC_INTEGRATION_VAD_MODEL_ID`,
`AIC_INTEGRATION_ANALYSIS_MODEL_ID`, and `AIC_INTEGRATION_MODEL_DIR`.

## Tests and checks

The test suite has three layers:

1. Unit tests mock the native SDK boundary and require no license or network access. Both
   runtimes replay the VAD event scenarios in `testdata/vad_scenarios.json`, so a change to VAD
   event behavior updates that one file.
2. Integration tests use downloaded models and the real SDK VAD and Analyzer directly.
3. End-to-end tests send microphone audio through a real LiveKit room and an `AgentSession`
   configured with the real VAD integration.

Run the Python suite and checks with:

```bash
cd python
uv run pytest tests/test_analyzer.py -q
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
  -> ai-coustics VAD processor
  -> AgentSession
```

They verify that the `AgentSession` consumes the SDK-backed VAD, VAD inference keeps succeeding
after the SDK authentication grace period, and RoomIO closes `vad.processor` during session
teardown.

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
