# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository shape

Two independently built packages that are functional mirrors of each other:

- `python/` — `ai-coustics-livekit-extras`, importable as `livekit.plugins.ai_coustics` (namespace
  package under `python/src/livekit/plugins/ai_coustics/`), built on `aic-sdk` 3.1.
- `node/` — `@ai-coustics/livekit-extras` (`node/src/`), built on `@ai-coustics/aic-sdk` 0.23.

Run all commands from inside `python/` or `node/`; there is no root-level build. The two packages
are released in lockstep and must always carry the same version.

`DEVELOPMENT.md` is the authoritative long-form document for architecture rationale, the logging
convention, the local end-to-end environment, release steps, and the planned upstream LiveKit
integrations (`download-files`, an audio observer/tap interface, frame-processor and VAD metrics).
Read it before changing public behavior; update it when that rationale changes.

## Commands

Python (`uv`):

```bash
cd python
uv sync --dev
uv run pytest tests/test_processor.py -q          # single unit test file
uv run pytest tests/test_vad.py::test_name -q     # single test
uv run pytest tests/test_integration.py -q        # needs AIC_SDK_LICENSE
uv run pytest tests/test_e2e_room.py -q           # needs AIC_SDK_LICENSE + LiveKit server
uv run ruff check . && uv run ruff format --check .
uv run mypy                                       # strict; CI skips it only on 3.10
```

Node (`npm`):

```bash
cd node
npm ci
npm test                                          # unit tests only (explicit file list in package.json)
npx vitest run test/vad.test.ts -t "name"         # single test
npm run test:integration                          # needs AIC_SDK_LICENSE
npm run test:e2e                                  # needs AIC_SDK_LICENSE + LiveKit server
npm run check                                     # tsc --noEmit
npm run build                                     # tsup -> dist
```

`npm test` names its unit test files explicitly — a new unit test file must be added to that script
or CI will not run it.

Test environment: `AIC_SDK_LICENSE` is required for integration and e2e (those suites skip without
it rather than fail). Models default to `quail-vf-2.2-s-16khz`, `vad-2.1-xxs-16khz`, and
`tyto-1.1-l-16khz`, overridable via `AIC_INTEGRATION_MODEL_ID`, `AIC_INTEGRATION_VAD_MODEL_ID`,
`AIC_INTEGRATION_ANALYSIS_MODEL_ID`, `AIC_INTEGRATION_MODEL_DIR`. E2E needs a LiveKit server —
`livekit-server --dev` (or the pinned `livekit/livekit-server:v1.13.1` container) with the dev
defaults `ws://127.0.0.1:7880` / `devkey` / `secret`. Never point e2e at a production project.

## Architecture

Four public objects, each mirrored across both runtimes:

- **`Processor`** (`processor.py` / `processor.ts`) — a LiveKit `FrameProcessor` doing SDK speech
  enhancement. Takes an already-loaded SDK `Model`, never a model ID or path; applications provision
  models themselves via `Model.download` + `Model.from_file`/`fromFile`. Native construction happens
  in the constructor (synchronous errors surface immediately, with backend auth continuing in the
  SDK grace period); **audio format init is lazy** because LiveKit only supplies stream geometry with
  the first frame. One LiveKit frame maps to one fixed-size SDK call. Multichannel input is
  downmixed and the enhanced mono signal duplicated back across channels to preserve frame geometry.
- **`VAD` / `VADProcessor`** (`vad.py` / `vad.ts`) — inference runs *once*, in the pass-through
  `vad.processor` frame processor, which attaches immutable per-block snapshots (including raw mono
  PCM) to `frame.userdata`. Every `VADStream` derives LiveKit events from those shared snapshots, so
  extra streams cost no native VADs and no extra inference. Closing a stream must not tear down the
  shared SDK session — RoomIO closes `vad.processor`. The wrapper overrides SDK duration defaults
  with LiveKit-compatible values (50 ms minimum speech, 250 ms speech hold, the latter satisfying
  LiveKit's streaming turn detector); explicit `VADParameters` still win.
- **`Analyzer` / `Collector`** (`analyzer.py` / `analyzer.ts`) — the public `collector` is a
  transparent `FrameProcessor` that buffers mono float32; the analyzer runs periodic
  `analyze_buffered()`/`analyzeBuffered()` inference off the audio path and emits
  `analysis_result` / `analysisResult` events plus aggregate OpenTelemetry instruments. Python uses
  an asyncio task + `asyncio.to_thread()`; Node uses a timer around the synchronous SDK call.
- **`FrameProcessorChain`** (`frame_processor_chain.py` / `.ts`) — lets these share RoomIO's single
  `noise_cancellation` slot. Order matters: `vad.processor` first, then `analyzer.collector`, then
  the enhancement `Processor`, so VAD and analysis see original input audio.

`ProcessorContext` wraps the SDK context purely to add structured logging around parameter and
bearer-token changes. Node's `sdk.ts` hand-declares structural types because aic-sdk 0.23 ships no
TypeScript declarations.

**Fail-open is a hard invariant.** Any processing error is logged and the *original* frame is
returned; room audio must keep flowing whatever the SDK does. Repeated failures and
slower-than-realtime warnings are rate-limited, with recovery/close records summarizing affected
frames, RTF, and maximum processing backlog. Keep the model's fixed audio delay conceptually
distinct from processing backlog (scheduling/compute debt) in metrics and logs.

## Conventions

**Parity.** A change to one runtime almost always needs the mirrored change in the other, including
tests. The file layouts intentionally correspond one-to-one.

**Logging** (full rules in `DEVELOPMENT.md`). Messages are `Component: event`
(`Processor: initialized`, `VAD: inference falling behind realtime`). Every record carries
`plugin=ai-coustics` and `component=processor|vad|analyzer|collector`. Put changing values in
structured fields, never in the message string. Python: `livekit.plugins.ai_coustics` logger,
`extra=`, snake_case fields, durations in seconds. Node: LiveKit's Pino logger with the object
before the message, camelCase fields, durations suffixed `Ms`, with a console fallback for objects
created before LiveKit initializes its global logger. Do not add a bracketed provider prefix.

**Releases.** Bump `project.version` in `python/pyproject.toml` (then `uv lock`) and run
`npm version <version> --no-git-tag-version` in `node/`; merge to `main`, then push an unprefixed
semver tag (`git tag 0.1.0`). The release workflow refuses tags that are not on `main` or that
disagree with either package version.
