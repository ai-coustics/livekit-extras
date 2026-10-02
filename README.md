# ai-coustics extras for LiveKit Agents

This repository contains the ai-coustics-maintained Python and Node.js integrations for
[LiveKit Agents](https://docs.livekit.io/agents/). They are thin wrappers around the public
ai-coustics SDKs and let LiveKit agents use:

- `VAD` for voice activity detection with a dedicated SDK VAD model, through `vad.processor` and
  LiveKit's streaming VAD interface.
- `Analyzer` for periodic audio-quality analysis.
- `FrameProcessorChain` for applying any number of frame processors in sequence.

Applications choose and provision the SDK models themselves. This keeps model selection, storage,
and deployment under application control while the plugins handle LiveKit audio formats, stream
lifecycle, observability, and error recovery.

> [!IMPORTANT]
> The official LiveKit plugins are the recommended integration path for most applications. They
> support LiveKit Cloud, integrate more deeply with the LiveKit ecosystem and tooling, and offer
> stronger stability guarantees. Avoiding breaking changes is an explicit goal of the official
> plugins.
>
> The ai-coustics-maintained plugins in this repository follow a faster release cadence and are
> designed for teams that want early access to the latest ai-coustics models and product features.
> Some of those capabilities may be experimental and subject to change, and use of these plugins is
> billed separately through ai-coustics. Choose them when early adoption of new ai-coustics
> capabilities is important for your application.

## Packages

| Runtime | Package | Documentation |
| --- | --- | --- |
| Python | `ai-coustics-livekit-extras` | [Installation and usage](python/README.md) |
| Node.js | `@ai-coustics/livekit-extras` | [Installation and usage](node/README.md) |

The runtime guides cover model provisioning, AgentSession integration, VAD parameters, audio
analysis, authentication, and operational constraints.

> These packages are separate from LiveKit's official
> [`livekit-plugins-ai-coustics`](https://pypi.org/project/livekit-plugins-ai-coustics/) and
> [`@livekit/plugins-ai-coustics`](https://www.npmjs.com/package/@livekit/plugins-ai-coustics)
> packages and can be installed alongside them. Speech enhancement is provided by the official
> plugins.

## Repository contents

- [`python/`](python/) contains the Python package and its tests.
- [`node/`](node/) contains the Node.js package and its tests.
- [`DEVELOPMENT.md`](DEVELOPMENT.md) documents architecture, local setup, test commands, and
  planned upstream integrations.

For SDK parameter behavior, supported models, and model-specific limits, see the
[ai-coustics SDK documentation](https://docs.ai-coustics.com/).
