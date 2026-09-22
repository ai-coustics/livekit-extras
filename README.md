# ai-coustics extras for LiveKit Agents

This repository contains the ai-coustics-maintained Python and Node.js integrations for
[LiveKit Agents](https://docs.livekit.io/agents/). They are thin wrappers around the public
ai-coustics SDKs and let LiveKit agents use:

- `Processor` for speech enhancement through LiveKit's audio `FrameProcessor` interface.
- `FrameProcessorChain` for applying any number of frame processors in sequence.
- `VAD` for shared voice-activity inference through `vad.processor` and LiveKit's streaming VAD
  interface.
- `Analyzer` for periodic audio-quality analysis.

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

The runtime guides cover migration from the official plugins, model provisioning, AgentSession
integration, Processor enhancement settings, VAD parameters, authentication, and operational
constraints.

> These packages are alternatives to LiveKit's official
> [`livekit-plugins-ai-coustics`](https://pypi.org/project/livekit-plugins-ai-coustics/) and
> [`@livekit/plugins-ai-coustics`](https://www.npmjs.com/package/@livekit/plugins-ai-coustics)
> packages, not extensions of them. If you choose to migrate, uninstall the official packages for
> your runtime and do not mix objects from the two implementations.

## Repository contents

- [`python/`](python/) contains the Python package and its tests.
- [`node/`](node/) contains the Node.js package and its tests.
- [`DEVELOPMENT.md`](DEVELOPMENT.md) documents architecture, local setup, test commands, and
  planned upstream integrations.

For SDK parameter behavior, supported models, and model-specific limits, see the
[ai-coustics SDK documentation](https://docs.ai-coustics.com/).
