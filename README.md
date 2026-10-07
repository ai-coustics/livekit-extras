# ai-coustics extras for LiveKit Agents

This repository contains the ai-coustics-maintained Python and Node.js integrations for
[LiveKit Agents](https://docs.livekit.io/agents/). They are thin wrappers around the public
ai-coustics SDKs and let LiveKit agents use:

- `VAD` for voice activity detection with a dedicated SDK VAD model, through `vad.processor` and
  LiveKit's streaming VAD interface.
- `Analyzer` for periodic Audio Insight analysis.
- `FrameProcessorChain` for applying any number of frame processors in sequence.

Applications choose and provision the SDK models themselves. This keeps model selection, storage,
and deployment under application control while the plugins handle LiveKit audio formats, stream
lifecycle, observability, and error recovery.

> [!IMPORTANT]
> Use the official LiveKit plugin for ai-coustics speech enhancement. ai-coustics LiveKit extras
> adds product categories that the official plugin does not support yet. Features here can change
> between releases. Usage is billed through ai-coustics.

## Feature Availability

| Feature | ai-coustics LiveKit extras | Official LiveKit plugin |
| --- | --- | --- |
| Speech Enhancement | No | Yes |
| Voice Activity Detection (`VAD`) | Dedicated Models | Legacy |
| Audio Insight (Tyto `Analyzer`) | Yes | No |

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

For SDK parameter behavior, supported models, and model-specific limits, see the
[ai-coustics SDK documentation](https://docs.ai-coustics.com/).
