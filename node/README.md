# ai-coustics LiveKit extras for Node.js

Audio enhancement, voice activity detection, and audio-quality analysis for LiveKit Agents, backed by the public
`@ai-coustics/aic-sdk` package.

> [!IMPORTANT]
> The official
> [`@livekit/plugins-ai-coustics`](https://www.npmjs.com/package/@livekit/plugins-ai-coustics)
> package is the recommended integration path for most applications. It supports LiveKit Cloud,
> integrates more deeply with the LiveKit ecosystem and tooling, and offers stronger stability
> guarantees. Avoiding breaking changes is an explicit goal of the official plugin.
>
> This ai-coustics-maintained package follows a faster release cadence and is designed for teams
> that want early access to the latest ai-coustics models and product features. Some of those
> capabilities may be experimental and subject to change, and use of this package is billed
> separately through ai-coustics. Choose it when early adoption of new ai-coustics capabilities is
> important for your application.

## Installation

> This package is an alternative to `@livekit/plugins-ai-coustics`, not an extension of it. If you
> choose to migrate, uninstall the official package and do not mix objects from the two
> implementations.

```bash
npm uninstall @livekit/plugins-ai-coustics
npm install @ai-coustics/livekit-extras
export AIC_SDK_LICENSE=...
```

## Migrating from the official LiveKit plugin

Replace the package import and update the following APIs:

| Official LiveKit plugin | This package |
| --- | --- |
| `audioEnhancement(...)` | `new Processor({ model })` |
| `EnhancerModel.*` | An SDK `Model` loaded from a provisioned model file |
| `modelParameters` or `updateModelParameters(...)` | `processor.getContext().setParameter(...)` |
| `vad()` and `vadSettings` | `new VAD({ model, vadParameters })` |
| `Auth.livekitCloud()` or `Auth.aiCousticsApi(...)` | `AIC_SDK_LICENSE` or `licenseKey` |

Before:

```ts
import * as aic from "@livekit/plugins-ai-coustics";

const processor = aic.audioEnhancement({ model: aic.EnhancerModel.QuailL });
const vad = aic.vad();
```

After loading the SDK models as described below:

```ts
import { FrameProcessorChain, Processor, VAD } from "@ai-coustics/livekit-extras";

const processor = new Processor({ model: enhancementModel });
const vad = new VAD({ model: vadModel });
const frameProcessor = new FrameProcessorChain(vad.processor, processor);
```

This package still uses a dedicated SDK VAD model. Like the official plugin, inference runs in the
RoomIO frame-processor path and the VAD streams consume frame metadata. Provision a separate VAD
model and install `vad.processor` as described below. LiveKit Cloud authentication is not carried
over; obtain an ai-coustics SDK license before migrating.

## Model provisioning

Download models during deployment or container setup:

```ts
import { Model } from "@ai-coustics/livekit-extras";

const enhancementPath = Model.download("quail-vf-2.2-l-16khz", "./models");
const vadPath = Model.download("vad-2.1-xxs-16khz", "./models");
const analysisPath = Model.download("tyto-1.1-l-16khz", "./models");
```

Enhancement and VAD models are different model types. Make the returned paths available to your
worker, then load each model once per worker process:

```ts
const enhancementModel = Model.fromFile(enhancementPath);
const vadModel = Model.fromFile(vadPath);
const analysisModel = Model.fromFile(analysisPath);
```

## Usage

Create a `Processor` and `VAD` for each agent session:

```ts
import { voice } from "@livekit/agents";
import { FrameProcessorChain, Processor, VAD } from "@ai-coustics/livekit-extras";

const processor = new Processor({ model: enhancementModel });
const vad = new VAD({ model: vadModel });
const frameProcessor = new FrameProcessorChain(vad.processor, processor);

const session = new voice.AgentSession({
  vad,
  // ... stt, llm, tts
});

await session.start({
  // ... agent, room
  inputOptions: { noiseCancellation: frameProcessor },
});
```

`vad.processor` must be installed in the `noiseCancellation` path whenever the VAD is used. Put it
first in the chain so it runs on original microphone audio before enhancement. All VAD streams
read the resulting immutable metadata, so the SDK model runs only once per audio block.

For VAD without enhancement, use `noiseCancellation: vad.processor`. For enhancement without VAD,
use `noiseCancellation: processor`.

### Audio-quality analysis

Create an `Analyzer`, install its collector in RoomIO's audio path, and subscribe to its results:

```ts
import { Analyzer } from "@ai-coustics/livekit-extras";

const analyzer = new Analyzer({
  model: analysisModel,
  analysisInterval: 5, // seconds; 5 is the default
});

analyzer.on("analysisResult", (event) => {
  console.log(event.result.riskScore);
});

await session.start({
  // ... agent, room
  inputOptions: { noiseCancellation: analyzer.collector },
});
```

The `Analyzer` receives audio through `analyzer.collector`; constructing the analyzer without
installing its collector does not feed it any room audio.

Results are not logged by the plugin; log or handle them in the callback.

### Combining frame processors

Use `FrameProcessorChain` to run any number of processors in the same RoomIO audio path. For
example, this runs VAD inference and analysis on the raw input before enhancement:

```ts
import { FrameProcessorChain } from "@ai-coustics/livekit-extras";

const frameProcessor = new FrameProcessorChain(
  vad.processor,
  analyzer.collector,
  processor,
);

await session.start({
  // ... agent, room
  inputOptions: { noiseCancellation: frameProcessor },
});
```

`FrameProcessorChain` runs its processors in order. Keep `vad.processor` first: it annotates the
original frame while preserving its audio. We recommend placing `analyzer.collector` before
`processor`; measuring raw input makes it easier to understand how input audio quality affects the
rest of the pipeline.

This still uses LiveKit's `noiseCancellation` slot as a temporary integration. RoomIO owns the
chain and closes the processor, collector, and analyzer together.

## Configuration

Set the enhancement level through the Processor context, and configure all SDK VAD parameters on
the VAD factory:

```ts
import { ProcessorParameter } from "@ai-coustics/livekit-extras";

processor.getContext().setParameter(ProcessorParameter.EnhancementLevel, 0.8);
vad.setParameters({
  sensitivity: 0.5,
  speechHoldDuration: 0.25,
  minimumSpeechDuration: 0.05,
});
```

SDK VAD parameter durations (`speechHoldDuration` and `minimumSpeechDuration`) are specified in
seconds. The `VAD` constructor options `prefixPaddingDuration` and `maxBufferedSpeech` are specified
in milliseconds. See the
[Node.js SDK reference](https://docs.ai-coustics.com/reference/sdk/language-bindings/nodejs) and
[VAD guide](https://docs.ai-coustics.com/models/voice-activity-detection/vad) for parameter ranges,
model support, and further details.

Set `AIC_SDK_LICENSE` or pass `licenseKey` to the constructor. Create a new Processor for each
concurrent room; RoomIO closes it with the input stream.

Models must be provisioned explicitly. This package does not support
`npx livekit-agents download-files`.
