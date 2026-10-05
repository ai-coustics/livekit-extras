# ai-coustics LiveKit extras for Node.js

Voice activity detection and Audio Insight for LiveKit Agents, backed by the public
`@ai-coustics/aic-sdk` package.

> [!IMPORTANT]
> Use the official LiveKit plugin for ai-coustics speech enhancement. ai-coustics LiveKit extras
> adds product categories that the official plugin does not support yet. Features here can change
> between releases. Usage is billed through ai-coustics.

## Feature Availability

| Feature | `@ai-coustics/livekit-extras` | `@livekit/plugins-ai-coustics` |
| --- | --- | --- |
| Speech Enhancement | No | Yes |
| Voice Activity Detection (`VAD`) | Dedicated Models | Legacy |
| Audio Insight (Tyto `Analyzer`) | Yes | No |

## Installation

```bash
npm install @ai-coustics/livekit-extras
export AIC_SDK_LICENSE=...
```

## Using this plugin next to the official plugin

You can install this package and the official
[`@livekit/plugins-ai-coustics`](https://www.npmjs.com/package/@livekit/plugins-ai-coustics)
package in the same project.

```bash
npm install @livekit/plugins-ai-coustics @ai-coustics/livekit-extras
```

RoomIO has one `noiseCancellation` slot. `FrameProcessorChain` lets `vad.processor` and
`analyzer.collector` share it. Install the official enhancement in that slot on its own, because
`FrameProcessorChain` does not pass LiveKit Cloud credentials to its processors.

This package needs its own ai-coustics license, even if the official plugin uses LiveKit Cloud.
Set `AIC_SDK_LICENSE` or pass `licenseKey`.

## Model provisioning

Download models during deployment or container setup:

```ts
import { Model } from "@ai-coustics/livekit-extras";

const vadPath = await Model.download("vad-2.1-xxs-16khz", "./models");
const analysisPath = await Model.download("tyto-1.1-l-16khz", "./models");
```

VAD and analysis models are different model types. Make the returned paths available to your
worker, then load each model once per worker process:

```ts
const vadModel = Model.fromFile(vadPath);
const analysisModel = Model.fromFile(analysisPath);
```

## Usage

RoomIO accepts one frame processor in its `noiseCancellation` slot. Install `vad.processor` or
`analyzer.collector` there directly, or combine them with `FrameProcessorChain` as described in
[Combining frame processors](#combining-frame-processors).

### Voice activity detection

Create a `VAD` for each agent session:

```ts
import { voice } from "@livekit/agents";
import { VAD } from "@ai-coustics/livekit-extras";

const vad = new VAD({ model: vadModel });

const session = new voice.AgentSession({
  vad,
  // ... stt, llm, tts
});

await session.start({
  // ... agent, room
  inputOptions: { noiseCancellation: vad.processor },
});
```

`vad.processor` must be installed in the `noiseCancellation` path whenever the VAD is used. All VAD
streams read the immutable metadata it attaches to each frame, so the SDK model runs only once per
audio block.

### Audio Insight

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
installing its collector does not feed it any room audio. RoomIO closes the collector, and with it
the analyzer, when the input stream ends.

Results are not logged by the plugin; log or handle them in the callback.

### Combining frame processors

Use `FrameProcessorChain` to run any number of processors in the same RoomIO audio path. For
example, this runs VAD inference and analysis in one session:

```ts
import { FrameProcessorChain } from "@ai-coustics/livekit-extras";

const frameProcessor = new FrameProcessorChain(vad.processor, analyzer.collector);

await session.start({
  // ... agent, room
  inputOptions: { noiseCancellation: frameProcessor },
});
```

`FrameProcessorChain` runs its processors in order. Keep `vad.processor` first: it annotates the
original frame while preserving its audio. We recommend placing `analyzer.collector` before any
processor that changes the audio; measuring raw input makes it easier to understand how input audio
quality affects the rest of the pipeline.

This still uses LiveKit's `noiseCancellation` slot as a temporary integration. RoomIO owns the
chain and closes `vad.processor`, the collector, and the analyzer together.

## Configuration

Configure all SDK VAD parameters on the VAD factory:

```ts
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

Set `AIC_SDK_LICENSE` or pass `licenseKey` to the constructor. Create a new `VAD` or `Analyzer` for
each concurrent room; RoomIO closes its frame processor with the input stream.

Models must be provisioned explicitly. This package does not support
`npx livekit-agents download-files`.
