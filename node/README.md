# ai-coustics LiveKit extras for Node.js

Voice activity detection and audio-quality analysis for LiveKit Agents, backed by the public
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

```bash
npm install @ai-coustics/livekit-extras
export AIC_SDK_LICENSE=...
```

This package is separate from `@livekit/plugins-ai-coustics`, which provides speech enhancement,
and can be installed alongside it.

## Migrating from the official plugin's VAD

| Official LiveKit plugin | This package |
| --- | --- |
| `vad()` and `vadSettings` | `new VAD({ model, vadParameters })` |
| `Auth.livekitCloud()` or `Auth.aiCousticsApi(...)` | `AIC_SDK_LICENSE` or `licenseKey` |

The official VAD reads its results from the official enhancement processor. This package runs a
dedicated SDK VAD model instead and needs nothing in RoomIO's `noiseCancellation` slot, so the
enhancement processor can stay there. Provision a VAD model as described below. LiveKit Cloud
authentication is not carried over; obtain an ai-coustics SDK license before migrating.

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

The VAD needs nothing in RoomIO's `noiseCancellation` slot. That slot holds one frame processor,
such as an enhancer or `analyzer.collector`, and the VAD works alongside either.

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
```

The VAD runs the SDK model on the audio AgentSession passes to it, after any processor in the
`noiseCancellation` slot. The SDK recommends running its VAD on original audio. Behind an
enhancer, detection can differ slightly.

Each VAD stream holds one native SDK VAD. When LiveKit replaces a stream, the new stream takes over
the same native VAD with its model history. Every stream of a `VAD` must see the same audio, so do
not share one `VAD` between sessions. A non-streaming STT adds a second concurrent stream, which
uses a second native VAD and doubles inference. Model history persists when RoomIO switches to
another participant's track within one stream.

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
installing its collector does not feed it any room audio. RoomIO closes the collector, and with it
the analyzer, when the input stream ends.

Results are not logged by the plugin; log or handle them in the callback.

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
each concurrent room. RoomIO closes the analyzer's collector with the input stream; a `VAD` ends
its SDK sessions at job shutdown or on `vad.close()`.

Models must be provisioned explicitly. This package does not support
`npx livekit-agents download-files`.
