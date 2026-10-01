# ai-coustics LiveKit extras for Python

Voice activity detection and audio-quality analysis for LiveKit Agents, backed by the public
`aic-sdk` package.

> [!IMPORTANT]
> The official
> [`livekit-plugins-ai-coustics`](https://pypi.org/project/livekit-plugins-ai-coustics/) package is
> the recommended integration path for most applications. It supports LiveKit Cloud, integrates
> more deeply with the LiveKit ecosystem and tooling, and offers stronger stability guarantees.
> Avoiding breaking changes is an explicit goal of the official plugin.
>
> This ai-coustics-maintained package follows a faster release cadence and is designed for teams
> that want early access to the latest ai-coustics models and product features. Some of those
> capabilities may be experimental and subject to change, and use of this package is billed
> separately through ai-coustics. Choose it when early adoption of new ai-coustics capabilities is
> important for your application.

## Installation

```bash
pip install ai-coustics-livekit-extras
export AIC_SDK_LICENSE=...
```

This package imports as `ai_coustics.livekit`. It is separate from `livekit-plugins-ai-coustics`,
which provides speech enhancement under `livekit.plugins.ai_coustics`, and can be installed
alongside it.

## Migrating from the official plugin's VAD

| Official LiveKit plugin | This package |
| --- | --- |
| `VAD()` and `VadSettings` | `VAD(model=..., vad_parameters=VADParameters(...))` |
| `Auth.livekit_cloud()` or `Auth.ai_coustics_api(...)` | `AIC_SDK_LICENSE` or `license_key=` |

The official VAD reads its results from the official enhancement processor. This package uses a
dedicated SDK VAD model instead. Provision a VAD model and install `vad.processor` as described
below. LiveKit Cloud authentication is not carried over; obtain an ai-coustics SDK license before
migrating.

## Model provisioning

Download models during deployment or container setup:

```python
from ai_coustics.livekit import Model

vad_path = Model.download("vad-2.1-xxs-16khz", "./models")
analysis_path = Model.download("tyto-1.1-l-16khz", "./models")
```

VAD and analysis models are different model types. Make the returned paths available to your
worker, then load each model once per worker process:

```python
vad_model = Model.from_file(vad_path)
analysis_model = Model.from_file(analysis_path)
```

## Usage

RoomIO accepts one frame processor in its `noise_cancellation` slot. A session installs either
`vad.processor` or `analyzer.collector` there.

### Voice activity detection

Create a `VAD` for each agent session:

```python
from livekit.agents import AgentSession, room_io

from ai_coustics.livekit import VAD

vad = VAD(model=vad_model)

session = AgentSession(
    vad=vad,
    # ... stt, llm, tts
)

await session.start(
    # ... agent, room
    room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(noise_cancellation=vad.processor),
    ),
)
```

`vad.processor` must be installed in the `noise_cancellation` path whenever the VAD is used. All
VAD streams read the immutable metadata it attaches to each frame, so the SDK model runs only once
per audio block.

### Audio-quality analysis

Create an `Analyzer`, install its collector in RoomIO's audio path, and subscribe to its results:

```python
from ai_coustics.livekit import AnalysisEvent, Analyzer

analyzer = Analyzer(
    model=analysis_model,
    analysis_interval=5.0,  # seconds; 5 is the default
)


@analyzer.on("analysis_result")
def on_analysis(event: AnalysisEvent) -> None:
    print(event.result.risk_score)


await session.start(
    # ... agent, room
    room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=analyzer.collector,
        ),
    ),
)
```

The `Analyzer` receives audio through `analyzer.collector`; constructing the analyzer without
installing its collector does not feed it any room audio. RoomIO closes the collector, and with it
the analyzer, when the input stream ends.

Results are not logged by the plugin; log or handle them in the callback.

## Configuration

Configure all SDK VAD parameters on the VAD factory:

```python
from ai_coustics.livekit import VADParameters

vad.set_parameters(
    VADParameters(
        sensitivity=0.5,
        speech_hold_duration=0.25,
        minimum_speech_duration=0.05,
    )
)
```

VAD durations are specified in seconds. See the
[Python SDK reference](https://docs.ai-coustics.com/reference/sdk/language-bindings/python) and
[VAD guide](https://docs.ai-coustics.com/models/voice-activity-detection/vad) for parameter ranges,
model support, and further details.

Set `AIC_SDK_LICENSE` or pass `license_key=` to the constructor. Create a new `VAD` or `Analyzer`
for each concurrent room; RoomIO closes its frame processor with the input stream.

Models must be provisioned explicitly. This package does not support
`python -m livekit.agents download-files`.
