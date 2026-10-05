# ai-coustics LiveKit extras for Python

Voice activity detection and Audio Insight for LiveKit Agents, backed by the
`aic-sdk` package.

> [!IMPORTANT]
> Use the official LiveKit plugin for ai-coustics speech enhancement. ai-coustics LiveKit extras
> adds product categories that the official plugin does not support yet. Features here can change
> between releases. Usage is billed through ai-coustics.

## Feature Availability

| Feature | `ai-coustics-livekit-extras` | `livekit-plugins-ai-coustics` |
| --- | --- | --- |
| Speech Enhancement | No | Yes |
| Voice Activity Detection (`VAD`) | Dedicated Models | Legacy |
| Audio Insight (Tyto `Analyzer`) | Yes | No |

## Installation

```bash
pip install ai-coustics-livekit-extras
export AIC_SDK_LICENSE=...
```

## Using this plugin next to the official plugin

You can install both packages in the same environment. This package imports as
`ai_coustics.livekit`, the official
[`livekit-plugins-ai-coustics`](https://pypi.org/project/livekit-plugins-ai-coustics/) package as
`livekit.plugins.ai_coustics`.

```bash
pip install livekit-plugins-ai-coustics ai-coustics-livekit-extras
```

RoomIO has one `noise_cancellation` slot. `FrameProcessorChain` lets `vad.processor` and
`analyzer.collector` share it. Install the official enhancement in that slot on its own, because
`FrameProcessorChain` does not pass LiveKit Cloud credentials to its processors.

This package needs its own ai-coustics license, even if the official plugin uses LiveKit Cloud.
Set `AIC_SDK_LICENSE` or pass `license_key=`.

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

RoomIO accepts one frame processor in its `noise_cancellation` slot. Install `vad.processor` or
`analyzer.collector` there directly, or combine them with `FrameProcessorChain` as described in
[Combining frame processors](#combining-frame-processors).

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

### Audio Insight

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

### Combining frame processors

Use `FrameProcessorChain` to run any number of processors in the same RoomIO audio path. For
example, this runs VAD inference and analysis in one session:

```python
from ai_coustics.livekit import FrameProcessorChain

frame_processor = FrameProcessorChain(vad.processor, analyzer.collector)

await session.start(
    # ... agent, room
    room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=frame_processor,
        ),
    ),
)
```

`FrameProcessorChain` runs its processors in order. Keep `vad.processor` first: it annotates the
original frame while preserving its audio. We recommend placing `analyzer.collector` before any
processor that changes the audio; measuring raw input makes it easier to understand how input audio
quality affects the rest of the pipeline.

This still uses LiveKit's `noise_cancellation` slot as a temporary integration. RoomIO owns the
chain and closes `vad.processor`, the collector, and the analyzer together.

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
