import {
  VADEventType,
  initializeLogger,
  voice,
  type VADStream,
} from "@livekit/agents";
import {
  AudioFrame,
  AudioSource,
  LocalAudioTrack,
  Room,
  TrackPublishOptions,
  TrackSource,
  dispose,
} from "@livekit/rtc-node";
import { AccessToken } from "livekit-server-sdk";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { randomUUID } from "node:crypto";
import { setTimeout as delay } from "node:timers/promises";
import { afterAll, describe, expect, it } from "vitest";

import { Model, VAD, type VADOptions } from "../src/index.js";

const describeIf = process.env.AIC_SDK_LICENSE ? describe : describe.skip;
const livekitUrl = process.env.LIVEKIT_URL ?? "ws://127.0.0.1:7880";
const livekitApiKey = process.env.LIVEKIT_API_KEY ?? "devkey";
const livekitApiSecret = process.env.LIVEKIT_API_SECRET ?? "secret";
const vadModelId =
  process.env.AIC_INTEGRATION_VAD_MODEL_ID ?? "vad-2.1-xxs-16khz";
const modelDir =
  process.env.AIC_INTEGRATION_MODEL_DIR ??
  path.join(os.homedir(), ".cache", "aic-sdk", "models");

const inputSampleRate = 16_000;
const publishSampleRate = 48_000;
const publishFrameMs = 20;
const licenseGracePeriodMs = 10_000;

initializeLogger({ pretty: false, level: "error" });

class ObservedVAD extends VAD {
  readonly createdAt = Date.now();
  inferenceTimes: number[] = [];
  processorClosed = false;

  constructor(options: VADOptions) {
    super(options);
    const close = this.processor.close.bind(this.processor);
    this.processor.close = () => {
      this.processorClosed = true;
      close();
    };
  }

  override stream(): VADStream {
    const stream = super.stream();
    const next = stream.next.bind(stream);
    stream.next = async () => {
      const result = await next();
      if (!result.done && result.value.type === VADEventType.INFERENCE_DONE) {
        this.inferenceTimes.push(Date.now());
      }
      return result;
    };
    return stream;
  }
}

async function token({
  identity,
  roomName,
  agentParticipant = false,
}: {
  identity: string;
  roomName: string;
  agentParticipant?: boolean;
}): Promise<string> {
  const accessToken = new AccessToken(livekitApiKey, livekitApiSecret, {
    identity,
    ttl: "10m",
  });
  accessToken.addGrant({
    room: roomName,
    roomJoin: true,
    roomCreate: true,
    canPublish: true,
    canSubscribe: true,
    agent: agentParticipant,
  });
  if (agentParticipant) {
    accessToken.kind = "agent";
  }
  return await accessToken.toJwt();
}

async function connectRoom({
  identity,
  roomName,
  agentParticipant = false,
}: {
  identity: string;
  roomName: string;
  agentParticipant?: boolean;
}): Promise<Room> {
  const room = new Room();
  await room.connect(
    livekitUrl,
    await token({ identity, roomName, agentParticipant }),
    { autoSubscribe: true, dynacast: false },
  );
  return room;
}

async function publishTestSignal(source: AudioSource, until: number): Promise<void> {
  const samplesPerFrame = (publishSampleRate * publishFrameMs) / 1000;
  let frameIndex = 0;
  let nextTick = Date.now();

  while (Date.now() < until) {
    const data = new Int16Array(samplesPerFrame);
    for (let sample = 0; sample < samplesPerFrame; sample += 1) {
      const position = (frameIndex * samplesPerFrame + sample) / publishSampleRate;
      const signal = 0.28 * Math.sin(2 * Math.PI * 220 * position);
      const noise = 0.1 * Math.sin(2 * Math.PI * 997 * position);
      data[sample] = Math.max(
        -32768,
        Math.min(32767, Math.round((signal + noise) * 32768)),
      );
    }
    await source.captureFrame(
      new AudioFrame(data, publishSampleRate, 1, samplesPerFrame),
    );

    frameIndex += 1;
    nextTick += publishFrameMs;
    const wait = nextTick - Date.now();
    if (wait > 0) {
      await delay(wait);
    } else {
      nextTick = Date.now();
    }
  }
}

afterAll(async () => {
  await dispose();
});

describeIf("VAD in a real AgentSession room", () => {
  it(
    "consumes VAD events after the license grace period",
    async () => {
      fs.mkdirSync(modelDir, { recursive: true });
      const vadModel = Model.fromFile(await Model.download(vadModelId, modelDir));
      const detector = new ObservedVAD({ model: vadModel });
      const roomName = `ai-coustics-e2e-${randomUUID()}`;
      let agentRoom: Room | undefined;
      let publisherRoom: Room | undefined;
      let source: AudioSource | undefined;
      let session: voice.AgentSession | undefined;
      let sessionStarted = false;
      let processorClosedByRoomIO = false;

      try {
        agentRoom = await connectRoom({
          identity: "agent",
          roomName,
          agentParticipant: true,
        });
        publisherRoom = await connectRoom({ identity: "publisher", roomName });

        session = new voice.AgentSession({
          vad: detector,
          turnHandling: { turnDetection: "manual" },
          userAwayTimeout: null,
        });
        await session.start({
          agent: new voice.Agent({ instructions: "E2E audio probe" }),
          room: agentRoom,
          record: false,
          inputOptions: {
            audioSampleRate: inputSampleRate,
            audioNumChannels: 1,
            participantIdentity: "publisher",
            noiseCancellation: detector.processor,
            textEnabled: false,
            videoEnabled: false,
          },
          outputOptions: {
            audioEnabled: false,
            transcriptionEnabled: false,
          },
        });
        sessionStarted = true;

        source = new AudioSource(publishSampleRate, 1);
        const track = LocalAudioTrack.createAudioTrack("e2e-microphone", source);
        const publishOptions = new TrackPublishOptions();
        publishOptions.source = TrackSource.SOURCE_MICROPHONE;
        await publisherRoom.localParticipant!.publishTrack(track, publishOptions);

        // Continue beyond the SDK's authentication grace period, while guaranteeing at least
        // three seconds of actual room audio if room setup itself took unusually long.
        const streamUntil = Math.max(Date.now() + 3_000, detector.createdAt + 12_000);
        await publishTestSignal(source, streamUntil);

        expect(detector.inferenceTimes.length).toBeGreaterThan(0);
        expect(
          detector.inferenceTimes.some(
            (inferredAt) => inferredAt >= detector.createdAt + licenseGracePeriodMs,
          ),
        ).toBe(true);
      } finally {
        if (sessionStarted) {
          await session!.close();
        } else {
          detector.processor.close();
        }
        processorClosedByRoomIO = detector.processorClosed;
        await source?.close();
        await publisherRoom?.disconnect();
        await agentRoom?.disconnect();
        await detector.close();
      }

      expect(processorClosedByRoomIO).toBe(true);
    },
    45_000,
  );
});
