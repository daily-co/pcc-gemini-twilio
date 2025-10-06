#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Live + Twilio Example.

TODO: add description.

Required AI services:
- Google Gemini Live (LLM)
- Twilio (Voice)

Run the bot locally:

    Using Twilio::

        uv run bot.py --transport twilio --proxy your_url.ngrok.io

    Using SmallWebRTC::

        uv run bot.py
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, TranscriptionMessage
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    llm = GeminiMultimodalLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        voice_id="Charon",  # Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, and Zephyr
        system_instruction="You are a friendly AI assistant. Respond naturally and keep your answers conversational.",
    )

    messages = [
        {
            "role": "user",
            "content": "Say hello. Then ask if I want to hear a joke.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # Emit `on_transcript_update` events for the user and assistant spoken responses
    transcript = TranscriptProcessor()

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            context_aggregator.user(),  # User respones
            transcript.user(),  # User transcript
            llm,  # LLM
            transport.output(),  # Transport bot output
            transcript.assistant(),  # Assistant transcript
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    # Register event handler for transcript updates
    @transcript.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        for msg in frame.messages:
            if isinstance(msg, TranscriptionMessage):
                timestamp = f"[{msg.timestamp}] " if msg.timestamp else ""
                line = f"{timestamp}{msg.role}: {msg.content}"
                logger.info(f"Transcript: {line}")

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
