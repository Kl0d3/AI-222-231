# Full-Duplex Voice AI Agent

A real-time, interruptible, full-duplex voice AI agent integrating speech recognition, language model reasoning, speech synthesis, and Model Context Protocol (MCP) tools. The system uses a Windows laptop for voice interaction and a Raspberry Pi 5 for hosting remote MCP servers.

## Project Overview

This project implements a voice-controlled AI agent designed for natural, continuous conversation.

Unlike a conventional voice assistant that waits for a complete request before responding, the system supports full-duplex interaction: the user can speak while the agent is responding, interrupt playback, and continue the conversation.

The system integrates:

- Voice activity detection (VAD)
- Real-time speech recognition (ASR)
- Large language model (LLM) reasoning
- Text-to-speech (TTS) generation
- Interruptible audio playback
- Model Context Protocol (MCP) tool integration
- Background task execution and conversational feedback
- Latency benchmarking

## System Architecture

The implementation uses a distributed architecture.

### Windows Laptop â€” Voice Agent Runtime

The Windows machine handles the real-time conversational pipeline:

- Microphone input and audio buffering
- Silero VAD for speech detection
- Deepgram Flux for speech recognition
- Google Gemini API for language model reasoning
- Deepgram TTS for speech generation
- Kokoro ONNX as the local TTS fallback
- Audio playback and interruption handling
- MCP client connections and tool orchestration

### Raspberry Pi 5 â€” Remote MCP Servers

The Raspberry Pi 5 hosts five remote MCP servers:

1. **System:** System monitoring and device information.
2. **Information:** Weather and web search services.
3. **Productivity:** Google Calendar and task-related operations.
4. **Website:** Website generation and GitHub Pages integration.
5. **Smart Home:** Smart-home operations.

### Local MCP Server

The Music MCP server runs on the Windows machine and provides music search and playback functionality.

The voice agent discovers and orchestrates tools from all connected MCP servers.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.13 |
| LLM | Google Gemini API |
| ASR | Deepgram Flux |
| Primary TTS | Deepgram TTS |
| Fallback TTS | Kokoro ONNX |
| VAD | Silero VAD |
| Voice Pipeline Framework | Pipecat |
| MCP Framework | MCP |
| Audio Processing | NumPy, SoundDevice, SoundFile, SoXR |
| Local Model Runtime | ONNX Runtime, PyTorch |
| Raspberry Pi | Raspberry Pi 5 |

## Key Features

- Continuous voice interaction
- Full-duplex conversational behavior
- Speech interruption and playback cancellation
- Real-time speech recognition
- LLM-driven tool selection and execution
- Multiple MCP server connections
- Background MCP task execution
- Conversational speech feedback
- Primary and fallback TTS providers
- Audio latency measurement and benchmarking

## Project Structure

```text
ME3-Full-Duplex-Voice-AI-Agent/
â”‚
â”œâ”€â”€ config/
â”œâ”€â”€ credentials/
â”œâ”€â”€ docs/
â”œâ”€â”€ models/
â”‚   â””â”€â”€ kokoro/
â”œâ”€â”€ outputs/
â”‚   â””â”€â”€ websites/
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ agent/
â”‚   â”œâ”€â”€ audio/
â”‚   â”œâ”€â”€ benchmark/
â”‚   â”œâ”€â”€ llm/
â”‚   â”œâ”€â”€ mcp/
â”‚   â”œâ”€â”€ mcp_servers/
â”‚   â”‚   â”œâ”€â”€ information/
â”‚   â”‚   â”œâ”€â”€ music/
â”‚   â”‚   â”œâ”€â”€ productivity/
â”‚   â”‚   â”œâ”€â”€ smart_home/
â”‚   â”‚   â”œâ”€â”€ system/
â”‚   â”‚   â””â”€â”€ website/
â”‚   â””â”€â”€ tts/
â”œâ”€â”€ tests/
â”œâ”€â”€ requirements.txt
â””â”€â”€ requirements-pi.txt
```

## Development Environment

| Component | Environment |
|---|---|
| Voice Agent Runtime | Windows Laptop |
| Remote MCP Servers | Raspberry Pi 5 |
| Python | 3.13.14 |
| Development Environment | Visual Studio Code |
| Version Control | Git and GitHub |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Kl0d3/AI-222-231.git
cd AI-222-231/ME3-Full-Duplex-Voice-AI-Agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

For the Windows voice agent:

```bash
python -m pip install -r requirements.txt
```

For the Raspberry Pi MCP environment:

```bash
python -m pip install -r requirements-pi.txt
```

## Configuration

The system uses external services that require API credentials and, for some MCP integrations, OAuth configuration.

Configure the required environment variables and credentials locally before running the application.

**Important:** Never commit API keys, OAuth tokens, private credentials, or `.env` files to GitHub.

The `credentials/` directory is excluded from version control.

## Running the Project

The main voice-agent runtime coordinates microphone input, speech recognition, LLM reasoning, speech synthesis, audio playback, and MCP tool execution.

Start the runtime using the project's configured entry point.

The Raspberry Pi MCP servers must be reachable through the configured remote connections before using their tools.

Refer to the project configuration for the actual server startup and connection settings.

## Benchmarking

The project includes a dedicated benchmark module under `src/benchmark/`.

The benchmark infrastructure supports measuring the stages of speech interaction, including:

- VAD end
- ASR final result
- LLM first token
- TTS first audio
- Speaker audio output

End-to-end latency is measured from the end of detected user speech to speaker output.

Barge-in latency measures the time between the start of an interruption and the stopping of the current playback.

Actual benchmark results should be reported separately from the measurement methodology.

## Project Status

The core voice-agent pipeline and MCP integrations have been implemented and tested in the development environment.

The project includes:

- A working full-duplex voice-agent runtime
- Integrated ASR, LLM, and TTS components
- Interruptible audio playback
- Five Raspberry Pi-hosted MCP servers
- One locally hosted Music MCP server
- A multi-server MCP client
- A dedicated audio benchmarking module

Further hardware-specific testing and performance evaluation on the Raspberry Pi 5 remain separate from the Windows-hosted voice runtime.

## Academic Project

**Machine Exercise 3 â€” Full-Duplex Voice AI Agent**

Developed as part of the Master of Engineering in Artificial Intelligence program.
