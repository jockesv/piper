#!/usr/bin/env python3
import argparse
import io
import logging
import wave
from pathlib import Path
from typing import Any, Dict, Iterable # Added Iterable

from flask import Flask, request

from . import PiperVoice
from .download import ensure_voice_exists, find_voice, get_voices

_LOGGER = logging.getLogger()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="HTTP server host")
    parser.add_argument("--port", type=int, default=5000, help="HTTP server port")
    #
    parser.add_argument("-m", "--model", required=True, help="Path to Onnx model file")
    parser.add_argument("-c", "--config", help="Path to model config file")
    #
    parser.add_argument("-s", "--speaker", type=int, help="Id of speaker (default: 0)")
    parser.add_argument(
        "--length-scale", "--length_scale", type=float, help="Phoneme length"
    )
    parser.add_argument(
        "--noise-scale", "--noise_scale", type=float, help="Generator noise"
    )
    parser.add_argument(
        "--noise-w", "--noise_w", type=float, help="Phoneme width noise"
    )
    #
    parser.add_argument("--cuda", action="store_true", help="Use GPU")
    #
    parser.add_argument(
        "--sentence-silence",
        "--sentence_silence",
        type=float,
        default=0.0,
        help="Seconds of silence after each sentence",
    )
    #
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        action="append",
        default=[str(Path.cwd())],
        help="Data directory to check for downloaded models (default: current directory)",
    )
    parser.add_argument(
        "--download-dir",
        "--download_dir",
        help="Directory to download voices into (default: first data dir)",
    )
    #
    parser.add_argument(
        "--update-voices",
        action="store_true",
        help="Download latest voices.json during startup",
    )
    #
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    if not args.download_dir:
        # Download to first data directory by default
        args.download_dir = args.data_dir[0]

    # Download voice if file doesn't exist
    model_path = Path(args.model)
    if not model_path.exists():
        # Load voice info
        voices_info = get_voices(args.download_dir, update_voices=args.update_voices)

        # Resolve aliases for backwards compatibility with old voice names
        aliases_info: Dict[str, Any] = {}
        for voice_info in voices_info.values():
            for voice_alias in voice_info.get("aliases", []):
                aliases_info[voice_alias] = {"_is_alias": True, **voice_info}

        voices_info.update(aliases_info)
        ensure_voice_exists(args.model, args.data_dir, args.download_dir, voices_info)
        args.model, args.config = find_voice(args.model, args.data_dir)

    # Load voice
    voice = PiperVoice.load(args.model, config_path=args.config, use_cuda=args.cuda)
    
    # Store synthesis args default values from command line or Piper defaults
    # Allow overriding via request parameters later if desired
    synthesis_defaults = {
        "speaker_id": args.speaker,
        "length_scale": args.length_scale,
        "noise_scale": args.noise_scale,
        "noise_w": args.noise_w,
        "sentence_silence": args.sentence_silence,
    }

    # Create web server
    app = Flask(__name__)

    def generate_pcm_stream(text: str, synthesize_args: Dict[str, Any]) -> Iterable[bytes]:
        """Generate streaming raw PCM audio response."""
        # Stream raw audio data directly from the voice model
        # The synthesize_stream_raw method already yields bytes (raw PCM S16LE)
        for audio_bytes in voice.synthesize_stream_raw(text, **synthesize_args):
            yield audio_bytes

    @app.route("/", methods=["GET", "POST"])
    def app_synthesize() -> bytes:
        """Synthesize audio to a complete WAV file."""
        if request.method == "POST":
            # Prefer JSON body for POST for easier parameter passing
            if request.is_json:
                data = request.get_json()
                text = data.get("text", "")
                # Allow overriding synthesis parameters via JSON
                req_args = {k: data.get(k) for k in synthesis_defaults if k in data}
            else:
                text = request.data.decode("utf-8")
                req_args = {} # No overrides from plain text body
        else: # GET request
            text = request.args.get("text", "")
            # Allow overriding synthesis parameters via query string
            req_args = {k: request.args.get(k, type=type(synthesis_defaults[k])) 
                        for k in synthesis_defaults if k in request.args}

        text = text.strip()
        if not text:
            return "No text provided", 400

        # Combine defaults with request-specific args, filtering out None values
        current_synthesize_args = {**synthesis_defaults, **req_args}
        current_synthesize_args = {k: v for k, v in current_synthesize_args.items() if v is not None}

        _LOGGER.debug("Synthesizing (WAV) text: '%s' with args: %s", text, current_synthesize_args)
        with io.BytesIO() as wav_io:
            # Use wave module to create a proper WAV header and structure
            with wave.open(wav_io, "wb") as wav_file:
                voice.synthesize(text, wav_file, **current_synthesize_args)
            
            response_data = wav_io.getvalue()

        return app.response_class(
            response_data,
            mimetype="audio/wav"
        )

    @app.route("/stream", methods=["GET", "POST"])
    def app_synthesize_stream():
        """Stream synthesized audio as raw PCM."""
        if request.method == "POST":
            # Prefer JSON body for POST for easier parameter passing
            if request.is_json:
                data = request.get_json()
                text = data.get("text", "")
                 # Allow overriding synthesis parameters via JSON
                req_args = {k: data.get(k) for k in synthesis_defaults if k in data}
            else:
                text = request.data.decode("utf-8")
                req_args = {} # No overrides from plain text body
        else: # GET request
            text = request.args.get("text", "")
             # Allow overriding synthesis parameters via query string
            req_args = {k: request.args.get(k, type=type(synthesis_defaults[k])) 
                        for k in synthesis_defaults if k in request.args}

        text = text.strip()
        if not text:
             return "No text provided", 400

        # Combine defaults with request-specific args, filtering out None values
        current_synthesize_args = {**synthesis_defaults, **req_args}
        current_synthesize_args = {k: v for k, v in current_synthesize_args.items() if v is not None}

        _LOGGER.debug("Streaming (PCM) text: '%s' with args: %s", text, current_synthesize_args)

        # Define the PCM format based on Piper's output (16-bit signed integer, mono)
        # audio/L16 is a common MIME type for this. Parameters specify details.
        # Piper produces Little Endian by default on common platforms via numpy.
        pcm_format = (
            f"audio/L16; "
            f"rate={voice.config.sample_rate}; "
            f"channels=1; "
            # signed-integer is implied by L16, bits=16 implied by L16
            # endianness=little is typical but less standard to include here
        )

        # Return the generator. Flask/Werkzeug handles Chunked Transfer Encoding.
        return app.response_class(
            generate_pcm_stream(text, current_synthesize_args),
            mimetype=pcm_format
        )

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()