#!/usr/bin/env python3
"""
Jarvis-Granite Voice Demo Script

Generates audio files from AI race engineer responses using Watson TTS.
No LiveKit required - just Watson TTS credentials.

Setup:
    1. Copy .env.example to .env
    2. Add your Watson TTS credentials:
       WATSON_TTS_API_KEY=your_key
       WATSON_TTS_URL=https://api.us-south.text-to-speech.watson.cloud.ibm.com

Run:
    python demo_voice.py

Output:
    Creates .wav files in the 'voice_demo_output' folder that you can play.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ANSI colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.ENDC} {text}")


def print_error(text: str):
    print(f"{Colors.RED}[ERROR]{Colors.ENDC} {text}")


def print_info(text: str):
    print(f"{Colors.CYAN}[INFO]{Colors.ENDC} {text}")


def print_generating(text: str):
    print(f"{Colors.YELLOW}[GENERATING]{Colors.ENDC} {text}")


# Sample race engineer responses to demonstrate
SAMPLE_RESPONSES = [
    {
        "id": "01_fuel_warning",
        "trigger": "Fuel Warning",
        "text": "Fuel is getting low. You have about 4 laps remaining. Consider boxing at the end of this lap for fuel and fresh mediums."
    },
    {
        "id": "02_tire_critical",
        "trigger": "Tire Critical",
        "text": "Box box box! Front right is critical. Pit this lap, we're losing too much time."
    },
    {
        "id": "03_gap_closing",
        "trigger": "Gap Change",
        "text": "Gap to Hamilton is down to 1.2 seconds. He's pushing hard. Defend into turn 1, use the DRS on the straight."
    },
    {
        "id": "04_lap_complete",
        "trigger": "Lap Complete",
        "text": "Good lap. P3, 1:21.4. You're matching the leader's pace. Keep it clean."
    },
    {
        "id": "05_position_change",
        "trigger": "Position Change",
        "text": "Great move! You're now P2. Verstappen is 3.5 seconds ahead. 12 laps to go."
    },
    {
        "id": "06_driver_query_tires",
        "trigger": "Driver Query: Tires",
        "text": "Fronts are at 45% wear, rears at 38%. Temps are good, you should make it to the end but manage them through the chicanes."
    },
    {
        "id": "07_driver_query_strategy",
        "trigger": "Driver Query: Strategy",
        "text": "Current plan is to stay out. If the safety car comes, we box immediately for softs. Otherwise, push to the end on these mediums."
    },
    {
        "id": "08_safety_car",
        "trigger": "Safety Car",
        "text": "Safety car deployed. Box box box! This is our chance. Pit for softs, we'll come out P4 but with a tire advantage."
    },
]


async def generate_audio_files():
    """Generate audio files from sample responses."""

    print_header("JARVIS-GRANITE VOICE DEMO")

    # Check for credentials
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print_info("python-dotenv not installed, using environment variables directly")

    api_key = os.getenv("WATSON_TTS_API_KEY")
    service_url = os.getenv("WATSON_TTS_URL", "https://api.us-south.text-to-speech.watson.cloud.ibm.com")

    if not api_key or api_key == "your_tts_api_key":
        print_error("Watson TTS credentials not configured!")
        print()
        print("To set up Watson TTS:")
        print(f"  1. Create IBM Cloud account: {Colors.CYAN}https://cloud.ibm.com{Colors.ENDC}")
        print(f"  2. Create Text-to-Speech service")
        print(f"  3. Copy API key and URL to your .env file:")
        print()
        print(f"     {Colors.YELLOW}WATSON_TTS_API_KEY=your_actual_key{Colors.ENDC}")
        print(f"     {Colors.YELLOW}WATSON_TTS_URL=https://api.us-south.text-to-speech.watson.cloud.ibm.com{Colors.ENDC}")
        print()

        # Offer to create placeholder files for demonstration
        print_header("DEMO MODE (No Audio)")
        print("Creating placeholder info files to show what would be generated...\n")

        output_dir = Path("voice_demo_output")
        output_dir.mkdir(exist_ok=True)

        # Create a README in the output folder
        readme_content = "# Voice Demo Output\n\n"
        readme_content += "This folder would contain .wav audio files of the AI race engineer.\n\n"
        readme_content += "## Sample Responses\n\n"

        for response in SAMPLE_RESPONSES:
            readme_content += f"### {response['id']}.wav\n"
            readme_content += f"**Trigger:** {response['trigger']}\n\n"
            readme_content += f"> \"{response['text']}\"\n\n"
            print(f"{Colors.CYAN}[{response['trigger']}]{Colors.ENDC}")
            print(f"  \"{response['text'][:60]}...\"\n")

        readme_content += "\n## Setup Instructions\n\n"
        readme_content += "1. Get Watson TTS credentials from IBM Cloud\n"
        readme_content += "2. Add to .env file\n"
        readme_content += "3. Run `python demo_voice.py` again\n"

        readme_path = output_dir / "README.md"
        readme_path.write_text(readme_content)

        print_success(f"Created {readme_path}")
        print(f"\n{Colors.YELLOW}Add Watson TTS credentials to .env and run again to generate actual audio.{Colors.ENDC}")
        return

    # Credentials exist - generate actual audio
    print_info(f"Using Watson TTS at: {service_url}")
    print_info("Voice: en-GB_JamesV3Voice (British Male)\n")

    # Import the TTS client
    try:
        from jarvis_granite.voice.watson_tts import WatsonTTSClient
    except ImportError as e:
        print_error(f"Could not import WatsonTTSClient: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return

    # Create output directory
    output_dir = Path("voice_demo_output")
    output_dir.mkdir(exist_ok=True)

    # Create TTS client
    tts_client = WatsonTTSClient(
        api_key=api_key,
        service_url=service_url,
        voice="en-GB_JamesV3Voice",  # British male voice - sounds like race engineer
        timeout=30.0
    )

    print_header("GENERATING AUDIO FILES")

    generated_files = []

    for response in SAMPLE_RESPONSES:
        filename = f"{response['id']}.wav"
        filepath = output_dir / filename

        print_generating(f"{response['trigger']}")
        print(f"  Text: \"{response['text'][:50]}...\"")

        try:
            # Generate audio
            audio_bytes = await tts_client.synthesize(response['text'])

            # Save to file
            filepath.write_bytes(audio_bytes)

            # Calculate duration estimate (rough: ~150 words/min for speech)
            word_count = len(response['text'].split())
            duration_est = word_count / 2.5  # ~2.5 words per second

            print_success(f"Saved: {filepath} ({len(audio_bytes):,} bytes, ~{duration_est:.1f}s)")
            generated_files.append(filepath)

        except Exception as e:
            print_error(f"Failed to generate {filename}: {e}")

        print()

    # Summary
    print_header("DEMO COMPLETE")

    if generated_files:
        print_success(f"Generated {len(generated_files)} audio files in: {output_dir.absolute()}\n")
        print("Files created:")
        for f in generated_files:
            print(f"  - {f.name}")

        print(f"\n{Colors.CYAN}Play these files to hear the AI race engineer voice!{Colors.ENDC}")
        print(f"\nOn Windows: double-click any .wav file")
        print(f"On Mac/Linux: afplay {generated_files[0]} or aplay {generated_files[0]}")

        # Get stats
        stats = tts_client.get_stats()
        print(f"\n{Colors.BLUE}TTS Statistics:{Colors.ENDC}")
        print(f"  Requests: {stats['total_requests']}")
        print(f"  Avg Latency: {stats['average_latency_ms']:.0f}ms")
    else:
        print_error("No audio files were generated. Check your credentials.")


async def generate_custom_audio():
    """Interactive mode to generate custom audio."""

    print_header("CUSTOM VOICE GENERATION")

    # Check credentials
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("WATSON_TTS_API_KEY")
    service_url = os.getenv("WATSON_TTS_URL", "https://api.us-south.text-to-speech.watson.cloud.ibm.com")

    if not api_key or api_key == "your_tts_api_key":
        print_error("Watson TTS credentials required for custom audio generation.")
        return

    from jarvis_granite.voice.watson_tts import WatsonTTSClient

    tts_client = WatsonTTSClient(
        api_key=api_key,
        service_url=service_url,
        voice="en-GB_JamesV3Voice",
    )

    output_dir = Path("voice_demo_output")
    output_dir.mkdir(exist_ok=True)

    print("Enter text to convert to speech (or 'quit' to exit):\n")

    counter = 1
    while True:
        try:
            text = input(f"{Colors.CYAN}Text>{Colors.ENDC} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break

        if text.lower() in ('quit', 'exit', 'q'):
            break

        if not text:
            continue

        filename = f"custom_{counter:02d}.wav"
        filepath = output_dir / filename

        try:
            print_generating("Converting to speech...")
            audio_bytes = await tts_client.synthesize(text)
            filepath.write_bytes(audio_bytes)
            print_success(f"Saved: {filepath}")
            counter += 1
        except Exception as e:
            print_error(f"Failed: {e}")

        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Jarvis-Granite Voice Demo")
    parser.add_argument(
        "--custom", "-c",
        action="store_true",
        help="Interactive mode to generate custom audio"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Generate audio for specific text"
    )

    args = parser.parse_args()

    if args.custom:
        asyncio.run(generate_custom_audio())
    elif args.text:
        # Quick one-off generation
        async def quick_generate():
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass

            api_key = os.getenv("WATSON_TTS_API_KEY")
            service_url = os.getenv("WATSON_TTS_URL", "https://api.us-south.text-to-speech.watson.cloud.ibm.com")

            if not api_key or api_key == "your_tts_api_key":
                print_error("Watson TTS credentials required.")
                return

            from jarvis_granite.voice.watson_tts import WatsonTTSClient

            tts_client = WatsonTTSClient(
                api_key=api_key,
                service_url=service_url,
                voice="en-GB_JamesV3Voice",
            )

            output_dir = Path("voice_demo_output")
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = output_dir / f"output_{timestamp}.wav"

            print_generating(f"Converting: \"{args.text[:50]}...\"")
            audio_bytes = await tts_client.synthesize(args.text)
            filepath.write_bytes(audio_bytes)
            print_success(f"Saved: {filepath}")

        asyncio.run(quick_generate())
    else:
        asyncio.run(generate_audio_files())
