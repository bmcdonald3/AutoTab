#!/usr/bin/env python3
"""
AutoTab: Automatically convert MP3 files to guitar tabs using AI.
Pipeline: Download → Isolate (Demucs) → Transcribe (Basic Pitch) → Optimize (Tuttut)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from shutil import copy2


def setup_directories():
    """Create temp and output directories if they don't exist."""
    temp_dir = Path("temp")
    output_dir = Path("output")
    temp_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    # Ensure Demucs cache directory exists with proper permissions
    cache_dir = Path.home() / ".cache" / "demucs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Set permissions to 0o755 (read/write/execute for owner, read/execute for others)
    os.chmod(cache_dir, 0o755)
    return temp_dir, output_dir


def check_dependencies():
    """Verify required CLI tools are available."""
    missing = []
    for cmd in ["demucs", "basic-pitch", "tuttut", "yt-dlp"]:
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=False)
        except FileNotFoundError:
            missing.append(cmd)
    
    if missing:
        print(f"Error: Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        sys.exit(1)


def validate_demucs_model():
    """Check if the htdemucs model is available."""
    print("Validating Demucs model...")
    try:
        # Use Python to check if model can be loaded
        check_script = """
import demucs.pretrained
try:
    demucs.pretrained.get_model('htdemucs')
    print("Model available")
except Exception as e:
    print(f"Error: {e}")
    exit(1)
"""
        result = subprocess.run(
            [sys.executable, "-c", check_script],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Demucs model validation failed: {result.stderr}")
            print("The model will be downloaded automatically on first use.")
        else:
            print("  ✓ Demucs model validated")
    except Exception as e:
        print(f"Warning: Could not validate model: {e}")


def download_audio(url: str, temp_dir: Path) -> Path:
    """Download audio from URL using yt-dlp."""
    print(f"[1/4] Downloading audio from URL...")
    output_template = str(temp_dir / "%(title)s.%(ext)s")
    
    try:
        subprocess.run([
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "mp3",
            "--no-playlist",  # Download single video only
            "--progress",  # Show progress bar
            "-o", output_template,
            url
        ], check=True, capture_output=True)
        
        # Find the downloaded file
        downloaded_files = list(temp_dir.glob("*.mp3"))
        if not downloaded_files:
            raise FileNotFoundError("No MP3 file downloaded")
        return downloaded_files[-1]  # Return most recent
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e.stderr.decode()}")
        sys.exit(1)


def get_input_file(args) -> Path:
    """Determine input file path from URL or local path."""
    temp_dir, _ = setup_directories()
    
    if args.url:
        return download_audio(args.url, temp_dir)
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)
        # Copy to temp for consistent processing
        dest = temp_dir / input_path.name
        copy2(input_path, dest)
        return dest
    else:
        print("Error: Must provide either --url or --input")
        sys.exit(1)


def isolate_guitar(input_file: Path, temp_dir: Path) -> Path:
    """Run Demucs v4 to isolate guitar stem using htdemucs (4-stem) with fallback to 'other'."""
    print(f"[2/4] Isolating guitar stem with Demucs...")
    
    try:
        # Demucs outputs to a 'separated' directory by default
        result = subprocess.run([
            "demucs",
            "--repo", str(Path.home() / ".cache" / "demucs"),
            "-n", "htdemucs",  # Use 4-stem model (most stable)
            "-d", "cpu",
            "--verbose",
            str(input_file)
        ], check=True, capture_output=True, text=True)
        
        # Find the guitar stem
        # For htdemucs (4-stem), guitar is typically in 'other' stem
        track_name = input_file.stem
        separated_dir = Path("separated") / "htdemucs" / track_name
        
        # Check for guitar.wav first (if using 6-stem model)
        guitar_stem = separated_dir / "guitar.wav"
        if not guitar_stem.exists():
            # Fallback to 'other' stem for 4-stem model
            guitar_stem = separated_dir / "other.wav"
        
        if not guitar_stem.exists():
            # Try alternative: list all stems and pick the most likely
            if separated_dir.exists():
                stems = list(separated_dir.glob("*.wav"))
                if stems:
                    # Prefer 'guitar' if available, otherwise 'other'
                    for stem in stems:
                        if "guitar" in stem.name.lower():
                            guitar_stem = stem
                            break
                    if not guitar_stem.exists():
                        # Use 'other' as last resort
                        for stem in stems:
                            if "other" in stem.name.lower():
                                guitar_stem = stem
                                break
                    # If still not found, use first available
                    if not guitar_stem.exists() and stems:
                        guitar_stem = stems[0]
        
        if not guitar_stem.exists():
            raise FileNotFoundError("Guitar stem not found after Demucs separation")
        
        print(f"  Using stem: {guitar_stem.name}")
        
        # Copy to temp for next steps
        dest = temp_dir / "guitar_stem.wav"
        copy2(guitar_stem, dest)
        return dest
    except subprocess.CalledProcessError as e:
        print(f"Demucs isolation failed: {e.stderr}")
        sys.exit(1)


def transcribe_to_midi(guitar_stem: Path, temp_dir: Path) -> Path:
    """Run Basic Pitch to transcribe guitar stem to MIDI."""
    print(f"[3/4] Transcribing to MIDI with Basic Pitch...")
    
    output_midi = temp_dir / "output.mid"
    
    try:
        subprocess.run([
            "basic-pitch",
            str(guitar_stem),
            "--output-midi", str(output_midi)
        ], check=True, capture_output=True)
        
        if not output_midi.exists():
            raise FileNotFoundError("MIDI file not generated by Basic Pitch")
        
        return output_midi
    except subprocess.CalledProcessError as e:
        print(f"Basic Pitch transcription failed: {e.stderr}")
        sys.exit(1)


def convert_to_tab(midi_file: Path, output_dir: Path, tuning: str) -> Path:
    """Convert MIDI to guitar tab using Tuttut."""
    print(f"[4/4] Converting to guitar tab with Tuttut...")
    
    output_tab = output_dir / f"{midi_file.stem}.txt"
    
    try:
        # Tuttut command - check if tuning flag is supported
        cmd = ["tuttut", str(midi_file), "-o", str(output_tab)]
        if tuning:
            cmd.extend(["--tuning", tuning])
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        if not output_tab.exists():
            # Tuttut might output to stdout, try alternative approach
            result = subprocess.run(
                ["tuttut", str(midi_file)],
                check=True,
                capture_output=True,
                text=True
            )
            output_tab.write_text(result.stdout)
        
        return output_tab
    except subprocess.CalledProcessError as e:
        print(f"Tuttut conversion failed: {e.stderr}")
        sys.exit(1)


def cleanup_temp_files(temp_dir: Path):
    """Remove large intermediate files from temp directory."""
    print("Cleaning up temporary files...")
    # Delete .wav and .mid files from temp
    for ext in ["*.wav", "*.mid"]:
        for file in temp_dir.glob(ext):
            try:
                file.unlink()
            except Exception as e:
                print(f"Warning: Could not delete {file}: {e}")
    
    # Also clean up separated directory if it exists (from Demucs)
    separated_dir = Path("separated")
    if separated_dir.exists():
        try:
            shutil.rmtree(separated_dir)
        except Exception as e:
            print(f"Warning: Could not delete separated directory: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="AutoTab: Convert MP3 to guitar tab using AI"
    )
    parser.add_argument(
        "--url",
        help="URL of audio to download (YouTube, etc.)"
    )
    parser.add_argument(
        "--input",
        help="Local MP3 file path"
    )
    parser.add_argument(
        "--tuning",
        default="standard",
        help="Guitar tuning (e.g., standard, drop_d, d_standard, etc.)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.url and not args.input:
        parser.error("Must provide either --url or --input")
    
    print("=" * 50)
    print("AutoTab - AI-Powered Guitar Tab Transcription")
    print("=" * 50)
    
    # Check dependencies
    check_dependencies()
    
    # Setup directories
    temp_dir, output_dir = setup_directories()
    
    # Validate Demucs model
    validate_demucs_model()
    
    try:
        # Step 1: Get input file
        input_file = get_input_file(args)
        print(f"  Input: {input_file.name}")
        
        # Step 2: Isolate guitar
        guitar_stem = isolate_guitar(input_file, temp_dir)
        print(f"  Guitar stem: {guitar_stem.name}")
        
        # Step 3: Transcribe to MIDI
        midi_file = transcribe_to_midi(guitar_stem, temp_dir)
        print(f"  MIDI: {midi_file.name}")
        
        # Step 4: Convert to tab
        tab_file = convert_to_tab(midi_file, output_dir, args.tuning)
        print(f"  Tab: {tab_file}")
        
        # Cleanup intermediate files
        cleanup_temp_files(temp_dir)
        
        print("=" * 50)
        print(f"✓ Tab saved to: {tab_file}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()