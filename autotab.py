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
    for cmd in ["demucs", "yt-dlp"]:
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
    """Run Basic Pitch to transcribe guitar stem to MIDI using Python API."""
    print(f"[3/4] Transcribing to MIDI with Basic Pitch...")
    
    try:
        # Patch scipy.signal.gaussian for compatibility with scipy >= 1.14
        import scipy.signal
        if not hasattr(scipy.signal, 'gaussian'):
            from scipy.signal.windows import gaussian
            scipy.signal.gaussian = gaussian
        
        from basic_pitch.inference import predict_and_save, verify_output_dir, verify_input_path
        from basic_pitch.predict import Model, build_icassp_2022_model_path, FilenameSuffix
        import pathlib
        
        # Force ONNX backend to avoid TensorFlow compatibility issues
        onnx_model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
        model = Model(onnx_model_path)
        
        # Prepare output directory
        output_dir = pathlib.Path(str(temp_dir))
        verify_output_dir(output_dir)
        
        # Process the audio file
        audio_path_list = [pathlib.Path(str(guitar_stem))]
        for audio_path in audio_path_list:
            verify_input_path(audio_path)
        
        # Run prediction and save
        predict_and_save(
            audio_path_list,
            output_dir,
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=model,
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=127.70,
            minimum_frequency=None,
            maximum_frequency=None,
            multiple_pitch_bends=False,
            melodia_trick=True,
            debug_file=None,
            sonification_samplerate=44100,
            midi_tempo=120,
        )
        
        # The MIDI file is saved with _basic_pitch suffix
        output_midi = temp_dir / f"{guitar_stem.stem}_basic_pitch.mid"
        
        if not output_midi.exists():
            raise FileNotFoundError("MIDI file not generated by Basic Pitch")
        
        return output_midi
    except ImportError as e:
        print(f"Basic Pitch import failed: {e}")
        print("Make sure basic-pitch is installed correctly")
        sys.exit(1)
    except Exception as e:
        print(f"Basic Pitch transcription failed: {e}")
        sys.exit(1)


def convert_to_tab(midi_file: Path, output_dir: Path, tuning: str) -> Path:
    """Convert MIDI to guitar tab using custom simple tab generator."""
    print(f"[4/4] Converting to guitar tab...")
    
    output_tab = output_dir / f"{midi_file.stem}.txt"
    
    try:
        import pretty_midi
        
        # Load MIDI file
        midi_data = pretty_midi.PrettyMIDI(str(midi_file))
        
        # Standard tuning: E2(82.41), A2(110.00), D3(146.83), G3(196.00), B3(246.94), E4(329.63)
        # Open string notes (MIDI numbers): E4=64, B3=59, G3=55, D3=50, A2=45, E2=40
        tuning_map = {
            'standard': [40, 45, 50, 55, 59, 64],  # E2, A2, D3, G3, B3, E4
            'drop_d': [38, 45, 50, 55, 59, 64],   # D2, A2, D3, G3, B3, E4
            'd_standard': [38, 43, 48, 53, 57, 62],  # D2, G2, C3, F3, A3, D4
        }
        
        strings = tuning_map.get(tuning, tuning_map['standard'])
        string_names = ['E', 'A', 'D', 'G', 'B', 'E'] if tuning == 'standard' else [f'S{i+1}' for i in range(6)]
        
        # Collect all notes from all instruments
        all_notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                all_notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity
                })
        
        # Sort by start time
        all_notes.sort(key=lambda x: x['start'])
        
        # Group notes by time (chords)
        chords = []
        current_time = None
        current_chord = []
        
        for note in all_notes:
            if current_time is None or abs(note['start'] - current_time) < 0.05:  # Within 50ms
                if current_time is None:
                    current_time = note['start']
                current_chord.append(note)
            else:
                if current_chord:
                    chords.append(current_chord)
                current_chord = [note]
                current_time = note['start']
        
        if current_chord:
            chords.append(current_chord)
        
        # Generate tab
        tab_lines = ['Guitar Tab (AutoTab)', 'Generated from MIDI', '=' * 40, '']
        
        for i, chord in enumerate(chords):
            # For each note in chord, find best string
            assignments = []
            used_strings = set()
            
            for note in sorted(chord, key=lambda x: -x['pitch']):  # Highest pitch first
                best_string = None
                best_fret = None
                min_distance = float('inf')
                
                # Find the string that gives the closest fret to the note
                for string_idx, open_note in enumerate(strings):
                    if string_idx in used_strings:
                        continue
                    fret = note['pitch'] - open_note
                    if 0 <= fret <= 22:  # Fret range
                        distance = abs(fret - 12)  # Prefer middle of fretboard
                        if distance < min_distance:
                            min_distance = distance
                            best_string = string_idx
                            best_fret = fret
                
                if best_string is not None:
                    assignments.append((best_string, best_fret, note))
                    used_strings.add(best_string)
                else:
                    # Could not assign to a string, skip note
                    pass
            
            # Build tab line for this chord
            if assignments:
                tab_lines.append(f'Measure {i+1}:')
                for string_idx in range(6):
                    line = f"{string_names[string_idx]} |"
                    for string_num, fret, note in assignments:
                        if string_num == string_idx:
                            line += f"{fret:2d}-"
                        else:
                            line += "  -"
                    tab_lines.append(line)
                tab_lines.append('')
        
        # Write tab file
        output_tab.write_text('\n'.join(tab_lines))
        
        return output_tab
        
    except ImportError as e:
        print(f"Tuttut import failed: {e}")
        print("Falling back to simple tab generation...")
        # Create a simple placeholder tab
        output_tab.write_text("Tab generation failed - missing dependencies")
        return output_tab
    except Exception as e:
        print(f"Tab conversion failed: {e}")
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
        tab_file = convert_to_tab(midi_file, Path("."), args.tuning)
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