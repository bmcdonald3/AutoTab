#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from shutil import copy2

def setup_directories():
    temp_dir = Path("temp")
    output_dir = Path("output")
    temp_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    cache_dir = Path.home() / ".cache" / "demucs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(cache_dir, 0o755)
    return temp_dir, output_dir

def check_dependencies():
    missing = []
    for cmd in ["demucs", "yt-dlp", "ffmpeg"]:
        try:
            subprocess.run([cmd, "-version" if cmd == "ffmpeg" else "--version"], capture_output=True, check=False)
        except FileNotFoundError:
            missing.append(cmd)
    if missing:
        print(f"Error: Missing dependencies: {', '.join(missing)}")
        sys.exit(1)

def download_audio(url: str, temp_dir: Path) -> Path:
    print(f"[1/4] Downloading audio...")
    output_path = temp_dir / "current_download.mp3"
    if output_path.exists():
        output_path.unlink()
    try:
        subprocess.run([
            "yt-dlp", "-x", "--audio-format", "mp3", "--no-playlist",
            "-o", str(temp_dir / "current_download.%(ext)s"), url
        ], check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e.stderr.decode()}")
        sys.exit(1)

def isolate_guitar(input_file: Path, temp_dir: Path) -> Path:
    print(f"[2/4] Isolating guitar stem...")
    try:
        subprocess.run(["demucs", "-n", "htdemucs", "-d", "cpu", str(input_file)], check=True, capture_output=True)
        separated_dir = Path("separated") / "htdemucs" / input_file.stem
        guitar_stem = separated_dir / "guitar.wav"
        if not guitar_stem.exists():
            guitar_stem = separated_dir / "other.wav"
        dest = temp_dir / "guitar_stem.wav"
        copy2(guitar_stem, dest)
        return dest
    except Exception as e:
        print(f"Demucs isolation failed: {e}")
        sys.exit(1)

def transcribe_to_midi(guitar_stem: Path, temp_dir: Path) -> Path:
    print(f"[3/4] Transcribing to MIDI...")
    try:
        import scipy.signal
        if not hasattr(scipy.signal, 'gaussian'):
            from scipy.signal.windows import gaussian
            scipy.signal.gaussian = gaussian
        from basic_pitch.inference import predict_and_save
        from basic_pitch.predict import Model, build_icassp_2022_model_path, FilenameSuffix
        
        model = Model(build_icassp_2022_model_path(FilenameSuffix.onnx))
        
        # Updated with all required positional arguments for newer basic-pitch versions
        predict_and_save(
            [Path(guitar_stem)],
            Path(temp_dir),
            True,  # save_midi
            False, # sonify_midi
            False, # save_model_outputs
            False, # save_notes
            model_or_model_path=model
        )
        return temp_dir / f"{guitar_stem.stem}_basic_pitch.mid"
    except Exception as e:
        print(f"Transcription failed: {e}")
        sys.exit(1)

def convert_to_tab(midi_file: Path, output_name: str, tuning: str) -> Path:
    print(f"[4/4] Generating continuous tab...")
    try:
        import pretty_midi
        midi_data = pretty_midi.PrettyMIDI(str(midi_file))
        tuning_map = {
            'standard': [40, 45, 50, 55, 59, 64],
            'eb_standard': [39, 44, 49, 54, 58, 63],
            'drop_d': [38, 45, 50, 55, 59, 64],
        }
        strings = tuning_map.get(tuning, tuning_map['standard'])
        all_notes = []
        for inst in midi_data.instruments:
            for note in inst.notes:
                all_notes.append({'pitch': note.pitch, 'start': note.start})
        all_notes.sort(key=lambda x: x['start'])

        columns = []
        curr_time, curr_chord = None, []
        for n in all_notes:
            if curr_time is None or abs(n['start'] - curr_time) < 0.05:
                if curr_time is None: curr_time = n['start']
                curr_chord.append(n)
            else:
                columns.append(curr_chord)
                curr_chord, curr_time = [n], n['start']
        if curr_chord: columns.append(curr_chord)

        formatted = []
        for chord in columns:
            used_strings = set()
            col = ["-"] * 6
            for n in sorted(chord, key=lambda x: -x['pitch']):
                for s_idx, open_note in enumerate(strings):
                    if s_idx in used_strings: continue
                    fret = n['pitch'] - open_note
                    if 0 <= fret <= 22:
                        col[5 - s_idx] = str(fret)
                        used_strings.add(s_idx)
                        break
            formatted.append(col)

        output_path = Path(output_name)
        with open(output_path, 'w') as f:
            step = 10
            names = ['e','B','G','D','A','E']
            for i in range(0, len(formatted), step):
                chunk = formatted[i : i + step]
                for s_idx in range(6):
                    line = f"{names[s_idx]}|"
                    for c in chunk:
                        line += f"-{c[s_idx]:- <9}"
                    f.write(line + "|\n")
                f.write("\n")
        return output_path
    except Exception as e:
        print(f"Tab conversion failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--input")
    parser.add_argument("--tuning", default="standard")
    parser.add_argument("-o", "--output", help="Output filename")
    args = parser.parse_args()

    temp_dir, _ = setup_directories()
    check_dependencies()
    
    input_file = download_audio(args.url, temp_dir) if args.url else Path(args.input)
    guitar_stem = isolate_guitar(input_file, temp_dir)
    midi_file = transcribe_to_midi(guitar_stem, temp_dir)
    
    out_name = args.output if args.output else f"{input_file.stem}_tab.txt"
    tab_file = convert_to_tab(midi_file, out_name, args.tuning)
    
    print(f"Processed into {tab_file}")

if __name__ == "__main__":
    main()