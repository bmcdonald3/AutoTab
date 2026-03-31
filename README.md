# AutoTab

AI-powered guitar tab transcription from MP3 files. Automatically converts audio to readable guitar tabs using state-of-the-art open-source models.

## Pipeline

1. **Download** (optional): Fetch audio from URL using yt-dlp
2. **Isolate**: Separate guitar stem from other instruments using Meta's Demucs v4 (htdemucs_6s model)
3. **Transcribe**: Convert isolated guitar to MIDI using Spotify's Basic Pitch
4. **Optimize**: Transform MIDI into playable guitar tab using Tuttut (prioritizes minimal fretboard distance)

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Upgrade pip first (important for Python 3.12)
python3 -m pip install --upgrade pip

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: The first time you run Demucs, it will automatically download the model (~500MB). Basic Pitch also downloads models on first use.

## Usage

### Basic Examples

```bash
# Transcribe a local MP3 file
python autotab.py --input path/to/song.mp3

# Transcribe with Drop D tuning
python autotab.py --input song.mp3 --tuning drop_d

# Download and transcribe from YouTube
python autotab.py --url "https://youtube.com/watch?v=..." --tuning standard
```

### Supported Tunings

Tuttut supports various guitar tunings. Common options:

- `standard` - Standard E tuning (E-A-D-G-B-e)
- `drop_d` - Drop D tuning (D-A-D-G-B-e)
- `d_standard` - D Standard (D-G-C-F-A-d)
- `half_step_down` - Half step down (Eb-Ab-Db-Gb-Bb-eb)

Check Tuttut documentation for full list of supported tunings.

## Output

- **Intermediate files**: Stored in `temp/` directory (guitar stems, MIDI files)
- **Final tab**: Saved to `output/` as a `.txt` file

Example output format:
```
e|----0-2-3---|
B|----1-3-5---|
G|------------|
D|------------|
A|------------|
E|------------|
```

## Dependencies

- **demucs** (>=4.0.0) - Audio source separation
- **basic-pitch** (>=0.2.6) - Music transcription
- **tuttut** (>=0.1.0) - MIDI to guitar tab conversion
- **yt-dlp** (>=2023.7.6) - Audio downloading

## Troubleshooting

### Python 3.12 / numpy installation errors
If you're using Python 3.12 and encounter numpy build errors, ensure you have the latest pip:
```bash
python3 -m pip install --upgrade pip
```
The requirements.txt explicitly includes `numpy>=1.24.0` which has pre-built wheels for Python 3.12.

### Demucs model not found
Demucs will auto-download the htdemucs_6s model on first run. If it fails:
```bash
demucs --repo ~/.cache/demucs -n htdemucs_6s -d cpu your_file.mp3
```

### Basic Pitch errors
Ensure the guitar stem is a clean WAV file. Demucs outputs WAV format by default.

### Tuttut tuning options
Use lowercase with underscores: `drop_d`, `d_standard`, etc. See `tuttut --help` for all options.

## Project Structure

```
guitar-tabs/
├── autotab.py          # Main script
├── requirements.txt    # Python dependencies
├── temp/              # Intermediate files (auto-created)
├── output/            # Final tabs (auto-created)
└── README.md          # This file
```

## Limitations

- Transcription accuracy depends on audio quality and guitar complexity
- Demucs v4 isolates 6 stems; guitar stem may contain some bleed from other instruments
- Basic Pitch works best with monophonic (single-note) guitar lines; chords may be less accurate
- Processing can be slow on CPU; GPU acceleration requires additional setup

## License

This tool uses open-source AI models. Refer to each model's license:
- Demucs: MIT
- Basic Pitch: CC BY-NC 4.0
- Tuttut: MIT

## Contributing

Issues and pull requests welcome for improving transcription quality, adding features, or fixing bugs.