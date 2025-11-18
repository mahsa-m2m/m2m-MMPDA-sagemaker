import os
import subprocess
from pathlib import Path
from tqdm import tqdm

def convert_mkv_to_mp4(directory):
    mkv_files = list(Path(directory).rglob('*.mkv'))

    if not mkv_files:
        print("No MKV files found.")
        return

    print(f"Found {len(mkv_files)} MKV files.\n")

    for mkv_file in tqdm(mkv_files, desc="Converting", unit="file"):
        mp4_file = mkv_file.with_suffix('.mp4')

        if mp4_file.exists():
            continue  # Already converted

        tqdm.write(f"→ Converting {mkv_file.name}...")

        cmd = [
            'ffmpeg', '-i', str(mkv_file),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            str(mp4_file)
        ]

        subprocess.run(cmd, capture_output=True)

        tqdm.write(f"✓ Created {mp4_file.name}")

# Usage
convert_mkv_to_mp4('/Users/mahsa/Mine/M2M/Prodigy/video-dataset-org/sample/val/truthful')
