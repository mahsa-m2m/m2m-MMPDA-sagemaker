import os
import subprocess
import argparse
import json
import shutil
import math
from pathlib import Path

def get_video_duration(input_path):
    """Returns the exact duration of the video in seconds."""
    cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        input_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(result.stdout.strip())
    except (ValueError, IndexError):
        # Fallback
        return 0.0

def split_video_precise(input_path, output_dir, chunk_size_sec):
    """
    Splits video with EXACT timing by re-encoding.
    Merges the last chunk if it is too short.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    filename = Path(input_path).stem
    extension = ".mp4" 
    
    total_duration = get_video_duration(input_path)
    if total_duration == 0:
        raise Exception("Could not determine video duration.")

    # --- 1. Calculate Split Points ---
    split_points = []
    current_time = chunk_size_sec
    
    while current_time < total_duration:
        split_points.append(current_time)
        current_time += chunk_size_sec
        
    # --- 2. Merge Last Chunk ---
    if split_points:
        last_split_point = split_points[-1]
        remainder = total_duration - last_split_point
        
        # If the remainder is shorter than the chunk size, remove the last cut.
        # Example: Duration 19s, Target 5s. 
        # Result chunks: 0-5 (5s), 5-10 (5s), 10-19 (9s).
        if remainder < chunk_size_sec:
            split_points.pop()

    output_pattern = os.path.join(output_dir, f"{filename}-chunk%d{extension}")
    
    # --- 3. Construct FFmpeg Command ---
    cmd = [
        'ffmpeg',
        '-y',
        '-i', input_path,
        
        # VIDEO ENCODING SETTINGS
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '23',
        
        '-c:a', 'aac', 
        
        '-f', 'segment',
        '-reset_timestamps', '1',
    ]

    # Handle Split Points
    if split_points:
        times_str = ",".join(map(str, split_points))
        cmd.extend(['-segment_times', times_str])
        
        # Force Keyframes at the exact cut points
        cmd.extend(['-force_key_frames', times_str])
    
    cmd.append(output_pattern)

    print(f"Processing {input_path} (Duration: {total_duration}s)")
    print(f"Calculated Cuts: {split_points}")
    
    try:
        subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=True
        )
        
        # Return list of created files
        generated_files = sorted([
            os.path.join(output_dir, f) 
            for f in os.listdir(output_dir) 
            if f.startswith(f"{filename}-chunk") and f.endswith(extension)
        ])
        return generated_files

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error Output: {e.stderr}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--seconds", type=int, default=5)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    split_video_precise(args.input, args.output_dir, args.seconds)