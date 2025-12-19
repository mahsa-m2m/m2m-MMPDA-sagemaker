import os
import subprocess
import argparse
import sys
from pathlib import Path

def split_video(input_path, output_dir, chunk_size_sec):
    """
    Splits video into chunks using FFmpeg stream copying (no re-encoding).
    
    Args:
        input_path (str): Path to source video.
        output_dir (str): Directory to save chunks.
        chunk_size_sec (int): Duration of chunks in seconds.
        
    Returns:
        list: List of generated chunk file paths.
    """
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    filename = Path(input_path).stem  #  'myvideo'
    extension = Path(input_path).suffix # '.mp4'
    
    # the output filename pattern: [original]-chunk[number].[ext]
    output_pattern = os.path.join(output_dir, f"{filename}-chunk%d{extension}")
    
    # FFmpeg Command Breakdown:
    # -i [input]: Input file
    # -c copy: Copy video and audio streams directly (No Re-encoding/Quality Loss)
    # -map 0: Include all streams (video, audio, subtitles, metadata)
    # -f segment: Use the segment muxer
    # -segment_time: Target length of each chunk
    # -reset_timestamps 1: Reset timestamps so each chunk starts at 0s
    command = [
        'ffmpeg',
        '-y',                 # Overwrite output files
        '-i', input_path,
        '-c', 'copy',         # Preserves exact quality/format
        '-map', '0',          # Keeps all tracks (subs/audio/etc)
        '-segment_time', str(chunk_size_sec),
        '-f', 'segment',
        '-reset_timestamps', '1',
        output_pattern
    ]

    print(f"Starting split for {input_path} into {chunk_size_sec}s chunks...")
    
    try:
        # Run ffmpeg command
        result = subprocess.run(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        if result.returncode != 0:
            print(f"FFmpeg Error: {result.stderr}")
            raise Exception("FFmpeg processing failed")
            
        print("Splitting complete.")
        
        # Verify and gather output files
        generated_files = sorted([
            os.path.join(output_dir, f) 
            for f in os.listdir(output_dir) 
            if f.startswith(f"{filename}-chunk") and f.endswith(extension)
        ])
        
        return generated_files

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split video into chunks without re-encoding.")
    parser.add_argument("--input", required=True, help="Input video file path")
    parser.add_argument("--output_dir", default=".", help="Directory to save output chunks")
    parser.add_argument("--seconds", type=int, default=10, help="Chunk size in seconds")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    split_video(args.input, args.output_dir, args.seconds)