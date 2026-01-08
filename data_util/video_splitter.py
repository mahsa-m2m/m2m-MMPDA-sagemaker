import os
import subprocess
import argparse
import json
import shutil
import math
from pathlib import Path
import boto3
from urllib.parse import urlparse

class VideoChunkingError(Exception):
    """Raised when video chunking fails"""
    pass

class S3WriteError(Exception):
    """Raised when S3 write operations fail"""
    pass

s3_client = boto3.client('s3')

def parse_s3_uri(s3_uri):
    parsed = urlparse(s3_uri)
    return parsed.netloc, parsed.path.lstrip('/')

def validate_video_streams(input_path):
    """
    Validates that the video has processable video and audio streams.
    Returns True if valid, raises VideoChunkingError if corrupted.
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'stream=codec_type,codec_name,duration,pix_fmt:format=duration',
        '-of', 'json',
        input_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        probe_data = json.loads(result.stdout)
        
        if 'streams' not in probe_data or len(probe_data['streams']) == 0:
            raise VideoChunkingError(f"No streams found in video file: {os.path.basename(input_path)}")
        
        video_stream = None
        audio_stream = None
        
        for stream in probe_data['streams']:
            if stream.get('codec_type') == 'video':
                video_stream = stream
            elif stream.get('codec_type') == 'audio':
                audio_stream = stream
        
        # Check video stream exists
        if not video_stream:
            raise VideoChunkingError(f"No video stream found in file: {os.path.basename(input_path)}")
        
        # Check if video stream has pixel format
        if video_stream.get('pix_fmt') in [None, 'none', '']:
            raise VideoChunkingError(f"Video stream has no pixel format - file is corrupted: {os.path.basename(input_path)}")
        
        # Check duration - try stream duration first, then fall back to format duration
        video_duration = video_stream.get('duration', '0')
        format_duration = probe_data.get('format', {}).get('duration', '0')
        
        try:
            stream_dur = float(video_duration)
            format_dur = float(format_duration)
            
            # Use format duration if stream duration is zero
            actual_duration = stream_dur if stream_dur > 0 else format_dur
            
            if actual_duration == 0:
                raise VideoChunkingError(f"Video has zero duration - file is corrupted: {os.path.basename(input_path)}")
            
            # Verify the file has actual video frames by checking for codec_name
            if not video_stream.get('codec_name'):
                raise VideoChunkingError(f"Video stream has no codec - file is corrupted: {os.path.basename(input_path)}")
                
        except (ValueError, TypeError):
            raise VideoChunkingError(f"Video has invalid duration - file is corrupted: {os.path.basename(input_path)}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        raise VideoChunkingError(f"Failed to probe video file - file may be corrupted: {os.path.basename(input_path)}")
    except json.JSONDecodeError:
        raise VideoChunkingError(f"Failed to parse video metadata - file is corrupted: {os.path.basename(input_path)}")
def check_has_audio(input_path):
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a',
        '-show_entries', 'stream=codec_type',
        '-of', 'csv=p=0',
        input_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return len(result.stdout.strip()) > 0
    except Exception:
        return False

def get_video_duration(input_path):
    """Returns the exact duration of the video in seconds."""
    cmd = [
        'ffprobe', '-v', 'error', 
        '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        input_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(result.stdout.strip())
    except (ValueError, IndexError):
        return 0.0

def split_video_precise(input_path, output_dir, chunk_size_sec):
    """
    Splits video with EXACT timing by re-encoding.
    Merges the last chunk if it is too short.
    """
    if not os.path.exists(input_path):
        raise VideoChunkingError(f"Input file not found: {input_path}")

    # Validate video before processing
    validate_video_streams(input_path)

    filename = Path(input_path).stem
    extension = ".mp4" 
    
    total_duration = get_video_duration(input_path)
    if total_duration == 0:
        raise VideoChunkingError("Could not determine video duration")

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
        
        if remainder < chunk_size_sec:
            split_points.pop()

    # CASE 1: The video is shorter than 1 chunk (or equals it). 
    if not split_points and total_duration <= chunk_size_sec:
        print(f"Video is shorter than chunk size ({total_duration}s < {chunk_size_sec}s). Processing as single file.")
        
        output_file = os.path.join(output_dir, f"{filename}-chunk00{extension}")
        
        cmd = [
            'ffmpeg', '-y',
            '-analyzeduration', '10M',
            '-probesize', '10M',
            '-i', input_path,
            '-pix_fmt', 'yuv420p',
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-c:a', 'aac',
            output_file
        ]

    # CASE 2: The video needs splitting
    else:
        output_pattern = os.path.join(output_dir, f"{filename}-chunk%d{extension}")
        
        cmd = [
            'ffmpeg',
            '-y',
            '-analyzeduration', '10M',
            '-probesize', '10M',
            '-i', input_path,
            '-pix_fmt', 'yuv420p',
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-crf', '23',
            '-c:a', 'aac',
            '-f', 'segment',
            '-reset_timestamps', '1',
        ]

        if split_points:
            times_str = ",".join(map(str, split_points))
            cmd.extend(['-segment_times', times_str])
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
        
        generated_files = sorted([
            os.path.join(output_dir, f) 
            for f in os.listdir(output_dir) 
            if f.startswith(f"{filename}-chunk") and f.endswith(extension)
        ])
        
        if not generated_files:
            raise VideoChunkingError("No output chunks were generated")
        
        return generated_files

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8') if isinstance(e.stderr, bytes) else str(e.stderr)
        raise VideoChunkingError(f"FFmpeg failed to process video: {error_msg[:200]}")

def lambda_handler(event, context):
    """
    AWS Lambda Entry Point
    event: The JSON input defined in Execution Contract > Input
    """
    session_id = event.get('sessionId')
    file_type = event.get('fileType', 'video')
    s3_input = event.get('s3Input')
    chunk_size = event.get('chunkSizeSeconds', 10)
    
    input_bucket, input_key = parse_s3_uri(s3_input)
    local_input_filename = os.path.basename(input_key)
    local_input_path = os.path.join('/tmp', local_input_filename)
    output_dir = os.path.join('/tmp', 'chunks')
    
    # Clean up /tmp
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    try:
        # Download file from S3
        try:
            print(f"Downloading {s3_input}...")
            s3_client.download_file(input_bucket, input_key, local_input_path)
        except Exception as e:
            raise VideoChunkingError(f"Failed to download input from S3: {str(e)}")

        # Run the split logic
        print("Splitting video...")
        local_chunk_files = split_video_precise(local_input_path, output_dir, chunk_size)

        # Check for audio
        has_audio = check_has_audio(local_input_path)

        # Upload chunks back to S3
        uploaded_uris = []
        try:
            for local_file in local_chunk_files:
                fname = os.path.basename(local_file)
                s3_output_key = f"results/{session_id}/video/{fname}"
                s3_client.upload_file(local_file, input_bucket, s3_output_key)
                uploaded_uris.append(f"s3://{input_bucket}/{s3_output_key}")
                
            if not uploaded_uris:
                raise S3WriteError("No chunks were uploaded to S3")

        except Exception as e:
            # Wrap any S3 upload errors in S3WriteError
            if isinstance(e, S3WriteError):
                raise
            raise S3WriteError(f"Failed to upload chunks to S3: {str(e)}")

        # Success response
        return {
            "sessionId": session_id,
            "fileType": file_type,
            "videoChunks": uploaded_uris,
            "hasAudio": has_audio,
            "status": "success",
            "error": None
        }

    except VideoChunkingError as e:
        # Video processing failed - return failed status
        print(f"VideoChunkingError: {e}")
        return {
            "sessionId": session_id,
            "fileType": file_type,
            "videoChunks": [],
            "hasAudio": False,
            "status": "failed",
            "error": str(e)
        }
    
    except S3WriteError as e:
        # S3 write failed - return failed status
        print(f"S3WriteError: {e}")
        return {
            "sessionId": session_id,
            "fileType": file_type,
            "videoChunks": [],
            "hasAudio": False,
            "status": "failed",
            "error": str(e)
        }
    
    except Exception as e:
        # Unexpected errors - return failed status
        print(f"Unexpected error: {e}")
        return {
            "sessionId": session_id,
            "fileType": file_type,
            "videoChunks": [],
            "hasAudio": False,
            "status": "failed",
            "error": f"Unexpected error: {str(e)}"
        }