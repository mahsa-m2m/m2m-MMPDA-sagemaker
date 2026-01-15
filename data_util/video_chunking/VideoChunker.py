import os
import json
import subprocess
import logging
from pathlib import Path
from typing import List, Optional

from utils import VideoChunkingError

logger = logging.getLogger(__name__)

class VideoChunker:
    """
    Handles all FFmpeg operations: validation, metadata extraction, and splitting.
    """

    def __init__(self):
        logger.info("Initializing VideoChunker")
        # Verify ffmpeg is installed/accessible
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("FFmpeg check passed")
        except FileNotFoundError:
            logger.critical("FFmpeg is NOT installed on this environment!")
            raise RuntimeError("FFmpeg not found")

    def _get_probe_data(self, input_path: str) -> dict:
        """Helper to run ffprobe and return JSON"""
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'stream=codec_type,codec_name,duration,pix_fmt:format=duration',
            '-of', 'json',
            input_path
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error(f"Probe failed for {input_path}: {e}")
            raise VideoChunkingError(f"Failed to probe video: {str(e)}")

    def validate_video(self, input_path: str) -> bool:
        """
        Validates video integrity. Raises VideoChunkingError if corrupted.
        """
        logger.info(f"Validating video file: {os.path.basename(input_path)}")
        
        if not os.path.exists(input_path):
            raise VideoChunkingError(f"Input file not found: {input_path}")

        probe_data = self._get_probe_data(input_path)
        
        if 'streams' not in probe_data or not probe_data['streams']:
            raise VideoChunkingError("No streams found in video file")

        video_stream = next((s for s in probe_data['streams'] if s.get('codec_type') == 'video'), None)
        
        if not video_stream:
            raise VideoChunkingError("No video stream found")
        
        if not video_stream.get('codec_name'):
            raise VideoChunkingError("Video stream has no codec defined")

        # Check for valid duration
        duration = self.get_duration(input_path)
        if duration <= 0:
            raise VideoChunkingError("Video has zero or invalid duration")
            
        logger.info("Video validation passed")
        return True

    def get_duration(self, input_path: str) -> float:
        """Returns duration in seconds"""
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
            logger.warning(f"Could not extract duration for {input_path}")
            return 0.0

    def check_has_audio(self, input_path: str) -> bool:
        """Checks if the file contains an audio stream"""
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a',
            '-show_entries', 'stream=codec_type',
            '-of', 'csv=p=0',
            input_path
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            has_audio = len(result.stdout.strip()) > 0
            logger.debug(f"Audio check: {'Found' if has_audio else 'None'}")
            return has_audio
        except Exception:
            return False

    def split_video_precise(self, input_path: str, output_dir: str, chunk_size_sec: int) -> List[str]:
        """
        Splits video into chunks using FFmpeg.
        """
        logger.info(f"Starting split process for {os.path.basename(input_path)}")
        
        # Validate video
        self.validate_video(input_path)
        
        filename = Path(input_path).stem
        extension = ".mp4"
        total_duration = self.get_duration(input_path)
        
        # Calculate Split Points
        split_points = []
        current_time = chunk_size_sec
        while current_time < total_duration:
            split_points.append(current_time)
            current_time += chunk_size_sec
            
        # Merge last chunk if too small
        if split_points:
            remainder = total_duration - split_points[-1]
            if remainder < chunk_size_sec:
                logger.debug(f"Last chunk remainder ({remainder}s) is small, merging with previous")
                split_points.pop()

        # Construct FFmpeg Command
        output_pattern = os.path.join(output_dir, f"{filename}-chunk%d{extension}")
        
        # Base command
        cmd = [
            'ffmpeg',
            '-y',
            '-analyzeduration', '10M',
            '-probesize', '10M',
            '-i', input_path,
            '-c', 'copy',             
            '-map', '0',              
            '-f', 'segment',          
            '-reset_timestamps', '1', 
        ]

        if not split_points and total_duration <= chunk_size_sec:
            # ---- Single file (No splitting needed)
            logger.info("Video shorter than chunk size, processing as single file")
            output_file = os.path.join(output_dir, f"{filename}-chunk00{extension}")
            cmd.append(output_file)
        else:
            # ---- Multiple segments
            logger.info(f"Splitting into approx {len(split_points) + 1} chunks")
            cmd.extend(['-f', 'segment', '-reset_timestamps', '1'])
            
            if split_points:
                times_str = ",".join(map(str, split_points))
                cmd.extend(['-segment_times', times_str])
                # cmd.extend(['-force_key_frames', times_str])
            
            cmd.append(output_pattern)

        # Execute
        logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            # Log the stderr from ffmpeg
            error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
            logger.error(f"FFmpeg Error Output: {error_msg[-500:]}") # Log last 500 chars
            raise VideoChunkingError(f"FFmpeg failed: {error_msg[:100]}")

        # Verify Output
        generated_files = sorted([
            os.path.join(output_dir, f) 
            for f in os.listdir(output_dir) 
            if f.startswith(filename) and f.endswith(extension)
        ])
        
        if not generated_files:
            raise VideoChunkingError("FFmpeg ran but no files were generated")
            
        return generated_files
