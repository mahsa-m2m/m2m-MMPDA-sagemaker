from urllib.parse import urlparse

class VideoChunkingError(Exception):
    """Raised when video processing fails (corrupt file, ffmpeg error)"""
    pass

class S3WriteError(Exception):
    """Raised when uploading to S3 fails"""
    pass

class InvalidInputError(Exception):
    """Raised when the input event/environment variables are wrong"""
    pass

def parse_s3_path(s3_path: str) -> tuple:
    """
    Parses s3://bucket/key into (bucket, key)
    """
    if not s3_path.startswith("s3://"):
        raise InvalidInputError(f"Invalid S3 URI: {s3_path}")
        
    parsed = urlparse(s3_path)
    return parsed.netloc, parsed.path.lstrip('/')