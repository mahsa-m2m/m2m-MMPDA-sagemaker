
class VideoPreprocessError(Exception):
    """Raised when video frame extraction or normalization fails."""
    pass

class AudioExtractionError(Exception):
    """Raised when audio extraction fails."""
    pass

class S3WriteError(Exception):
    """Raised when uploading results to S3 fails."""
    pass

class InvalidInputError(Exception):
    """Raised when input parameters or files are missing/invalid."""
    pass

def parse_s3_path(s3_path: str) -> tuple:
    parts = s3_path.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]