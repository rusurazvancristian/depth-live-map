"""Model auto-download — ensures all .hef model files exist before pipeline startup.

Downloads missing models from the URLs specified in config.MODEL_REGISTRY,
with progress reporting, retry logic, and atomic file writes via temp+rename.
"""

import logging
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _format_bytes(num_bytes: int) -> str:
    """Format byte count into a human-readable string.

    Args:
        num_bytes: Raw byte count.

    Returns:
        Human-readable string like '12.34 MB' or '956 KB'.
    """
    if num_bytes < 0:
        return "0 B"
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def _print_progress(
    filename: str,
    downloaded: int,
    total: Optional[int],
    start_time: float,
) -> None:
    """Print a progress bar to stdout showing download status.

    Args:
        filename: Name of the file being downloaded.
        downloaded: Number of bytes downloaded so far.
        total: Total expected bytes (None if Content-Length unavailable).
        start_time: Monotonic timestamp when the download started.

    Returns:
        None. Output is written directly to stdout.
    """
    elapsed = time.monotonic() - start_time
    speed_bps = downloaded / elapsed if elapsed > 0.0 else 0.0
    speed_str = f"{speed_bps / (1024 * 1024):.2f} MB/s"

    if total is not None and total > 0:
        pct = min(100.0, downloaded / total * 100.0)
        bar_width = 30
        filled = int(bar_width * pct / 100.0)
        bar = "=" * filled + "-" * (bar_width - filled)
        sys.stdout.write(
            f"\r  {filename}: {bar} {pct:5.1f}% "
            f"[{_format_bytes(downloaded)}/{_format_bytes(total)}] {speed_str}"
        )
    else:
        sys.stdout.write(
            f"\r  {filename}: {_format_bytes(downloaded)} downloaded | {speed_str}"
        )
    sys.stdout.flush()


def _download_file(url: str, dest_path: str, filename: str) -> None:
    """Download a single file from URL to dest_path using atomic temp+rename.

    Writes to a temporary file in the same directory as dest_path, then
    renames atomically to avoid leaving partial downloads on failure.

    Args:
        url: Source URL to download from.
        dest_path: Final filesystem path for the downloaded file.
        filename: Display name for progress output.

    Raises:
        urllib.error.URLError: On network errors (caught by caller for retry).
        OSError: On filesystem errors during write or rename.
    """
    dest_dir = os.path.dirname(dest_path) or "."

    # Create a temp file in the same directory so os.replace() works (same filesystem)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f".{filename}.",
        dir=dest_dir,
    )
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "depth-live-map/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            total_str = response.headers.get("Content-Length")
            total: Optional[int] = int(total_str) if total_str else None

            downloaded = 0
            chunk_size = 256 * 1024  # 256 KB chunks
            start_time = time.monotonic()

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                os.write(fd, chunk)
                downloaded += len(chunk)
                _print_progress(filename, downloaded, total, start_time)

        os.close(fd)
        fd = -1  # Mark as closed so cleanup doesn't double-close

        # Print final newline after progress bar
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Atomic rename into place
        os.replace(tmp_path, dest_path)
        logger.info("Downloaded %s (%s)", filename, _format_bytes(downloaded))

    except BaseException:
        # Close fd if still open
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def download_model(
    filename: str,
    url: str,
    models_dir: str,
    max_retries: int = 3,
) -> None:
    """Download a single model file with retry and exponential backoff.

    Args:
        filename: The .hef filename (e.g. 'yolo26s.hef').
        url: URL to download from.
        models_dir: Local directory to store the downloaded file.
        max_retries: Maximum number of download attempts.

    Raises:
        RuntimeError: If all retry attempts are exhausted.
    """
    dest_path = os.path.join(models_dir, filename)
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Downloading %s (attempt %d/%d) from %s",
                filename, attempt, max_retries, url,
            )
            _download_file(url, dest_path, filename)
            return  # Success — exit immediately
        except (urllib.error.URLError, OSError, IOError) as exc:
            last_error = exc
            logger.warning(
                "Download attempt %d/%d for %s failed: %s",
                attempt, max_retries, filename, exc,
            )
            if attempt < max_retries:
                backoff_secs = 2.0 ** (attempt - 1)  # 1s, 2s, 4s
                logger.info("Retrying in %.1f seconds...", backoff_secs)
                time.sleep(backoff_secs)

    raise RuntimeError(
        f"Failed to download {filename} after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


def ensure_models(models_dir: str, registry: Dict[str, str]) -> None:
    """Ensure all required model files are present, downloading any that are missing.

    Iterates through the model registry and checks if each .hef file exists
    in models_dir. Missing models are downloaded with retry logic.

    Args:
        models_dir: Path to the directory where models should be stored.
            Created automatically if it does not exist.
        registry: Mapping of filename -> download URL for each required model.
            Example: ``{"yolo26s.hef": "https://..."}``

    Raises:
        RuntimeError: If any model cannot be downloaded after all retries.
    """
    os.makedirs(models_dir, exist_ok=True)

    missing: Dict[str, str] = {}
    for filename, url in registry.items():
        filepath = os.path.join(models_dir, filename)
        if os.path.isfile(filepath):
            file_size = os.path.getsize(filepath)
            logger.info(
                "Model %s already exists (%s), skipping download.",
                filename, _format_bytes(file_size),
            )
        else:
            missing[filename] = url

    if not missing:
        logger.info("All %d models are present.", len(registry))
        return

    logger.info(
        "%d of %d models missing, starting downloads...",
        len(missing), len(registry),
    )

    for filename, url in missing.items():
        download_model(filename, url, models_dir, max_retries=3)

    logger.info("All models downloaded successfully to %s", models_dir)
