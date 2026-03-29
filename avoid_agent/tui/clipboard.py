"""Clipboard image capture for the TUI.

Reads image data from the system clipboard and returns it as a base64-encoded
string suitable for sending to vision-capable LLM providers.

Supported platforms:
- macOS: uses ``osascript`` to read clipboard image data via AppleScript.
- Linux (X11): uses ``xclip -selection clipboard -t image/png -o``.
- Linux (Wayland): uses ``wl-paste --type image/png``.

Returns ``None`` if no image is on the clipboard or the platform is unsupported.
"""

from __future__ import annotations

import base64
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class ClipboardImage:
    """A captured clipboard image ready for provider submission."""

    data: str           # base64-encoded image bytes
    media_type: str     # e.g. "image/png" or "image/jpeg"
    size_bytes: int     # raw byte count before base64 encoding


def _capture_macos() -> ClipboardImage | None:
    """Read an image from the macOS clipboard using AppleScript."""
    # AppleScript writes clipboard image data to stdout as raw bytes when piped
    # through ``osascript -e``.  We write to a temp file via shell to avoid
    # binary data issues with subprocess stdout.
    script = (
        'set imgData to the clipboard as «class PNGf»\n'
        'set posixPath to POSIX path of (path to temporary items folder)\n'
        'set tmpFile to posixPath & "avoid_agent_clipboard.png"\n'
        'set fileRef to open for access POSIX file tmpFile with write permission\n'
        'set eof fileRef to 0\n'
        'write imgData to fileRef\n'
        'close access fileRef\n'
        'return tmpFile'
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        tmp_path = result.stdout.decode("utf-8").strip()
        if not tmp_path:
            return None

        with open(tmp_path, "rb") as f:
            raw = f.read()

        # Clean up temp file (best-effort)
        try:
            import os
            os.unlink(tmp_path)
        except OSError:
            pass

        if not raw:
            return None

        return ClipboardImage(
            data=base64.standard_b64encode(raw).decode("ascii"),
            media_type="image/png",
            size_bytes=len(raw),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def _capture_linux_xclip() -> ClipboardImage | None:
    """Read an image from the X11 clipboard using xclip."""
    try:
        result = subprocess.run(
            ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout:
            return None
        raw = result.stdout
        return ClipboardImage(
            data=base64.standard_b64encode(raw).decode("ascii"),
            media_type="image/png",
            size_bytes=len(raw),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def _capture_linux_wl_paste() -> ClipboardImage | None:
    """Read an image from the Wayland clipboard using wl-paste."""
    try:
        result = subprocess.run(
            ["wl-paste", "--type", "image/png"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout:
            return None
        raw = result.stdout
        return ClipboardImage(
            data=base64.standard_b64encode(raw).decode("ascii"),
            media_type="image/png",
            size_bytes=len(raw),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def capture_clipboard_image() -> ClipboardImage | None:
    """Attempt to read an image from the system clipboard.

    Tries platform-appropriate tools in order, returning the first success.
    Returns ``None`` if no image is available or the platform is unsupported.
    """
    if sys.platform == "darwin":
        return _capture_macos()

    if sys.platform.startswith("linux"):
        # Try Wayland first (environment variable is set when running under Wayland)
        import os
        if os.environ.get("WAYLAND_DISPLAY"):
            img = _capture_linux_wl_paste()
            if img is not None:
                return img

        # Fall back to X11 xclip
        return _capture_linux_xclip()

    return None


def format_size(size_bytes: int) -> str:
    """Human-readable file size string."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes // 1024}KB"
    return f"{size_bytes / (1024 * 1024):.1f}MB"
