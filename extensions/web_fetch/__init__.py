"""Extension tool for fetching and reading web pages as plain text."""

import re
import urllib.error
import urllib.request
from html.parser import HTMLParser

from typing_extensions import Annotated

from avoid_agent.agent.tools import ToolRunResult, tool

_CONTENT_LIMIT = 8000
_USER_AGENT = "avoid-agent/webfetch (urllib; +https://github.com/a-m-brewer/avoid-agent)"

# HTML void elements — they have no closing tag, so we must never increment
# skip_depth for them (there will be no matching end-tag to decrement).
_VOID_ELEMENTS = frozenset(
    {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }
)

# Tags whose entire subtree (tag + all descendants + closing tag) should be
# discarded.  Keep script/style OUT of here — html.parser handles them in
# native CDATA mode (no handle_starttag/handle_endtag events for their inner
# content), so skip_depth accounting never needs to deal with them.
_SKIP_TAGS = frozenset(
    {
        "head",
        "noscript",
        "svg",
        "canvas",
        "template",
        "iframe",
        "nav",
        "footer",
        "header",
    }
)

# Block-level tags — emit newlines around their content.
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "dt",
        "dd",
        "tr",
        "blockquote",
        "pre",
        "br",
    }
)


class _TextExtractor(HTMLParser):
    """Minimal HTML -> plain-text converter using only stdlib html.parser.

    Design notes
    ------------
    * script/style are handled by html.parser's native CDATA mode — their raw
      content is never delivered to handle_data, and their end-tags DO fire
      handle_endtag.  We do not add them to _SKIP_TAGS.
    * Void elements (meta, img, br, input, …) never produce a handle_endtag
      event, so we must not increment _skip_depth for them.
    * All other tags inside a skipped subtree increment/decrement _skip_depth
      symmetrically via their start/end tag events.
    """

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Void elements never produce an end-tag event; skip them entirely.
        if tag in _VOID_ELEMENTS:
            return

        if self._skip_depth > 0:
            self._skip_depth += 1
            return

        if tag in _SKIP_TAGS:
            self._skip_depth = 1
            return

        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_ELEMENTS:
            return

        if self._skip_depth > 0:
            self._skip_depth -= 1
            return

        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse runs of blank lines to at most two newlines
        text = re.sub(r"\n{3,}", "\n\n", raw)
        # Strip trailing whitespace from each line
        lines = [line.rstrip() for line in text.splitlines()]
        return "\n".join(lines).strip()


def _fetch_url(url: str, timeout: int) -> tuple[str, str]:
    """Return (content_type, body_text). Raises urllib.error.URLError on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        content_type: str = resp.headers.get_content_type() or ""
        charset: str = resp.headers.get_content_charset("utf-8") or "utf-8"
        raw_bytes: bytes = resp.read()

    try:
        body = raw_bytes.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        body = raw_bytes.decode("utf-8", errors="replace")

    return content_type, body


@tool
def web_fetch(
    url: Annotated[str, "The full URL of the page to fetch (must start with http:// or https://)."],
    max_chars: Annotated[int, "Maximum number of characters to return (default 8000, max 32000)."] = _CONTENT_LIMIT,
) -> ToolRunResult:
    """Fetch a web page and return its readable text content.

    Strips HTML tags, scripts, styles and navigation chrome, then returns
    clean paragraph text suitable for reading or quoting.  Plain-text and
    JSON responses are returned as-is (up to max_chars).  Binary content
    (images, PDFs, etc.) is rejected with an informative message.
    """
    if not url.startswith(("http://", "https://")):
        return ToolRunResult(content="Error: URL must start with http:// or https://")

    max_chars = max(256, min(max_chars, 32_000))
    timeout = 15  # seconds

    try:
        content_type, body = _fetch_url(url, timeout)
    except urllib.error.HTTPError as exc:
        return ToolRunResult(content=f"HTTP error {exc.code} fetching {url}: {exc.reason}")
    except urllib.error.URLError as exc:
        return ToolRunResult(content=f"Failed to fetch {url}: {exc.reason}")
    except TimeoutError:
        return ToolRunResult(content=f"Timed out after {timeout}s fetching {url}")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return ToolRunResult(content=f"Unexpected error fetching {url}: {exc}")

    # Reject binary content types early
    _binary_prefixes = ("image/", "video/", "audio/", "application/pdf", "application/zip")
    if any(content_type.startswith(p) for p in _binary_prefixes):
        return ToolRunResult(
            content=f"Cannot display binary content (type: {content_type}) from {url}"
        )

    # Parse HTML -> plain text; leave JSON / plain-text as-is
    if "html" in content_type:
        parser = _TextExtractor()
        parser.feed(body)
        text = parser.get_text()
    else:
        text = body.strip()

    if not text:
        return ToolRunResult(content=f"Page fetched but no readable text found at {url}")

    truncated = len(text) > max_chars
    output = text[:max_chars]
    if truncated:
        output += (
            f"\n\n... [truncated - {len(text) - max_chars} more characters available;"
            f" request a higher max_chars to see more]"
        )

    return ToolRunResult(
        content=output,
        details={
            "proof": {
                "kind": "web_fetch",
                "url": url,
                "content_type": content_type,
                "total_chars": len(text),
                "returned_chars": len(output),
                "truncated": truncated,
            }
        },
    )
