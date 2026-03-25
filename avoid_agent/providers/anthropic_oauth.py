"""OAuth flow for Anthropic (Claude Pro/Max subscription)."""

import base64
import hashlib
import json
import os
import secrets
import subprocess
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

CREDENTIALS_PATH = Path.home() / ".avoid-agent" / "anthropic-credentials.json"
_RATE_LIMIT_PATH = Path.home() / ".avoid-agent" / "anthropic-oauth-ratelimit.json"
_DEFAULT_RATE_LIMIT_WAIT = 300  # 5-minute fallback when no Retry-After header

CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 53692
CALLBACK_PATH = "/callback"
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}{CALLBACK_PATH}"
SCOPES = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"


def _generate_pkce() -> tuple[str, str]:
    verifier_bytes = secrets.token_bytes(32)
    verifier = base64.urlsafe_b64encode(verifier_bytes).rstrip(b"=").decode()
    challenge_bytes = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(challenge_bytes).rstrip(b"=").decode()
    return verifier, challenge


def _save_rate_limit(wait_seconds: int) -> None:
    retry_at = int(time.time()) + wait_seconds
    _RATE_LIMIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RATE_LIMIT_PATH.write_text(json.dumps({"retry_at": retry_at}))


def _rate_limit_remaining() -> int:
    """Return seconds remaining on the rate limit, or 0 if not rate limited."""
    if not _RATE_LIMIT_PATH.exists():
        return 0
    try:
        state = json.loads(_RATE_LIMIT_PATH.read_text())
        remaining = int(state.get("retry_at", 0) - time.time())
        return max(0, remaining)
    except Exception:
        return 0


def _format_wait(seconds: int) -> str:
    mins, secs = divmod(seconds, 60)
    return f"{mins}m {secs}s" if mins else f"{secs}s"


def _curl_post_json(url: str, payload: dict) -> dict:
    """POST JSON via system curl to avoid Cloudflare TLS fingerprint blocking."""
    headers_fd, headers_path = tempfile.mkstemp(suffix=".txt")
    os.close(headers_fd)
    try:
        result = subprocess.run(
            [
                "curl", "-sS", "-X", "POST", url,
                "-H", "Content-Type: application/json",
                "-H", "Accept: application/json",
                "-H", "User-Agent: claude-code/1.0.0",
                "-D", headers_path,
                "-w", "\n%{http_code}",
                "--data-binary", json.dumps(payload),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"curl error: {result.stderr.strip()}")

        *body_lines, status_line = result.stdout.split("\n")
        status_code = int(status_line.strip())
        body = "\n".join(body_lines)

        if status_code == 429:
            retry_after = _parse_retry_after(headers_path)
            wait = retry_after or _DEFAULT_RATE_LIMIT_WAIT
            _save_rate_limit(wait)
            raise RuntimeError(
                f"Anthropic OAuth token endpoint is rate limited. "
                f"Wait {_format_wait(wait)} before trying again."
            )

        if status_code >= 400:
            raise RuntimeError(f"Token request failed: {status_code}; body={body}")

        return json.loads(body)
    finally:
        try:
            os.unlink(headers_path)
        except OSError:
            pass


def _parse_retry_after(headers_path: str) -> int | None:
    """Extract Retry-After seconds from a curl header dump file."""
    try:
        with open(headers_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.lower().startswith("retry-after:"):
                    value = line.split(":", 1)[1].strip()
                    return int(value)
    except (OSError, ValueError):
        pass
    return None


def _exchange_code(code: str, state: str, verifier: str) -> dict:
    return _curl_post_json(TOKEN_URL, {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code,
        "state": state,
        "redirect_uri": REDIRECT_URI,
        "code_verifier": verifier,
    })


def _do_refresh(refresh_token: str) -> dict:
    return _curl_post_json(TOKEN_URL, {
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": refresh_token,
    })


def _creds_from_token_response(j: dict) -> dict:
    # expires_in is in seconds; subtract 5-minute buffer
    expires = int(time.time()) + j["expires_in"] - 300
    return {
        "access": j["access_token"],
        "refresh": j["refresh_token"],
        "expires": expires,
    }


def save_credentials(creds: dict) -> None:
    CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_PATH.write_text(json.dumps(creds))


def load_credentials() -> dict | None:
    if not CREDENTIALS_PATH.exists():
        return None
    try:
        return json.loads(CREDENTIALS_PATH.read_text())
    except Exception:
        return None


def refresh_credentials(creds: dict) -> dict:
    j = _do_refresh(creds["refresh"])
    new_creds = _creds_from_token_response(j)
    save_credentials(new_creds)
    return new_creds


def login() -> dict:
    """Run the full OAuth PKCE login flow. Opens a browser and listens on port 53692."""
    verifier, challenge = _generate_pkce()
    # State is set to the verifier (matches the reference implementation)
    state = verifier

    params = {
        "code": "true",
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    url = f"{AUTHORIZE_URL}?{urlencode(params)}"

    code_holder: list[str | None] = [None]
    server_done = threading.Event()

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path != CALLBACK_PATH:
                self.send_response(404)
                self.end_headers()
                return
            qs = parse_qs(parsed.query)
            got_state = qs.get("state", [None])[0]
            code = qs.get("code", [None])[0]
            error = qs.get("error", [None])[0]

            if error:
                self.send_response(400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    f"<h1>Authentication error: {error}</h1>".encode()
                )
                server_done.set()
                return

            if got_state != state or not code:
                self.send_response(400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<h1>Bad request: state mismatch or missing code</h1>")
                server_done.set()
                return

            code_holder[0] = code
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<h1>Anthropic authentication successful.</h1>"
                b"<p>You can close this window and return to the terminal.</p>"
            )
            server_done.set()

        def log_message(self, format, *args):
            pass  # suppress server logs

    server = HTTPServer((CALLBACK_HOST, CALLBACK_PORT), _Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("\nOpening browser for Anthropic authentication...")
    print(f"If your browser does not open automatically, visit:\n{url}\n")
    webbrowser.open(url)

    server_done.wait(timeout=300)
    server.shutdown()

    code = code_holder[0]
    if not code:
        raise RuntimeError("Authentication timed out or was cancelled.")

    j = _exchange_code(code, state, verifier)

    creds = _creds_from_token_response(j)
    save_credentials(creds)
    print("Anthropic authentication successful.\n")
    return creds


def get_valid_credentials() -> dict:
    """Return valid credentials, refreshing or re-authenticating as needed."""
    remaining = _rate_limit_remaining()
    if remaining:
        raise RuntimeError(
            f"Anthropic OAuth token endpoint is rate limited. "
            f"Wait {_format_wait(remaining)} before trying again."
        )
    creds = load_credentials()
    if not creds:
        return login()
    # Refresh if expiring within 60 seconds
    if int(time.time()) >= creds["expires"] - 60:
        try:
            return refresh_credentials(creds)
        except RuntimeError:
            return login()
    return creds
