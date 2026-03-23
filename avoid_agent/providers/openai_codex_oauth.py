"""OAuth flow for OpenAI Codex (ChatGPT Plus/Pro subscription)."""

import base64
import hashlib
import json
import os
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError

CREDENTIALS_PATH = Path.home() / ".avoid-agent" / "openai-codex-credentials.json"

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid profile email offline_access"
JWT_CLAIM_PATH = "https://api.openai.com/auth"


def _generate_pkce() -> tuple[str, str]:
    verifier_bytes = secrets.token_bytes(32)
    verifier = base64.urlsafe_b64encode(verifier_bytes).rstrip(b"=").decode()
    challenge_bytes = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(challenge_bytes).rstrip(b"=").decode()
    return verifier, challenge


def _decode_jwt(token: str) -> dict | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1]
    # Add padding
    payload += "=" * (-len(payload) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return None


def _get_account_id(access_token: str) -> str | None:
    payload = _decode_jwt(access_token)
    if not payload:
        return None
    auth = payload.get(JWT_CLAIM_PATH, {})
    account_id = auth.get("chatgpt_account_id")
    return account_id if isinstance(account_id, str) and account_id else None


def _exchange_code(code: str, verifier: str) -> dict:
    data = urlencode({
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": code,
        "code_verifier": verifier,
        "redirect_uri": REDIRECT_URI,
    }).encode()
    req = Request(TOKEN_URL, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urlopen(req) as resp:
        return json.loads(resp.read())


def _do_refresh(refresh_token: str) -> dict:
    data = urlencode({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLIENT_ID,
    }).encode()
    req = Request(TOKEN_URL, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urlopen(req) as resp:
        return json.loads(resp.read())


def _creds_from_token_response(j: dict) -> dict:
    access = j["access_token"]
    refresh = j["refresh_token"]
    expires = int(time.time()) + j["expires_in"]
    account_id = _get_account_id(access)
    if not account_id:
        raise RuntimeError("Failed to extract account ID from token")
    return {"access": access, "refresh": refresh, "expires": expires, "account_id": account_id}


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
    try:
        j = _do_refresh(creds["refresh"])
        new_creds = _creds_from_token_response(j)
        save_credentials(new_creds)
        return new_creds
    except HTTPError as e:
        raise RuntimeError(f"Token refresh failed: {e.code} {e.reason}") from e


def login() -> dict:
    """Run the full OAuth PKCE login flow. Opens a browser and listens on port 1455."""
    verifier, challenge = _generate_pkce()
    state = secrets.token_hex(16)

    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "avoid-agent",
    }
    url = f"{AUTHORIZE_URL}?{urlencode(params)}"

    code_holder: list[str | None] = [None]
    server_done = threading.Event()

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path != "/auth/callback":
                self.send_response(404)
                self.end_headers()
                return
            qs = parse_qs(parsed.query)
            got_state = qs.get("state", [None])[0]
            code = qs.get("code", [None])[0]
            if got_state != state or not code:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Bad request: state mismatch or missing code")
                return
            code_holder[0] = code
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<h1>Authentication successful.</h1><p>You can close this window and return to the terminal.</p>"
            )
            server_done.set()

        def log_message(self, format, *args):
            pass  # suppress server logs

    server = HTTPServer(("127.0.0.1", 1455), _Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print(f"\nOpening browser for OpenAI Codex authentication...")
    print(f"If your browser does not open automatically, visit:\n{url}\n")
    webbrowser.open(url)

    server_done.wait(timeout=300)
    server.shutdown()

    code = code_holder[0]
    if not code:
        raise RuntimeError("Authentication timed out or was cancelled.")

    try:
        j = _exchange_code(code, verifier)
    except HTTPError as e:
        raise RuntimeError(f"Token exchange failed: {e.code} {e.reason}") from e

    creds = _creds_from_token_response(j)
    save_credentials(creds)
    print(f"Authenticated as account {creds['account_id']}\n")
    return creds


def get_valid_credentials() -> dict:
    """Return valid credentials, refreshing or re-authenticating as needed."""
    creds = load_credentials()
    if not creds:
        return login()
    # Refresh if expiring within 60 seconds
    if int(time.time()) >= creds["expires"] - 60:
        try:
            return refresh_credentials(creds)
        except RuntimeError:
            # Refresh failed — re-authenticate
            return login()
    return creds
