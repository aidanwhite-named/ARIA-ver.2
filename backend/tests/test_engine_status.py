from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.models.schemas import ModelListResponse
from backend.services.ai_engine import _extract_codex_auth_email, find_cli_account_email


def _b64url_json(data: dict) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


class CodexAccountDetectionTests(unittest.TestCase):
    def test_openai_model_list_includes_full_and_mini_gpt_5_4(self):
        self.assertIn("gpt-5.4", ModelListResponse().openai)
        self.assertIn("gpt-5.4-mini", ModelListResponse().openai)

    def test_extract_codex_auth_email_from_jwt_claims(self):
        jwt = ".".join(
            [
                _b64url_json({"alg": "none"}),
                _b64url_json({"https://api.openai.com/profile": {"email": "tester@example.com"}}),
                "signature",
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            auth_path = Path(temp_dir) / "auth.json"
            auth_path.write_text(
                json.dumps({"auth_mode": "chatgpt", "tokens": {"id_token": jwt}}),
                encoding="utf-8",
            )

            self.assertEqual(_extract_codex_auth_email(auth_path), "tester@example.com")

    def test_find_cli_account_email_prefers_codex_auth_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            auth_dir = home / ".codex"
            auth_dir.mkdir(parents=True)
            (auth_dir / "auth.json").write_text(
                json.dumps({"email": "preferred@example.com"}),
                encoding="utf-8",
            )

            with patch("backend.services.ai_engine._get_fixed_home", return_value=home):
                with patch("backend.services.ai_engine._account_cache", {}):
                    self.assertEqual(find_cli_account_email("openai"), "preferred@example.com")
