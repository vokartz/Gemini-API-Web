import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from gemini_webapi.constants import AccountStatus
from gemini_webapi.client import GeminiClient
from gemini_webapi.server.app import create_app
from gemini_webapi.server.config import ServerConfig
from gemini_webapi.types import Candidate, ModelOutput


class FakeSession:
    cookies = {}


class ServerEndpointTests(unittest.TestCase):
    def test_gemini_generate_passes_media_mode_to_client_and_rotator(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                ServerConfig(
                    database_path=Path(tmp) / "app.db",
                    accounts_file=None,
                    switch_on_uses=40,
                    failure_threshold=3,
                    immediate_switch_status_codes=(429, 503),
                    proxy=None,
                    request_timeout=300,
                    auto_refresh=True,
                    auth_url="https://gemini.google.com/",
                    auth_headless=True,
                    api_keys=(),
                    host="127.0.0.1",
                    port=7860,
                )
            )
            calls = []

            async def fake_init(self, *args, **kwargs):
                self.client = FakeSession()
                self.account_status = AccountStatus.AVAILABLE

            async def fake_close(self):
                self.client = None

            async def fake_generate_content(self, prompt, **kwargs):
                calls.append((prompt, kwargs))
                return ModelOutput(
                    metadata=["cid", "rid"],
                    candidates=[Candidate(rcid="rcid", text="ok")],
                )

            with (
                patch.object(GeminiClient, "init", fake_init),
                patch.object(GeminiClient, "close", fake_close),
                patch.object(GeminiClient, "generate_content", fake_generate_content),
                TestClient(app) as client,
            ):
                app.state.store.upsert_account(
                    secure_1psid="psid-one",
                    cookies={"__Secure-1PSID": "psid-one"},
                    name="one",
                )
                response = client.post(
                    "/v1/gemini/generate",
                    json={
                        "prompt": "make image",
                        "model": "gemini-3.5-flash",
                        "mode": "image",
                    },
                )
                logs = app.state.store.list_request_logs(limit=10)

            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["ok"])
            self.assertEqual(calls[0][1]["model"], "gemini-3.5-flash")
            self.assertEqual(calls[0][1]["generation_mode"], "image")
            self.assertTrue(
                any(log.output_type == "image_generation_attempt" for log in logs)
            )

    def test_generate_passes_video_mode_to_client_and_rotator(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                ServerConfig(
                    database_path=Path(tmp) / "app.db",
                    accounts_file=None,
                    switch_on_uses=40,
                    failure_threshold=3,
                    immediate_switch_status_codes=(429, 503),
                    proxy=None,
                    request_timeout=300,
                    auto_refresh=True,
                    auth_url="https://gemini.google.com/",
                    auth_headless=True,
                    api_keys=(),
                    host="127.0.0.1",
                    port=7860,
                )
            )
            calls = []

            async def fake_init(self, *args, **kwargs):
                self.client = FakeSession()
                self.account_status = AccountStatus.AVAILABLE

            async def fake_close(self):
                self.client = None

            async def fake_generate_content(self, prompt, **kwargs):
                calls.append((prompt, kwargs))
                return ModelOutput(
                    metadata=["cid", "rid"],
                    candidates=[Candidate(rcid="rcid", text="ok")],
                )

            with (
                patch.object(GeminiClient, "init", fake_init),
                patch.object(GeminiClient, "close", fake_close),
                patch.object(GeminiClient, "generate_content", fake_generate_content),
                TestClient(app) as client,
            ):
                app.state.store.upsert_account(
                    secure_1psid="psid-one",
                    cookies={"__Secure-1PSID": "psid-one"},
                    name="one",
                )
                response = client.post(
                    "/v1/generate",
                    json={
                        "prompt": "make video",
                        "model": "gemini",
                        "mode": "video",
                    },
                )
                logs = app.state.store.list_request_logs(limit=10)

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["text"], "ok")
            self.assertEqual(calls[0][1]["model"], "gemini-3.1-pro")
            self.assertEqual(calls[0][1]["generation_mode"], "video")
            self.assertTrue(
                any(log.output_type == "video_generation_attempt" for log in logs)
            )


if __name__ == "__main__":
    unittest.main()
