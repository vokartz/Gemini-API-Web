import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from gemini_webapi.constants import AccountStatus
from gemini_webapi.client import GeminiClient
from gemini_webapi.server.app import create_app
from gemini_webapi.server.config import ServerConfig
from gemini_webapi.types.image import GeneratedImage
from gemini_webapi.types.video import GeneratedVideo
from gemini_webapi.types import Candidate, ModelOutput


class FakeSession:
    cookies = {}


class FakeNoVncResponse:
    content = b"<html>noVNC</html>"
    status_code = 200
    headers = {"content-type": "text/html"}


class FakeNoVncClient:
    calls = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return FakeNoVncResponse()


class FakeMediaResponse:
    headers = {"content-type": "image/png"}
    content = b"image-bytes"

    def raise_for_status(self):
        return None


class FakeMediaHTTPClient:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url):
        return FakeMediaResponse()


class ServerEndpointTests(unittest.TestCase):
    def _config(self, tmp: str) -> ServerConfig:
        return ServerConfig(
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

    def test_parse_candidate_falls_back_to_nested_video_urls(self):
        client = GeminiClient()
        candidate_data = [
            None,
            ["您的视频准备好了！"],
            None,
            {
                "status": "ready",
                "placeholder": "http://googleusercontent.com/video_gen_chip/0",
                "thumb": "https://lh3.googleusercontent.com/preview/video_generation_content/thumb.jpg",
                "video": "https://rr1---sn.googlevideo.com/videoplayback?id=video-test",
            },
        ]

        _, _, _, _, videos, _ = client._parse_candidate(
            candidate_data, "cid", "rid", "rcid"
        )

        self.assertEqual(len(videos), 1)
        self.assertIsInstance(videos[0], GeneratedVideo)
        self.assertEqual(
            videos[0].url,
            "https://rr1---sn.googlevideo.com/videoplayback?id=video-test",
        )
        self.assertNotIn("video_gen_chip", videos[0].url)

    def test_parse_candidate_falls_back_to_google_download_video_urls(self):
        client = GeminiClient()
        candidate_data = [
            None,
            ["Your video is ready!\nhttp://googleusercontent.com/generated_video_content/0"],
            None,
            None,
            None,
            None,
            None,
            None,
            [2],
            None,
            None,
            None,
            [
                {
                    "60": [
                        [
                            [
                                [
                                    [
                                        None,
                                        None,
                                        "video.mp4",
                                        None,
                                        None,
                                        None,
                                        None,
                                        [
                                            "",
                                            "https://contribution.usercontent.google.com/download?filename=video.mp4&opi=103135050",
                                        ],
                                    ]
                                ]
                            ]
                        ]
                    ]
                }
            ],
        ]

        _, _, _, _, videos, _ = client._parse_candidate(
            candidate_data, "cid", "rid", "rcid"
        )

        self.assertEqual(len(videos), 1)
        self.assertIsInstance(videos[0], GeneratedVideo)
        self.assertEqual(
            videos[0].url,
            "https://contribution.usercontent.google.com/download?filename=video.mp4&opi=103135050",
        )

    def test_accounts_endpoint_lists_account_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(self._config(tmp))
            with TestClient(app) as client:
                account = app.state.store.upsert_account(
                    secure_1psid="psid-one",
                    cookies={"__Secure-1PSID": "psid-one"},
                    name="one",
                )
                app.state.store.set_state("current_account_id", str(account.id))

                response = client.get("/v1/accounts")

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["ok"])
            self.assertEqual(data["current_account_id"], account.id)
            self.assertEqual(len(data["accounts"]), 1)
            self.assertEqual(data["accounts"][0]["name"], "one")

    def test_novnc_proxy_uses_same_origin_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(self._config(tmp))
            FakeNoVncClient.calls = []
            with (
                patch("httpx.AsyncClient", FakeNoVncClient),
                TestClient(app) as client,
            ):
                response = client.get(
                    "/novnc/vnc.html?autoconnect=true&resize=scale&path=websockify"
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "text/html; charset=utf-8")
            self.assertEqual(len(FakeNoVncClient.calls), 1)
            method, url, kwargs = FakeNoVncClient.calls[0]
            self.assertEqual(method, "GET")
            self.assertEqual(
                url,
                "http://127.0.0.1:6080/vnc.html?autoconnect=true&resize=scale&path=websockify",
            )
            self.assertIn("headers", kwargs)

    def test_auth_session_returns_vnc_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(self._config(tmp))

            async def fake_start_session(self):
                return {
                    "id": "auth-1",
                    "vnc_url": "/novnc/vnc.html?autoconnect=true&resize=scale&path=websockify",
                }

            with (
                patch(
                    "gemini_webapi.server.auth_browser.AuthBrowserManager.start_session",
                    fake_start_session,
                ),
                TestClient(app) as client,
            ):
                response = client.post("/v1/auth/session", json={})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json()["vnc_url"],
                "/novnc/vnc.html?autoconnect=true&resize=scale&path=websockify",
            )

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
                    candidates=[
                        Candidate(
                            rcid="rcid",
                            text="ok",
                            generated_images=[
                                GeneratedImage(
                                    url="https://example.invalid/generated.png",
                                    title="generated",
                                )
                            ],
                        )
                    ],
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
            self.assertEqual(response.json()["media_count"], 1)
            self.assertEqual(calls[0][1]["model"], "gemini-3.5-flash")
            self.assertEqual(calls[0][1]["generation_mode"], "image")
            self.assertTrue(
                any(log.output_type == "image_generation_attempt" for log in logs)
            )
            self.assertTrue(
                any(
                    log.output_type == "gemini_image" and log.media_count == 1
                    for log in logs
                )
            )

    def test_system_settings_api_keys_are_dynamic_and_masked(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(self._config(tmp))
            with TestClient(app) as client:
                response = client.patch(
                    "/v1/system-settings",
                    json={
                        "api_keys": ["sk-local-secret"],
                        "object_storage": {
                            "enabled": True,
                            "endpoint": "https://s3.example.test",
                            "region": "auto",
                            "bucket": "media",
                            "access_key_id": "access",
                            "secret_access_key": "secret-value",
                            "prefix": "gemini-web",
                            "public_url": "https://cdn.example.test/media",
                            "force_path_style": True,
                        },
                    },
                )
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["settings"]["api_keys"][0]["masked"], "sk-l...cret")
                fingerprint = data["settings"]["api_keys"][0]["fingerprint"]
                self.assertTrue(fingerprint)
                self.assertEqual(
                    data["settings"]["object_storage"]["secret_access_key"],
                    "secr...alue",
                )
                self.assertTrue(data["object_storage_ready"])

                unauthorized = client.get("/v1/request-logs")
                self.assertEqual(unauthorized.status_code, 401)
                authorized = client.get(
                    "/v1/request-logs",
                    headers={"Authorization": "Bearer sk-local-secret"},
                )
                self.assertEqual(authorized.status_code, 200)

                generated = client.post(
                    "/v1/system-settings/api-keys",
                    headers={"Authorization": "Bearer sk-local-secret"},
                    json={},
                )
                self.assertEqual(generated.status_code, 200)
                generated_key = generated.json()["api_key"]
                self.assertTrue(generated_key.startswith("sk-gemini-"))
                generated_fp = generated.json()["fingerprint"]
                self.assertEqual(
                    client.get(
                        "/v1/request-logs",
                        headers={"Authorization": f"Bearer {generated_key}"},
                    ).status_code,
                    200,
                )
                deleted = client.delete(
                    f"/v1/system-settings/api-keys/{generated_fp}",
                    headers={"Authorization": "Bearer sk-local-secret"},
                )
                self.assertEqual(deleted.status_code, 200)
                self.assertEqual(deleted.json()["deleted"], 1)

    def test_console_media_generation_can_store_media_to_object_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(self._config(tmp))

            async def fake_init(self, *args, **kwargs):
                self.client = FakeSession()
                self.account_status = AccountStatus.AVAILABLE

            async def fake_close(self):
                self.client = None

            async def fake_generate_content(self, prompt, **kwargs):
                return ModelOutput(
                    metadata=["cid", "rid"],
                    candidates=[
                        Candidate(
                            rcid="rcid",
                            text="ok",
                            generated_images=[
                                GeneratedImage(
                                    url="https://lh3.googleusercontent.com/demo.png",
                                    title="generated",
                                )
                            ],
                        )
                    ],
                )

            async def fake_upload(**kwargs):
                return {
                    "url": "https://cdn.example.test/tmp-assets/gemini-web/image.png",
                    "key": "tmp-assets/gemini-web/image.png",
                    "size": len(kwargs["data"]),
                    "content_type": kwargs["content_type"],
                }

            with (
                patch.object(GeminiClient, "init", fake_init),
                patch.object(GeminiClient, "close", fake_close),
                patch.object(GeminiClient, "generate_content", fake_generate_content),
                patch("gemini_webapi.server.app.httpx.Client", FakeMediaHTTPClient),
                patch("gemini_webapi.server.app.upload_s3_compatible", fake_upload),
                TestClient(app) as client,
            ):
                app.state.store.upsert_account(
                    secure_1psid="psid-one",
                    cookies={"__Secure-1PSID": "psid-one"},
                    name="one",
                )
                app.state.store.set_json_state(
                    "system_settings",
                    {
                        "api_keys": [],
                        "object_storage": {
                            "enabled": True,
                            "endpoint": "https://s3.example.test",
                            "region": "auto",
                            "bucket": "media",
                            "access_key_id": "access",
                            "secret_access_key": "secret",
                            "prefix": "gemini-web",
                            "public_url": "https://cdn.example.test",
                            "force_path_style": True,
                        },
                    },
                )
                response = client.post(
                    "/v1/gemini/generate",
                    json={
                        "prompt": "make image",
                        "mode": "image",
                        "store_media": True,
                    },
                )

                self.assertEqual(response.status_code, 200)
                media = client.get("/v1/gemini/media").json()["media"][0]
                self.assertTrue(media["stored"])
                self.assertEqual(
                    media["url"],
                    "https://cdn.example.test/tmp-assets/gemini-web/image.png",
                )
                self.assertEqual(media["content_url"], media["url"])
                self.assertEqual(
                    media["metadata"]["original_url"],
                    "https://lh3.googleusercontent.com/demo.png",
                )

    def test_gemini_generate_media_mode_requires_media_result(self):
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

            async def fake_init(self, *args, **kwargs):
                self.client = FakeSession()
                self.account_status = AccountStatus.AVAILABLE

            async def fake_close(self):
                self.client = None

            async def fake_generate_content(self, prompt, **kwargs):
                return ModelOutput(
                    metadata=["cid", "rid"],
                    candidates=[Candidate(rcid="rcid", text='{"ok":true}')],
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

            self.assertEqual(response.status_code, 502)
            response_body = response.json()
            error_detail = response_body.get("detail") or response_body.get("error", "")
            if isinstance(error_detail, dict):
                error_detail = error_detail.get("message", "")
            self.assertIn("没有可用的图片结果", error_detail)
            self.assertTrue(
                any(
                    log.output_type == "gemini_image"
                    and not log.ok
                    and "没有可用的图片结果" in (log.error or "")
                    for log in logs
                )
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
                    candidates=[
                        Candidate(
                            rcid="rcid",
                            text="ok",
                            generated_videos=[
                                GeneratedVideo(
                                    url="https://example.invalid/generated.mp4",
                                    title="generated",
                                )
                            ],
                        )
                    ],
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

    def test_clear_account_media_cooldowns_endpoint(self):
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
            with TestClient(app) as client:
                account = app.state.store.upsert_account(
                    secure_1psid="psid-one",
                    cookies={"__Secure-1PSID": "psid-one"},
                    name="one",
                )
                blocked_until = (
                    datetime.now(timezone.utc) + timedelta(hours=5)
                ).isoformat().replace("+00:00", "Z")
                for kind in ("image", "video"):
                    app.state.store.set_media_cooldown(
                        account_id=account.id,
                        kind=kind,
                        blocked_until=blocked_until,
                        reason=f"{kind} limit",
                    )

                response = client.post(
                    f"/v1/accounts/{account.id}/media-cooldowns/clear",
                    json={"kind": "image"},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["cleared"], ["image"])
                self.assertIsNone(app.state.store.get_media_cooldown(account.id, "image"))
                self.assertIsNotNone(app.state.store.get_media_cooldown(account.id, "video"))

                response = client.post(
                    f"/v1/accounts/{account.id}/media-cooldowns/clear",
                    json={},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["cleared"], ["video"])
                self.assertIsNone(app.state.store.get_media_cooldown(account.id, "video"))

                response = client.post(
                    f"/v1/accounts/{account.id}/media-cooldowns/clear",
                    json={"kind": "bad"},
                )
                self.assertEqual(response.status_code, 400)

                response = client.post(
                    "/v1/accounts/999/media-cooldowns/clear",
                    json={},
                )
                self.assertEqual(response.status_code, 404)

    def test_media_cooldowns_summary_endpoint(self):
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
            with TestClient(app) as client:
                active = app.state.store.upsert_account(
                    secure_1psid="psid-active",
                    cookies={"__Secure-1PSID": "psid-active"},
                    name="active",
                )
                expired = app.state.store.upsert_account(
                    secure_1psid="psid-expired",
                    cookies={"__Secure-1PSID": "psid-expired"},
                    name="expired",
                )
                app.state.store.set_account_validation(
                    expired.id,
                    expired=True,
                    status="UNAUTHENTICATED",
                    message="expired",
                )
                blocked_until = (
                    datetime.now(timezone.utc) + timedelta(hours=5)
                ).isoformat().replace("+00:00", "Z")
                app.state.store.set_media_cooldown(
                    account_id=active.id,
                    kind="video",
                    blocked_until=blocked_until,
                    reason="limit",
                )
                app.state.store.set_media_cooldown(
                    account_id=expired.id,
                    kind="video",
                    blocked_until=blocked_until,
                    reason="expired account limit",
                )

                response = client.get("/v1/media-cooldowns")
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertTrue(data["ok"])
                self.assertEqual(data["active_account_count"], 1)
                by_kind = {item["kind"]: item for item in data["summary"]}
                self.assertEqual(by_kind["video"]["total"], 1)
                self.assertEqual(by_kind["video"]["blocked"], 1)
                self.assertEqual(by_kind["video"]["available"], 0)
                self.assertEqual(by_kind["video"]["next"]["account_id"], active.id)
                self.assertEqual(by_kind["image"]["blocked"], 0)

                response = client.post(
                    "/v1/media-cooldowns/clear",
                    json={"kind": "video"},
                )
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["cleared"], 2)
                by_kind = {item["kind"]: item for item in data["summary"]}
                self.assertEqual(by_kind["video"]["blocked"], 0)

                response = client.post(
                    "/v1/media-cooldowns/clear",
                    json={"kind": "bad"},
                )
                self.assertEqual(response.status_code, 400)

    def test_request_validation_runs_before_account_selection(self):
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
            cases = [
                (
                    "/v1/gemini/generate",
                    {"prompt": "test", "model": "gemini", "mode": "bad"},
                    "mode must be one of",
                ),
                (
                    "/v1/gemini/stream",
                    {"prompt": "test", "model": "gemini", "mode": "bad"},
                    "mode must be one of",
                ),
                (
                    "/v1/generate",
                    {"prompt": "test", "model": "gemini", "mode": "bad"},
                    "mode must be one of",
                ),
                (
                    "/v1/gemini/generate",
                    {"prompt": "test", "model": "gemini-3-pro"},
                    "no longer exposed",
                ),
                (
                    "/v1/chat/completions",
                    {
                        "model": "gemini-3-pro",
                        "messages": [{"role": "user", "content": "test"}],
                    },
                    "no longer exposed",
                ),
            ]

            with TestClient(app) as client:
                for path, payload, expected_message in cases:
                    # 账号池为空时，参数错误仍应先返回 400，避免被未授权 401 掩盖。
                    with self.subTest(path=path):
                        response = client.post(path, json=payload)
                        self.assertEqual(response.status_code, 400)
                        self.assertIn(
                            expected_message, response.json()["error"]["message"]
                        )


if __name__ == "__main__":
    unittest.main()
