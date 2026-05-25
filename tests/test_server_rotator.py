import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gemini_webapi.exceptions import (
    MediaGenerationTemporarilyUnavailable,
    UsageLimitExceeded,
    VideoGenerationFailed,
    VideoGenerationNotSubmitted,
)
from gemini_webapi.server.database import AccountStore
from gemini_webapi.server.rotator import AccountRotator, ClientSlot


class FakeClient:
    def __init__(self):
        self.client = object()


class TestRotator(AccountRotator):
    def __init__(self, store):
        super().__init__(store)
        self.fake_client = FakeClient()
        self.slot = None

    async def _get_slot(self, account):
        if self.slot is None:
            self.slot = ClientSlot(
                account=account,
                client=self.fake_client,
                initialized=True,
            )
        return self.slot


class MultiAccountRotator(AccountRotator):
    def __init__(self, store):
        super().__init__(store)
        self.slots = {}

    async def _get_slot(self, account):
        slot = self.slots.get(account.id)
        if slot is None:
            slot = ClientSlot(
                account=account,
                client=FakeClient(),
                initialized=True,
            )
            self.slots[account.id] = slot
        return slot


class AccountRotatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_optional_failure_does_not_penalize_account_and_marks_client_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AccountStore(Path(tmp) / "app.db")
            try:
                store.upsert_account(
                    name="one",
                    secure_1psid="psid-one",
                    secure_1psidts="ts-one",
                    cookies={"__Secure-1PSID": "psid-one"},
                )
                rotator = TestRotator(store)

                async def operation(client):
                    client.client = None
                    raise RuntimeError("optional endpoint failed")

                with self.assertRaises(RuntimeError):
                    await rotator.run(
                        operation,
                        count_usage=False,
                        count_failure=False,
                        endpoint="/v1/gemini/gems",
                    )

                account = store.get_active_accounts()[0]
                self.assertEqual(account.failure_count, 0)
                self.assertFalse(rotator.slot.initialized)
                logs = store.list_request_logs()
                self.assertEqual(logs[0].endpoint, "/v1/gemini/gems")
                self.assertFalse(logs[0].ok)
            finally:
                store.close()

    async def test_video_attempt_is_recorded_only_for_video_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AccountStore(Path(tmp) / "app.db")
            try:
                store.upsert_account(
                    name="one",
                    secure_1psid="psid-one",
                    secure_1psidts="ts-one",
                    cookies={"__Secure-1PSID": "psid-one"},
                )
                rotator = TestRotator(store)

                async def operation(client):
                    return "ok"

                await rotator.run(
                    operation,
                    endpoint="/v1/gemini/generate",
                    output_type="gemini_native",
                )
                logs = store.list_request_logs(limit=10)
                self.assertFalse(
                    any(log.output_type == "video_generation_attempt" for log in logs)
                )

                await rotator.run(
                    operation,
                    endpoint="/v1/gemini/generate",
                    output_type="gemini_video",
                    require_video_generation=True,
                )
                logs = store.list_request_logs(limit=10)
                self.assertTrue(
                    any(log.output_type == "video_generation_attempt" for log in logs)
                )

                await rotator.run(
                    operation,
                    endpoint="/v1/gemini/generate",
                    output_type="gemini_image",
                    media_generation_mode="image",
                )
                logs = store.list_request_logs(limit=10)
                self.assertTrue(
                    any(log.output_type == "image_generation_attempt" for log in logs)
                )
            finally:
                store.close()

    async def test_unconfirmed_video_generation_deletes_attempt_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AccountStore(Path(tmp) / "app.db")
            try:
                store.upsert_account(
                    name="one",
                    secure_1psid="psid-one",
                    secure_1psidts="ts-one",
                    cookies={"__Secure-1PSID": "psid-one"},
                )
                rotator = TestRotator(store)

                async def operation(client):
                    raise VideoGenerationNotSubmitted("not submitted")

                with self.assertRaises(VideoGenerationNotSubmitted):
                    await rotator.run(
                        operation,
                        endpoint="/v1/gemini/generate",
                        output_type="gemini_video",
                        require_video_generation=True,
                    )

                logs = store.list_request_logs(limit=10)
                self.assertFalse(
                    any(log.output_type == "video_generation_attempt" for log in logs)
                )
                self.assertFalse(any(log.error == "not submitted" for log in logs))
            finally:
                store.close()

    async def test_failed_video_generation_deletes_attempt_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AccountStore(Path(tmp) / "app.db")
            try:
                store.upsert_account(
                    name="one",
                    secure_1psid="psid-one",
                    secure_1psidts="ts-one",
                    cookies={"__Secure-1PSID": "psid-one"},
                )
                rotator = TestRotator(store)

                async def operation(client):
                    raise VideoGenerationFailed("no video")

                with self.assertRaises(VideoGenerationFailed):
                    await rotator.run(
                        operation,
                        endpoint="/v1/gemini/generate",
                        output_type="gemini_video",
                        require_video_generation=True,
                    )

                logs = store.list_request_logs(limit=10)
                self.assertFalse(
                    any(log.output_type == "video_generation_attempt" for log in logs)
                )
                self.assertFalse(any(log.error == "no video" for log in logs))
            finally:
                store.close()

    async def test_media_limit_sets_cooldown_and_tries_next_account(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AccountStore(Path(tmp) / "app.db")
            try:
                first = store.upsert_account(
                    name="one",
                    secure_1psid="psid-one",
                    secure_1psidts="ts-one",
                    cookies={"__Secure-1PSID": "psid-one"},
                )
                second = store.upsert_account(
                    name="two",
                    secure_1psid="psid-two",
                    secure_1psidts="ts-two",
                    cookies={"__Secure-1PSID": "psid-two"},
                )
                store.set_state("current_account_id", str(first.id))
                rotator = MultiAccountRotator(store)
                calls = []

                async def operation(client):
                    account_id = next(
                        slot.account.id
                        for slot in rotator.slots.values()
                        if slot.client is client
                    )
                    calls.append(account_id)
                    if account_id == first.id:
                        raise UsageLimitExceeded("image usage limit reached")
                    return "ok"

                result = await rotator.run(
                    operation,
                    endpoint="/v1/gemini/generate",
                    output_type="gemini_image",
                    media_generation_mode="image",
                )

                self.assertEqual(result, "ok")
                self.assertEqual(calls, [first.id, second.id])
                self.assertIsNotNone(store.get_media_cooldown(first.id, "image"))
                logs = store.list_request_logs(limit=20)
                self.assertTrue(any(log.output_type == "image_generation_attempt" for log in logs))
            finally:
                store.close()

    async def test_media_cooldown_skips_account_without_calling_operation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AccountStore(Path(tmp) / "app.db")
            try:
                first = store.upsert_account(
                    name="one",
                    secure_1psid="psid-one",
                    secure_1psidts="ts-one",
                    cookies={"__Secure-1PSID": "psid-one"},
                )
                second = store.upsert_account(
                    name="two",
                    secure_1psid="psid-two",
                    secure_1psidts="ts-two",
                    cookies={"__Secure-1PSID": "psid-two"},
                )
                blocked_until = (
                    datetime.now(timezone.utc) + timedelta(hours=1)
                ).isoformat().replace("+00:00", "Z")
                store.set_media_cooldown(
                    account_id=first.id,
                    kind="video",
                    blocked_until=blocked_until,
                    reason="video limit",
                )
                store.set_state("current_account_id", str(first.id))
                rotator = MultiAccountRotator(store)
                calls = []

                async def operation(client):
                    account_id = next(
                        slot.account.id
                        for slot in rotator.slots.values()
                        if slot.client is client
                    )
                    calls.append(account_id)
                    return "ok"

                result = await rotator.run(
                    operation,
                    endpoint="/v1/gemini/generate",
                    output_type="gemini_video",
                    media_generation_mode="video",
                )

                self.assertEqual(result, "ok")
                self.assertEqual(calls, [second.id])
            finally:
                store.close()

    async def test_all_media_accounts_in_cooldown_raise_without_calling_operation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AccountStore(Path(tmp) / "app.db")
            try:
                first = store.upsert_account(
                    name="one",
                    secure_1psid="psid-one",
                    secure_1psidts="ts-one",
                    cookies={"__Secure-1PSID": "psid-one"},
                )
                second = store.upsert_account(
                    name="two",
                    secure_1psid="psid-two",
                    secure_1psidts="ts-two",
                    cookies={"__Secure-1PSID": "psid-two"},
                )
                blocked_until = (
                    datetime.now(timezone.utc) + timedelta(hours=1)
                ).isoformat().replace("+00:00", "Z")
                for account in (first, second):
                    store.set_media_cooldown(
                        account_id=account.id,
                        kind="video",
                        blocked_until=blocked_until,
                        reason="video limit",
                    )
                rotator = MultiAccountRotator(store)

                async def operation(client):
                    raise AssertionError("operation should not be called")

                with self.assertRaises(MediaGenerationTemporarilyUnavailable):
                    await rotator.run(
                        operation,
                        endpoint="/v1/gemini/generate",
                        output_type="gemini_video",
                        media_generation_mode="video",
                    )
            finally:
                store.close()

    async def test_unconfirmed_video_does_not_set_cooldown(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AccountStore(Path(tmp) / "app.db")
            try:
                account = store.upsert_account(
                    name="one",
                    secure_1psid="psid-one",
                    secure_1psidts="ts-one",
                    cookies={"__Secure-1PSID": "psid-one"},
                )
                rotator = TestRotator(store)

                async def operation(client):
                    raise VideoGenerationNotSubmitted("not submitted")

                with self.assertRaises(VideoGenerationNotSubmitted):
                    await rotator.run(
                        operation,
                        endpoint="/v1/gemini/generate",
                        output_type="gemini_video",
                        media_generation_mode="video",
                    )

                self.assertIsNone(store.get_media_cooldown(account.id, "video"))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
