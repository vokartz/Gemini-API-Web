import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
