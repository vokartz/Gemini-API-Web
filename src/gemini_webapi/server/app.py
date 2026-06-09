from __future__ import annotations

import mimetypes
import asyncio
import hashlib
import hmac
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import orjson as json
import websockets
import httpx
from curl_cffi.requests import AsyncSession
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..constants import Model
from ..utils import logger
from ..exceptions import (
    AuthError,
    GeminiError,
    MediaGenerationEmptyResult,
    MediaGenerationTemporarilyUnavailable,
    ModelInvalid,
    VideoGenerationFailed,
    VideoGenerationNotSubmitted,
)
from ..types import DeepResearchPlan
from .auth_browser import AuthBrowserManager, AuthBrowserUnavailable
from .config import ServerConfig
from .database import AccountStore
from .object_storage import (
    ObjectStorageConfig,
    build_media_object_key,
    upload_s3_compatible,
)
from .rotator import AccountRotator


MEDIA_CONTENT_MAX_BYTES = 100 * 1024 * 1024
MEDIA_CONTENT_ALLOWED_HOST_SUFFIXES = (
    ".google.com",
    ".googleusercontent.com",
    ".usercontent.google.com",
    ".gstatic.com",
    ".googlevideo.com",
    "storage.googleapis.com",
)
SYSTEM_SETTINGS_KEY = "system_settings"
MASKED_SECRET = "********"
PAGE_ROUTES = {"/", "/accounts.html", "/gems.html", "/api.html", "/history.html"}
DEFAULT_SYSTEM_SETTINGS = {
    "api_keys": [],
    "object_storage": {
        "enabled": False,
        "endpoint": "",
        "region": "auto",
        "bucket": "",
        "access_key_id": "",
        "secret_access_key": "",
        "prefix": "gemini-web",
        "public_url": "",
        "force_path_style": True,
    },
}


class GenerateRequest(BaseModel):
    prompt: str
    model: str | None = None
    mode: str | None = "image"
    temporary: bool = False
    gem_id: str | None = None
    aspect_ratio: str | None = None
    output_format: str = "png"
    store_media: bool = True
    file_ids: list[str] = Field(default_factory=list)


class GeminiGenerateRequest(BaseModel):
    prompt: str
    model: str | None = None
    mode: str | None = None
    temporary: bool = False
    store_media: bool = False
    gem: str | None = None
    gem_id: str | None = None
    gem_name: str | None = None
    deep_research: bool = False
    extensions: list[str] | dict[str, Any] | None = None
    file_ids: list[str] = Field(default_factory=list)


class GemRequest(BaseModel):
    name: str
    prompt: str
    description: str = ""


class CustomGemRequest(BaseModel):
    name: str
    prompt: str
    description: str = ""
    is_default: bool = False


class DeepResearchCreateRequest(BaseModel):
    prompt: str
    model: str | None = None


class DeepResearchStartRequest(BaseModel):
    job_id: str | None = None
    plan: dict[str, Any] | None = None
    confirm_prompt: str | None = None


class DeepResearchWaitRequest(BaseModel):
    job_id: str
    poll_interval: float = 10.0
    timeout: float = 600.0


class AccountRequest(BaseModel):
    name: str | None = None
    secure_1psid: str = Field(alias="__Secure-1PSID")
    secure_1psidts: str | None = Field(default=None, alias="__Secure-1PSIDTS")
    cookies: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True

    model_config = {"populate_by_name": True}


class AccountToggleRequest(BaseModel):
    enabled: bool


class SettingsRequest(BaseModel):
    switch_on_uses: int | None = None
    failure_threshold: int | None = None


class SwitchAccountRequest(BaseModel):
    account_id: int | None = None


class ClearMediaCooldownRequest(BaseModel):
    kind: str | None = None


class ObjectStorageSettings(BaseModel):
    enabled: bool = False
    endpoint: str = ""
    region: str = "auto"
    bucket: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    prefix: str = "gemini-web"
    public_url: str = ""
    force_path_style: bool = True


class SystemSettingsRequest(BaseModel):
    api_keys: list[str] | None = None
    object_storage: ObjectStorageSettings | None = None


class AdminLoginRequest(BaseModel):
    password: str


class AuthClickRequest(BaseModel):
    x: float
    y: float


class AuthTypeRequest(BaseModel):
    text: str


class AuthPressRequest(BaseModel):
    key: str


class AuthSaveRequest(BaseModel):
    name: str | None = None


MODEL_ALIASES = {
    "gemini": "gemini-3.1-pro",
}

PUBLIC_MODEL_IDS = {
    "gemini-3.1-pro",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
}

REMOVED_MODEL_IDS = {
    "gemini-3-pro",
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
    "gemini-3-flash",
    "gemini-3-flash-preview",
    "gemini-3-flash-thinking",
}


def _resolve_model_arg(model: str | None) -> str | None:
    if not model:
        return None
    model_key = model.lower()
    if model_key == "unspecified":
        return None
    if model_key in REMOVED_MODEL_IDS:
        raise ValueError(
            f"Model '{model}' is no longer exposed. Use gemini-3.1-pro, gemini-3.5-flash, or gemini-3.1-flash-lite."
        )
    resolved = MODEL_ALIASES.get(model_key, model_key)
    if resolved not in PUBLIC_MODEL_IDS:
        raise ValueError(
            f"Unsupported model '{model}'. Use gemini, gemini-3.1-pro, gemini-3.5-flash, or gemini-3.1-flash-lite."
        )
    return resolved


GEM_DEFAULT_PROMPT = "Merhaba"


def _effective_prompt(prompt: str | None, gem_arg: str | None) -> str:
    # Bir gem seçildiğinde kullanıcının boş istek göndermesine izin vermek için, prompt boşsa
    # gem'i başlatan varsayılan bir mesaj kullanılır. Gemini boş prompt'u kabul etmediğinden
    # (client tarafında `assert prompt`) bu olmadan gem-only gönderim başarısız olur.
    text = prompt or ""
    if gem_arg and not text.strip():
        return GEM_DEFAULT_PROMPT
    return text


def _generation_mode_arg(mode: str | None) -> str | None:
    normalized = (mode or "").strip().lower()
    if not normalized:
        return None
    if normalized not in {"image", "video", "audio"}:
        raise ValueError("mode must be one of: image, video, audio.")
    return normalized


SUPPORTED_ASPECT_RATIOS = {"1:1", "16:9", "9:16"}


def _aspect_ratio_arg(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().replace(" ", "")
    if normalized not in SUPPORTED_ASPECT_RATIOS:
        raise ValueError("aspect_ratio must be one of: 1:1, 16:9, 9:16.")
    return normalized


def _apply_aspect_ratio(prompt: str, aspect_ratio: str | None) -> str:
    # Gemini Web'in görsel üretiminde en-boy oranı için ayrı bir parametre yoktur; oran prompt
    # içine açık bir talimat olarak eklenir.
    if not aspect_ratio:
        return prompt
    return f"{prompt}\n\n(Generate the image with a {aspect_ratio} aspect ratio.)"


def _output_format_arg(value: str | None) -> str:
    normalized = (value or "png").strip().lower()
    if normalized in {"jpg", "jpeg"}:
        return "jpeg"
    if normalized == "png":
        return "png"
    raise ValueError("output_format must be png or jpg.")


def _ensure_media_generation_result(output: Any, mode: str | None) -> None:
    if not mode:
        return
    classified = _classified_output(output)
    has_result = {
        "image": bool(classified["images"]),
        "video": bool(classified["videos"]),
        "audio": bool(classified["media"]),
    }[mode]
    if has_result:
        return
    labels = {"image": "Resim", "video": "Video", "audio": "Ses"}
    # Üst sunucu 2xx döndürdü ancak medya sonucu yok; çağıran tarafın metni/JSON'u başarılı medya görevi sanmaması için açıkça başarısız olmalıdır.
    raise MediaGenerationEmptyResult(
        f"{labels[mode]} oluşturma isteği döndü, ancak yanıtta kullanılabilir {labels[mode]} sonucu yok."
    )


def _error_status(exc: Exception) -> int:
    if isinstance(exc, AuthError):
        return 401
    if isinstance(exc, (ValueError, ModelInvalid)):
        return 400
    if isinstance(exc, VideoGenerationNotSubmitted):
        return 409
    if isinstance(exc, VideoGenerationFailed):
        return 502
    if isinstance(exc, MediaGenerationTemporarilyUnavailable):
        return 429
    if isinstance(exc, MediaGenerationEmptyResult):
        return 502
    if isinstance(exc, GeminiError):
        return 502
    return 500


def _openai_error(message: str, status_code: int, error_type: str = "api_error") -> dict:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": status_code,
        }
    }


def _dump_model(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _image_dict(image: Any, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "url": getattr(image, "url", ""),
        "title": getattr(image, "title", None),
        "alt": getattr(image, "alt", None),
        "image_id": getattr(image, "image_id", None),
    }


def _video_dict(video: Any, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "url": getattr(video, "url", ""),
        "thumbnail": getattr(video, "thumbnail", None),
        "title": getattr(video, "title", None),
    }


def _media_dict(media: Any) -> dict[str, Any]:
    return {
        "kind": "audio",
        "url": getattr(media, "mp3_url", "") or getattr(media, "url", ""),
        "mp3_url": getattr(media, "mp3_url", ""),
        "mp4_url": getattr(media, "mp4_url", ""),
        "thumbnail": getattr(media, "mp3_thumbnail", "")
        or getattr(media, "mp4_thumbnail", ""),
        "title": getattr(media, "title", None),
    }


def _classified_output(output: Any) -> dict[str, Any]:
    candidate = output.candidates[output.chosen] if output.candidates else None
    web_images = [_image_dict(image, "web_image") for image in getattr(candidate, "web_images", [])]
    generated_images = [
        _image_dict(image, "image") for image in getattr(candidate, "generated_images", [])
    ]
    videos = [_video_dict(video, "video") for video in getattr(candidate, "generated_videos", [])]
    media = [_media_dict(item) for item in getattr(candidate, "generated_media", [])]
    return {
        "text": output.text,
        "thoughts": output.thoughts,
        "images": generated_images,
        "videos": videos,
        "media": media,
        "web_images": web_images,
        "deep_research_plan": _dump_model(output.deep_research_plan),
    }


def _media_entries(output: Any) -> list[dict[str, Any]]:
    classified = _classified_output(output)
    entries: list[dict[str, Any]] = []
    entries.extend(classified["images"])
    entries.extend(classified["videos"])
    entries.extend(classified["media"])
    entries.extend(classified["web_images"])
    return [entry for entry in entries if entry.get("url")]


def _media_record_dict(item: Any) -> dict[str, Any]:
    data = item.__dict__.copy()
    if "token" not in data and hasattr(item, "token"):
        data["token"] = getattr(item, "token")
    storage = (data.get("metadata") or {}).get("object_storage") or {}
    if data.get("token"):
        data["content_url"] = (
            storage.get("url") or f"/v1/gemini/media/{data['token']}/content"
        )
        data["cached"] = bool(data.get("local_path"))
        data["stored"] = bool(storage.get("url"))
    return data


def _public_media_content_path(path: str) -> bool:
    # OpenAI resim arayüzünden dönen medya proxy bağlantıları harici istemciler tarafından doğrudan istenebilir; burada yalnızca rastgele token'lı içerik indirme yolu açık bırakılmaktadır.
    return path.startswith("/v1/gemini/media/") and path.endswith("/content")


def _mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return MASKED_SECRET
    return f"{value[:4]}...{value[-4:]}"


def _key_fingerprint(value: str) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL, value).hex[:16]


def _admin_secret(config: ServerConfig) -> str:
    # Yönetim paneli oturum imzalama anahtarı: Sunucu dağıtımında ayrı olarak yapılandırılması önerilir; yeniden başlatma veya parola değişikliği sonrasında oturum durumunun kontrolden çıkmaması için.
    return (
        config.admin_session_secret
        or config.admin_password
        or "gemini-webapi-local-admin"
    )


def _admin_session_value(config: ServerConfig) -> str:
    # Cookie'de yalnızca imzalanmış zaman damgası saklanır, yönetici parolasının kendisi saklanmaz.
    timestamp = str(int(time.time()))
    signature = hmac.new(
        _admin_secret(config).encode("utf-8"),
        timestamp.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{timestamp}.{signature}"


def _admin_session_valid(config: ServerConfig, value: str | None) -> bool:
    # Yönetici parolası yapılandırılmamışsa yerel geliştirme modunda kalınır; varsayılan başlatmada kullanıcının yönetim panelinin dışında kalmaması için.
    if not config.admin_password or not value:
        return not config.admin_password
    try:
        timestamp, signature = value.split(".", 1)
        issued_at = int(timestamp)
    except (ValueError, TypeError):
        return False
    if time.time() - issued_at > 7 * 24 * 60 * 60:
        return False
    expected = hmac.new(
        _admin_secret(config).encode("utf-8"),
        timestamp.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(signature, expected)


def _normalize_api_keys(values: list[str] | tuple[str, ...] | None) -> list[str]:
    seen: set[str] = set()
    keys: list[str] = []
    for value in values or []:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        keys.append(item)
    return keys


def _resolve_masked_api_keys(
    incoming: list[str] | None,
    existing: list[str],
) -> list[str]:
    """Ön yüz maskelenmiş API Key gösterir; kaydetme sırasında maskelenmiş değerle eşleştirilir ve orijinal anahtar korunur."""
    resolved: list[str] = []
    masks = {_mask_secret(key): key for key in existing}
    for value in incoming or []:
        item = str(value or "").strip()
        if not item:
            continue
        resolved.append(masks.get(item, item))
    return _normalize_api_keys(resolved)


def _merge_system_settings(
    current: dict[str, Any],
    request: SystemSettingsRequest | None = None,
) -> dict[str, Any]:
    merged = {
        "api_keys": _normalize_api_keys(current.get("api_keys")),
        "object_storage": {
            **DEFAULT_SYSTEM_SETTINGS["object_storage"],
            **(current.get("object_storage") or {}),
        },
    }
    if request is None:
        return merged
    if request.api_keys is not None:
        merged["api_keys"] = _resolve_masked_api_keys(
            request.api_keys,
            merged["api_keys"],
        )
    if request.object_storage is not None:
        incoming = request.object_storage.model_dump()
        current_secret = merged["object_storage"].get("secret_access_key", "")
        if incoming.get("secret_access_key") in {
            MASKED_SECRET,
            _mask_secret(current_secret),
        }:
            incoming["secret_access_key"] = merged["object_storage"].get(
                "secret_access_key",
                "",
            )
        merged["object_storage"] = {
            **merged["object_storage"],
            **incoming,
        }
    return merged


def _public_system_settings(settings: dict[str, Any]) -> dict[str, Any]:
    public = _merge_system_settings(settings)
    public["api_keys"] = [
        {
            "fingerprint": _key_fingerprint(key),
            "masked": _mask_secret(key),
        }
        for key in public["api_keys"]
    ]
    storage = dict(public["object_storage"])
    storage["secret_access_key"] = _mask_secret(storage.get("secret_access_key"))
    public["object_storage"] = storage
    return public


def _media_cooldown_summary(status: dict[str, Any]) -> dict[str, Any]:
    accounts = status.get("accounts") or []
    active_accounts = [
        account for account in accounts if account.get("enabled") and not account.get("expired")
    ]
    labels = {"image": "Resim", "video": "Video", "audio": "Ses"}
    summary: list[dict[str, Any]] = []
    for kind in ("image", "video", "audio"):
        blocked: list[dict[str, Any]] = []
        for account in active_accounts:
            cooldown = (account.get("media_cooldowns") or {}).get(kind)
            if not cooldown:
                continue
            blocked.append(
                {
                    "account_id": account.get("id"),
                    "account_name": account.get("name"),
                    "blocked_until": cooldown.get("blocked_until"),
                    "remaining_seconds": cooldown.get("remaining_seconds", 0),
                    "reason": cooldown.get("reason", ""),
                }
            )
        blocked.sort(key=lambda item: item.get("remaining_seconds") or 0)
        summary.append(
            {
                "kind": kind,
                "label": labels[kind],
                "total": len(active_accounts),
                "blocked": len(blocked),
                "available": max(0, len(active_accounts) - len(blocked)),
                "next": blocked[0] if blocked else None,
                "accounts": blocked,
            }
        )
    return {"summary": summary, "active_account_count": len(active_accounts)}


def _media_host_allowed(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    return any(
        host == suffix or host.endswith(suffix)
        for suffix in MEDIA_CONTENT_ALLOWED_HOST_SUFFIXES
    )


def _media_content_type_allowed(kind: str | None, content_type: str | None) -> bool:
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    if not media_type:
        return True
    if kind in {"image", "web_image"}:
        return media_type.startswith("image/")
    if kind == "video":
        return media_type.startswith("video/")
    if kind == "audio":
        return media_type.startswith("audio/")
    return media_type.startswith(("image/", "video/", "audio/"))


def _job_dict(job: Any) -> dict[str, Any]:
    return job.__dict__


def _file_dict(file_record: Any) -> dict[str, Any]:
    return file_record.__dict__


def _gem_dict(gem: Any) -> dict[str, Any]:
    return {
        "id": gem.id,
        "name": gem.name,
        "description": gem.description,
        "prompt": gem.prompt,
        "predefined": gem.predefined,
    }


def create_app(config: ServerConfig | None = None):
    import orjson as json
    from pathlib import Path

    config = config or ServerConfig.from_env()
    store = AccountStore(config.database_path)
    store.import_accounts_file(config.accounts_file)
    switch_on_uses = int(store.get_state("switch_on_uses", str(config.switch_on_uses)))
    failure_threshold = int(
        store.get_state("failure_threshold", str(config.failure_threshold))
    )
    rotator = AccountRotator(
        store,
        switch_on_uses=switch_on_uses,
        failure_threshold=failure_threshold,
        immediate_switch_status_codes=config.immediate_switch_status_codes,
        proxy=config.proxy,
        request_timeout=config.request_timeout,
        auto_refresh=config.auto_refresh,
        account_refresh_interval=config.account_refresh_interval,
    )
    auth_browser = AuthBrowserManager(
        store,
        start_url=config.auth_url,
        headless=config.auth_headless,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store
        app.state.rotator = rotator
        app.state.auth_browser = auth_browser
        rotator.start_background_refresh()
        yield
        await auth_browser.close()
        await rotator.close()
        store.close()

    app = FastAPI(title="gemini-webapi server", version="0.1.0", lifespan=lifespan)
    # Harici Web paneli veya tarayıcı SDK çağrıları için CORS gereklidir; sunucu dağıtımında izin verilen kaynaklar ortam değişkeniyle daraltılabilir.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_allow_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        error_type = "invalid_request_error"
        if exc.status_code in {401, 403}:
            error_type = "authentication_error"
        return JSONResponse(
            status_code=exc.status_code,
            content=_openai_error(detail, exc.status_code, error_type),
        )

    @app.middleware("http")
    async def bearer_auth(request: Request, call_next):
        path = request.url.path
        if request.method == "OPTIONS":
            return await call_next(request)
        # Yönetim paneli girişi yalnızca konsolu ve yönetim uç noktalarını korur; OpenAI uyumlu uç noktalar harici API Key ile kimlik doğrulamaya devam eder.
        admin_public_paths = {
            "/health",
            "/v1/admin/status",
            "/v1/admin/login",
            "/static",
        }
        if config.admin_password and not (
            path == "/"
            or path in PAGE_ROUTES
            or path.startswith("/static/")
            or path in admin_public_paths
        ):
            admin_ok = _admin_session_valid(
                config,
                request.cookies.get("gemini_admin_session"),
            )
            external_api_path = path in {
                "/v1/generate",
                "/v1/gemini/generate",
                "/v1/gemini/stream",
                "/v1/gemini/media",
                "/v1/gemini/files",
            } or _public_media_content_path(path)
            if not admin_ok and not external_api_path:
                return JSONResponse(
                    status_code=401,
                    content={"ok": False, "detail": "Admin login required."},
                )

        system_settings = _merge_system_settings(
            store.get_json_state(SYSTEM_SETTINGS_KEY, DEFAULT_SYSTEM_SETTINGS)
        )
        allowed_api_keys = set(config.api_keys) | set(system_settings["api_keys"])
        if (
            allowed_api_keys
            and path.startswith("/v1/")
            and path not in {
                "/v1/status",
                "/v1/admin/status",
                "/v1/admin/login",
                "/v1/admin/logout",
            }
            and not _public_media_content_path(path)
            and not (
                config.admin_password
                and _admin_session_valid(
                    config,
                    request.cookies.get("gemini_admin_session"),
                )
            )
        ):
            auth = request.headers.get("authorization", "")
            token = auth.removeprefix("Bearer ").strip()
            if token not in allowed_api_keys:
                return JSONResponse(
                    status_code=401,
                    content=_openai_error(
                        "Invalid or missing API key.",
                        401,
                        "authentication_error",
                    ),
                )
        return await call_next(request)

    async def _proxy_novnc(path: str, request: Request) -> Response:
        url = f"http://127.0.0.1:6080/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                proxied = await client.request(
                    request.method,
                    url,
                    headers={
                        key: value
                        for key, value in request.headers.items()
                        if key.lower() not in {"host", "connection"}
                    },
                    content=await request.body(),
                )
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Yetkilendirme tarayıcısı henüz başlatılmadı, lütfen önce hesap ayarlarından web yetkilendirmesine tıklayın.",
                ) from exc
        return Response(
            content=proxied.content,
            status_code=proxied.status_code,
            media_type=proxied.headers.get("content-type"),
        )

    async def _proxy_novnc_websocket(websocket: WebSocket) -> None:
        """Yönetim paneliyle aynı kaynaktan gelen WebSocket'i konteyner içindeki noVNC'ye yönlendirir; yetkilendirme sayfasının çapraz port nedeniyle kullanılamamasını önler."""
        await websocket.accept()
        try:
            async with websockets.connect("ws://127.0.0.1:6080/websockify") as upstream:
                async def client_to_upstream() -> None:
                    try:
                        while True:
                            message = await websocket.receive()
                            if message["type"] == "websocket.disconnect":
                                await upstream.close()
                                return
                            if "bytes" in message and message["bytes"] is not None:
                                await upstream.send(message["bytes"])
                            elif "text" in message and message["text"] is not None:
                                await upstream.send(message["text"])
                    except WebSocketDisconnect:
                        await upstream.close()

                async def upstream_to_client() -> None:
                    async for message in upstream:
                        if isinstance(message, bytes):
                            await websocket.send_bytes(message)
                        else:
                            await websocket.send_text(message)

                done, pending = await asyncio.wait(
                    {
                        asyncio.create_task(client_to_upstream()),
                        asyncio.create_task(upstream_to_client()),
                    },
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                for task in done:
                    task.result()
        except Exception:
            try:
                await websocket.close(code=1011)
            except RuntimeError:
                pass

    @app.websocket("/novnc/websockify")
    async def novnc_websockify(websocket: WebSocket) -> None:
        await _proxy_novnc_websocket(websocket)

    @app.websocket("/novnc/novnc/websockify")
    async def novnc_websockify_legacy(websocket: WebSocket) -> None:
        await _proxy_novnc_websocket(websocket)

    @app.api_route("/novnc", methods=["GET", "POST"])
    async def novnc_root(request: Request) -> Response:
        return await _proxy_novnc("", request)

    @app.api_route("/novnc/{path:path}", methods=["GET", "POST"])
    async def novnc_proxy(path: str, request: Request) -> Response:
        return await _proxy_novnc(path, request)

    def _page(filename: str) -> FileResponse:
        return FileResponse(static_dir / filename)

    @app.get("/")
    async def console() -> FileResponse:
        return _page("index.html")

    @app.get("/accounts.html")
    async def page_accounts() -> FileResponse:
        return _page("accounts.html")

    @app.get("/gems.html")
    async def page_gems() -> FileResponse:
        return _page("gems.html")

    @app.get("/api.html")
    async def page_api() -> FileResponse:
        return _page("api.html")

    @app.get("/history.html")
    async def page_history() -> FileResponse:
        return _page("history.html")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/v1/admin/status")
    async def admin_status(request: Request) -> dict[str, Any]:
        enabled = bool(config.admin_password)
        return {
            "enabled": enabled,
            "authenticated": _admin_session_valid(
                config,
                request.cookies.get("gemini_admin_session"),
            ),
        }

    @app.post("/v1/admin/login")
    async def admin_login(request: AdminLoginRequest) -> Response:
        if not config.admin_password:
            return JSONResponse({"ok": True, "enabled": False, "authenticated": True})
        if not hmac.compare_digest(request.password, config.admin_password):
            raise HTTPException(status_code=401, detail="Yönetici şifresi yanlış.")
        response = JSONResponse(
            {"ok": True, "enabled": True, "authenticated": True}
        )
        response.set_cookie(
            "gemini_admin_session",
            _admin_session_value(config),
            httponly=True,
            samesite="lax",
            secure=False,
            max_age=7 * 24 * 60 * 60,
            path="/",
        )
        return response

    @app.post("/v1/admin/logout")
    async def admin_logout() -> Response:
        response = JSONResponse({"ok": True})
        response.delete_cookie("gemini_admin_session", path="/")
        return response

    @app.get("/v1/status")
    async def status() -> dict[str, Any]:
        return rotator.status()

    @app.get("/v1/media-cooldowns")
    async def media_cooldowns() -> dict[str, Any]:
        status_data = rotator.status()
        return {
            "ok": True,
            **_media_cooldown_summary(status_data),
        }

    @app.post("/v1/media-cooldowns/clear")
    async def clear_media_cooldowns(request: ClearMediaCooldownRequest) -> dict[str, Any]:
        kind = (request.kind or "").strip().lower()
        if kind and kind not in {"audio", "image", "video"}:
            raise HTTPException(
                status_code=400,
                detail="kind must be one of: image, video, audio.",
            )
        cleared = store.clear_media_cooldowns(kind or None)
        status_data = rotator.status()
        return {
            "ok": True,
            "kind": kind or None,
            "cleared": cleared,
            **_media_cooldown_summary(status_data),
        }

    @app.get("/v1/settings")
    async def get_settings() -> dict[str, Any]:
        return {
            "switch_on_uses": rotator.switch_on_uses,
            "failure_threshold": rotator.failure_threshold,
        }

    @app.patch("/v1/settings")
    async def update_settings(request: SettingsRequest) -> dict[str, Any]:
        if request.switch_on_uses is not None:
            store.set_state("switch_on_uses", str(max(0, request.switch_on_uses)))
        if request.failure_threshold is not None:
            store.set_state("failure_threshold", str(max(0, request.failure_threshold)))
        rotator.configure(
            switch_on_uses=request.switch_on_uses,
            failure_threshold=request.failure_threshold,
        )
        return {"ok": True, "settings": await get_settings()}

    @app.get("/v1/system-settings")
    async def get_system_settings() -> dict[str, Any]:
        settings = _merge_system_settings(
            store.get_json_state(SYSTEM_SETTINGS_KEY, DEFAULT_SYSTEM_SETTINGS)
        )
        return {
            "ok": True,
            "settings": _public_system_settings(settings),
            "object_storage_ready": ObjectStorageConfig.from_dict(
                settings["object_storage"]
            ).usable(),
        }

    @app.patch("/v1/system-settings")
    async def update_system_settings(request: SystemSettingsRequest) -> dict[str, Any]:
        current = store.get_json_state(SYSTEM_SETTINGS_KEY, DEFAULT_SYSTEM_SETTINGS)
        settings = _merge_system_settings(current, request)
        store.set_json_state(SYSTEM_SETTINGS_KEY, settings)
        return {
            "ok": True,
            "settings": _public_system_settings(settings),
            "object_storage_ready": ObjectStorageConfig.from_dict(
                settings["object_storage"]
            ).usable(),
        }

    @app.post("/v1/system-settings/api-keys")
    async def create_system_api_key() -> dict[str, Any]:
        current = store.get_json_state(SYSTEM_SETTINGS_KEY, DEFAULT_SYSTEM_SETTINGS)
        settings = _merge_system_settings(current)
        api_key = f"sk-gemini-{secrets.token_urlsafe(32)}"
        settings["api_keys"] = _normalize_api_keys([*settings["api_keys"], api_key])
        store.set_json_state(SYSTEM_SETTINGS_KEY, settings)
        return {
            "ok": True,
            "api_key": api_key,
            "fingerprint": _key_fingerprint(api_key),
            "settings": _public_system_settings(settings),
        }

    @app.delete("/v1/system-settings/api-keys/{fingerprint}")
    async def delete_system_api_key(fingerprint: str) -> dict[str, Any]:
        current = store.get_json_state(SYSTEM_SETTINGS_KEY, DEFAULT_SYSTEM_SETTINGS)
        settings = _merge_system_settings(current)
        before = len(settings["api_keys"])
        settings["api_keys"] = [
            key for key in settings["api_keys"] if _key_fingerprint(key) != fingerprint
        ]
        store.set_json_state(SYSTEM_SETTINGS_KEY, settings)
        return {
            "ok": True,
            "deleted": before - len(settings["api_keys"]),
            "settings": _public_system_settings(settings),
        }

    @app.get("/v1/request-logs")
    async def request_logs(limit: int = 80) -> dict[str, Any]:
        return {"logs": rotator.request_logs(limit=max(1, min(limit, 500)))}

    @app.get("/v1/gemini/media")
    async def gemini_media(limit: int = 80, kind: str | None = None) -> dict[str, Any]:
        return {
            "media": [
                _media_record_dict(item)
                for item in store.list_media_outputs(
                    limit=max(1, min(limit, 500)), kind=kind
                )
            ]
        }

    def _media_content_url(item: Any) -> str:
        storage = (item.metadata or {}).get("object_storage") or {}
        if storage.get("url"):
            return storage["url"]
        if item.token:
            return f"/v1/gemini/media/{item.token}/content"
        return item.url

    @app.get("/v1/generations")
    async def list_generations(limit: int = 50) -> dict[str, Any]:
        # Üretim geçmişi: başarılı üretim isteklerini (panel + harici API) süreleriyle birlikte
        # döndürür ve her isteğin ürettiği görselleri PNG/JPG varyantlarına göre tekilleştirerek ekler.
        generation_endpoints = {
            "/v1/generate",
            "/v1/gemini/generate",
            "/v1/gemini/stream",
        }
        logs = [
            log
            for log in store.list_request_logs(limit=max(1, min(limit * 4, 1000)))
            if log.ok
            and log.endpoint in generation_endpoints
            and not (log.output_type or "").endswith("_generation_attempt")
            and log.job_id
        ]
        logs = logs[: max(1, min(limit, 200))]
        request_ids = [log.job_id for log in logs]
        media_by_request: dict[str, list[Any]] = {}
        for item in store.list_media_outputs_by_request_ids(request_ids):
            media_by_request.setdefault(item.request_id, []).append(item)

        generations: list[dict[str, Any]] = []
        for log in logs:
            grouped: dict[str, dict[str, Any]] = {}
            order: list[str] = []
            for item in media_by_request.get(log.job_id, []):
                if item.kind != "image":
                    continue
                metadata = item.metadata or {}
                variant_key = metadata.get("variant_key") or f"{item.id}"
                image_format = metadata.get("image_format")
                if variant_key not in grouped:
                    grouped[variant_key] = {"title": item.title, "png_url": None, "jpg_url": None}
                    order.append(variant_key)
                url = _media_content_url(item)
                if image_format == "jpeg":
                    grouped[variant_key]["jpg_url"] = url
                elif image_format == "png":
                    grouped[variant_key]["png_url"] = url
                else:
                    grouped[variant_key]["png_url"] = grouped[variant_key]["png_url"] or url
            images = []
            for key in order:
                entry = grouped[key]
                entry["url"] = entry["png_url"] or entry["jpg_url"]
                images.append(entry)
            generations.append(
                {
                    "request_id": log.job_id,
                    "time": log.time,
                    "duration_ms": log.duration_ms,
                    "model": log.model,
                    "account_name": log.account_name,
                    "account_id": log.account_id,
                    "endpoint": log.endpoint,
                    "media_count": log.media_count,
                    "images": images,
                }
            )
        return {"generations": generations}

    @app.get("/v1/gemini/media/{media_token}/content")
    async def gemini_media_content(media_token: str) -> Response:
        item = store.get_media_output_by_token(media_token)
        if item is None:
            raise HTTPException(status_code=404, detail="Media not found.")
        if item.local_path:
            local_path = Path(item.local_path)
            if local_path.is_file():
                return FileResponse(
                    local_path,
                    media_type=item.local_content_type or "application/octet-stream",
                )
        if not _media_host_allowed(item.url):
            raise HTTPException(status_code=400, detail="Media host is not allowed.")
        account = store.get_account(item.account_id) if item.account_id else None
        cookies = account.cookies if account else None
        async with AsyncSession(timeout=120) as client:
            response = await client.get(item.url, allow_redirects=True, cookies=cookies)
        content_type = response.headers.get("content-type") or "application/octet-stream"
        if not _media_content_type_allowed(item.kind, content_type):
            raise HTTPException(
                status_code=502,
                detail=f"Media source returned {content_type}, not {item.kind} content.",
            )
        content = response.content
        if len(content) > MEDIA_CONTENT_MAX_BYTES:
            raise HTTPException(status_code=413, detail="Media file is too large.")
        return Response(
            content=content,
            media_type=content_type,
        )

    @app.get("/v1/gemini/jobs")
    async def gemini_jobs(limit: int = 80, job_type: str | None = None) -> dict[str, Any]:
        return {
            "jobs": [
                _job_dict(job)
                for job in store.list_jobs(limit=max(1, min(limit, 500)), job_type=job_type)
            ]
        }

    @app.get("/v1/gemini/files")
    async def list_files(limit: int = 80) -> dict[str, Any]:
        return {"files": [_file_dict(item) for item in store.list_files(limit=limit)]}

    async def _save_uploaded_file(file: UploadFile) -> Any:
        # Yüklenen dosyalar data/uploads konumuna yazılır; OpenAI uyumlu ve Gemini yerel arayüzü aynı file_id'yi paylaşır.
        file_id = f"file-{uuid.uuid4().hex}"
        upload_dir = Path(config.database_path).resolve().parent / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename or "upload").suffix
        safe_name = f"{file_id}{suffix}"
        dest = upload_dir / safe_name
        data = await file.read()
        dest.write_bytes(data)
        store.add_file(
            file_id=file_id,
            filename=file.filename or safe_name,
            content_type=file.content_type,
            path=str(dest),
            size=len(data),
        )
        return store.get_file(file_id)

    @app.post("/v1/gemini/files")
    async def upload_file(file: UploadFile = File(...)) -> dict[str, Any]:
        record = await _save_uploaded_file(file)
        return {
            "ok": True,
            "file": _file_dict(record),
        }

    async def _gem_arg(client: Any, request: GeminiGenerateRequest) -> str | None:
        if request.gem:
            return request.gem
        # Gem id'leri hesaba özeldir; aynı gem her hesapta farklı bir id'ye sahiptir. Bu yüzden
        # önce ada göre, çalışan hesabın kendi gem id'sine çözümleriz; böylece rotator hangi hesaba
        # geçerse geçsin doğru gem kullanılır. Ad bulunamazsa gem_id'ye geri düşeriz.
        if request.gem_name:
            gems = await client.fetch_gems()
            gem = gems.get(name=request.gem_name)
            if gem is not None:
                return gem.id
        if request.gem_id:
            return request.gem_id
        return None

    def _file_paths(file_ids: list[str]) -> list[str]:
        paths: list[str] = []
        for file_id in file_ids:
            record = store.get_file(file_id)
            if record is None:
                raise ValueError(f"File not found: {file_id}")
            paths.append(record.path)
        return paths

    def _media_file_suffix(kind: str, content_type: str | None, url: str) -> str:
        media_type = (content_type or "").split(";", 1)[0].strip()
        suffix = mimetypes.guess_extension(media_type) if media_type else None
        if suffix:
            return suffix
        parsed_suffix = Path(urlparse(url).path).suffix
        if parsed_suffix:
            return parsed_suffix[:16]
        return {
            "image": ".png",
            "web_image": ".png",
            "video": ".mp4",
            "audio": ".mp3",
        }.get(kind, ".bin")

    def _generated_image_map(output: Any) -> dict[str, Any]:
        # image_id -> GeneratedImage nesnesi. Sunucu, tam boyutlu (full-size) orijinal görseli
        # çözmek için bu nesnelerin client_ref/cid/rid/rcid/image_id alanlarına ihtiyaç duyar.
        result: dict[str, Any] = {}
        candidate = output.candidates[output.chosen] if output.candidates else None
        for image in getattr(candidate, "generated_images", []) or []:
            image_id = getattr(image, "image_id", None)
            if image_id:
                result[image_id] = image
        return result

    async def _download_full_size_generated_image(image_obj: Any) -> dict[str, Any]:
        # Üretilen görselin önizleme (düşük çözünürlük) yerine tam boyutlu orijinalini indirir.
        # GeneratedImage.save() kütüphanenin RPC tabanlı tam-boyut çözümleme mantığını kullanır.
        try:
            tmp_dir = Path(config.database_path).resolve().parent / "media-cache" / "_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            saved_path = await image_obj.save(
                path=str(tmp_dir), full_size=True, verbose=False
            )
            file_path = Path(saved_path)
            content = file_path.read_bytes()
            content_type = (
                mimetypes.guess_type(file_path.name)[0] or "image/png"
            )
            try:
                file_path.unlink()
            except OSError:
                pass
            if not content or len(content) > MEDIA_CONTENT_MAX_BYTES:
                return {}
            return {
                "content": content,
                "content_type": content_type,
                "size": len(content),
            }
        except Exception as exc:
            logger.warning(f"Tam boyutlu görsel indirilemedi, önizlemeye düşülüyor: {exc}")
            return {}

    def _strip_watermark_bytes(content: bytes, output_format: str = "PNG") -> dict[str, Any]:
        # Üretilen her görselin Gemini filigranı, kaydedilmeden önce kaldırılır. Filigran kaldırma
        # ters alfa harmanlamayla görselin yerel (native) çözünürlüğünde çalışır; bu yüzden bu adım
        # yalnızca tam-boyut indirme sonrası uygulanır. pillow/numpy yoksa adım atlanır.
        try:
            from ..utils.remover import remove_watermark_bytes
        except Exception as exc:
            logger.warning(
                "Filigran kaldırma atlandı: 'pillow' ve 'numpy' kurulu değil "
                f"(pip install -e '.[server]'). Ayrıntı: {exc}"
            )
            return {}
        fmt = (output_format or "PNG").upper()
        if fmt == "JPG":
            fmt = "JPEG"
        content_type = "image/jpeg" if fmt == "JPEG" else "image/png"
        try:
            cleaned = remove_watermark_bytes(content, output_format=fmt, quality=92)
            return {"content": cleaned, "content_type": content_type, "size": len(cleaned)}
        except Exception as exc:
            logger.warning(f"Filigran kaldırma başarısız oldu: {exc}")
            return {}

    def _full_size_image_url(url: str) -> str:
        # Üretilen görsel URL'sini önizleme yerine tam boyuta yükseltir.
        if "=s0" in url:
            return url
        if "=s1024-rj" in url:
            return url.replace("=s1024-rj", "=s0")
        if "=s2048-rj" in url:
            return url.replace("=s2048-rj", "=s0")
        return f"{url}=s0"

    def _download_media_item(item: dict[str, Any], full_size: bool = False) -> dict[str, Any]:
        url = item.get("url") or ""
        if not url or not _media_host_allowed(url):
            return {}
        # Üretilen görsellerde RPC tabanlı tam-boyut yolu kullanılamadığında bile önizleme yerine
        # çözünürlük sınırı olmadan tam boyutu indir.
        if full_size and item.get("kind") == "image":
            url = _full_size_image_url(url)
            item["url"] = url
        try:
            account = store.get_account(rotator.status()["current_account_id"])
            cookies = account.cookies if account else None
            with httpx.Client(timeout=120, follow_redirects=True, cookies=cookies) as client:
                response = client.get(url)
                response.raise_for_status()
            content_type = response.headers.get("content-type")
            if not _media_content_type_allowed(item.get("kind"), content_type):
                return {}
            content = response.content
            if len(content) > MEDIA_CONTENT_MAX_BYTES:
                return {}
            return {
                "content": content,
                "content_type": content_type,
                "size": len(content),
            }
        except Exception as exc:
            logger.warning(f"Medya indirilemedi ({item.get('kind')}): {exc}")
            return {}

    def _cache_downloaded_media(item: dict[str, Any], downloaded: dict[str, Any]) -> dict[str, Any]:
        content = downloaded.get("content")
        if not content:
            return {}
        try:
            url = item.get("url") or ""
            content_type = downloaded.get("content_type")
            suffix = _media_file_suffix(item.get("kind", "media"), content_type, url)
            cache_dir = Path(config.database_path).resolve().parent / "media-cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            dest = cache_dir / f"{uuid.uuid4().hex}{suffix}"
            dest.write_bytes(content)
            return {
                "path": str(dest),
                "content_type": content_type,
                "size": downloaded.get("size") or len(content),
            }
        except Exception:
            return {}

    async def _upload_media_to_object_storage(
        item: dict[str, Any],
        downloaded: dict[str, Any],
    ) -> dict[str, Any]:
        settings = _merge_system_settings(
            store.get_json_state(SYSTEM_SETTINGS_KEY, DEFAULT_SYSTEM_SETTINGS)
        )
        storage_config = ObjectStorageConfig.from_dict(settings["object_storage"])
        content = downloaded.get("content")
        if not storage_config.usable() or not content:
            return {}
        content_type = downloaded.get("content_type") or "application/octet-stream"
        category = {
            "image": "gemini/images",
            "web_image": "gemini/images",
            "video": "gemini/videos",
            "audio": "gemini/audio",
        }.get(item.get("kind"), "gemini/media")
        key = build_media_object_key(
            prefix=storage_config.prefix,
            category=category,
            data=content,
            content_type=content_type,
            source_url=item.get("url") or "",
        )
        return await upload_s3_compatible(
            config=storage_config,
            key=key,
            data=content,
            content_type=content_type,
        )

    async def _save_media_index(
        *,
        request_id: str,
        account_id: int | None,
        output: Any,
        store_media: bool = False,
        image_format: str = "PNG",
    ) -> int:
        count = 0
        image_objs = _generated_image_map(output)
        for item in _media_entries(output):
            downloaded: dict[str, Any] = {}
            is_generated_image = item.get("kind") == "image"
            if is_generated_image:
                image_obj = image_objs.get(item.get("image_id"))
                if image_obj is not None:
                    downloaded = await _download_full_size_generated_image(image_obj)
                    if downloaded and image_obj.url:
                        item["url"] = image_obj.url
            if not downloaded:
                downloaded = _download_media_item(item, full_size=is_generated_image)
            # Üretilen görsellerde filigranı kaydetmeden önce kaldır (CPU işini ayrı iş parçacığında çalıştır).
            if downloaded and item.get("kind") == "image":
                cleaned = await asyncio.to_thread(
                    _strip_watermark_bytes, downloaded["content"], image_format
                )
                if cleaned:
                    downloaded = {**downloaded, **cleaned}
            cache: dict[str, Any] = {}
            storage: dict[str, Any] = {}
            if store_media and downloaded:
                try:
                    storage = await _upload_media_to_object_storage(item, downloaded)
                except Exception as exc:
                    storage = {"error": str(exc)}
            if not storage.get("url"):
                cache = _cache_downloaded_media(item, downloaded)
            metadata = {
                **item,
                "original_url": item["url"],
            }
            if storage:
                metadata["object_storage"] = storage
            store.add_media_output(
                request_id=request_id,
                account_id=account_id,
                kind=item["kind"],
                title=item.get("title"),
                url=storage.get("url") or item["url"],
                thumbnail=item.get("thumbnail"),
                local_path=cache.get("path"),
                local_content_type=cache.get("content_type"),
                local_size=cache.get("size"),
                metadata=metadata,
            )
            count += 1
        return count

    @app.post("/v1/gemini/generate")
    async def gemini_generate(request: GeminiGenerateRequest) -> dict[str, Any]:
        request_id = f"req-{uuid.uuid4().hex}"
        try:
            generation_mode = _generation_mode_arg(request.mode)
            resolved_model = _resolve_model_arg(request.model)

            async def operation(client):
                kwargs: dict[str, Any] = {
                    "temporary": request.temporary,
                    "deep_research": request.deep_research,
                }
                if resolved_model:
                    kwargs["model"] = resolved_model
                if generation_mode:
                    kwargs["generation_mode"] = generation_mode
                gem_arg = await _gem_arg(client, request)
                if gem_arg:
                    kwargs["gem"] = gem_arg
                files = _file_paths(request.file_ids)
                if files:
                    kwargs["files"] = files
                output = await client.generate_content(
                    _effective_prompt(request.prompt, gem_arg), **kwargs
                )
                _ensure_media_generation_result(output, generation_mode)
                return output

            output = await rotator.run(
                operation,
                endpoint="/v1/gemini/generate",
                model=request.model or "gemini",
                output_type=f"gemini_{request.mode or 'native'}",
                job_id=request_id,
                require_video_generation=generation_mode == "video",
                media_generation_mode=generation_mode,
            )
        except Exception as exc:
            raise HTTPException(status_code=_error_status(exc), detail=str(exc)) from exc

        account_id = rotator.status()["current_account_id"]
        media_count = await _save_media_index(
            request_id=request_id,
            account_id=account_id,
            output=output,
            store_media=request.store_media,
        )
        if media_count:
            store.update_request_log_media_count(request_id, media_count)
        job = None
        if output.deep_research_plan:
            job_id = output.deep_research_plan.research_id or f"dr-{uuid.uuid4().hex}"
            store.upsert_job(
                job_id=job_id,
                job_type="deep_research",
                state="planned",
                account_id=account_id,
                model=request.model or "gemini",
                prompt=request.prompt,
                plan=_dump_model(output.deep_research_plan),
            )
            job = _job_dict(store.get_job(job_id))

        return {
            "ok": True,
            "account": account_id,
            "model": request.model or "gemini",
            "metadata": output.metadata,
            "output": _classified_output(output),
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "job": job,
            "request_id": request_id,
            "media_count": media_count,
        }

    @app.post("/v1/gemini/stream")
    async def gemini_stream(request: GeminiGenerateRequest):
        request_id = f"req-{uuid.uuid4().hex}"
        try:
            generation_mode = _generation_mode_arg(request.mode)
            resolved_model = _resolve_model_arg(request.model)
        except Exception as exc:
            raise HTTPException(status_code=_error_status(exc), detail=str(exc)) from exc

        async def event_stream():
            final_output = None

            async def operation(client):
                kwargs: dict[str, Any] = {
                    "temporary": request.temporary,
                    "deep_research": request.deep_research,
                }
                if resolved_model:
                    kwargs["model"] = resolved_model
                if generation_mode:
                    kwargs["generation_mode"] = generation_mode
                gem_arg = await _gem_arg(client, request)
                if gem_arg:
                    kwargs["gem"] = gem_arg
                files = _file_paths(request.file_ids)
                if files:
                    kwargs["files"] = files
                async for output in client.generate_content_stream(
                    _effective_prompt(request.prompt, gem_arg), **kwargs
                ):
                    yield output

            try:
                async for output in rotator.run_stream(
                    operation,
                    endpoint="/v1/gemini/stream",
                    model=request.model or "gemini",
                    output_type=f"gemini_{request.mode or 'native'}",
                    job_id=request_id,
                    require_video_generation=generation_mode == "video",
                    media_generation_mode=generation_mode,
                ):
                    final_output = output
                    chunk = {
                        "type": "delta",
                        "text_delta": output.text_delta,
                        "thoughts_delta": output.thoughts_delta,
                        "metadata": output.metadata,
                    }
                    yield f"data: {json.dumps(chunk).decode()}\n\n"
            except Exception as exc:
                error = {"ok": False, "error": str(exc), "status": _error_status(exc)}
                yield f"data: {json.dumps(error).decode()}\n\n"
                yield "data: [DONE]\n\n"
                return

            if final_output is not None:
                try:
                    _ensure_media_generation_result(final_output, generation_mode)
                except Exception as exc:
                    error = {"ok": False, "error": str(exc), "status": _error_status(exc)}
                    yield f"data: {json.dumps(error).decode()}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                account_id = rotator.status()["current_account_id"]
                media_count = await _save_media_index(
                    request_id=request_id,
                    account_id=account_id,
                    output=final_output,
                    store_media=request.store_media,
                )
                if media_count:
                    store.update_request_log_media_count(request_id, media_count)
                final = {
                    "type": "final",
                    "ok": True,
                    "account": account_id,
                    "model": request.model or "gemini",
                    "metadata": final_output.metadata,
                    "output": _classified_output(final_output),
                    "request_id": request_id,
                    "media_count": media_count,
                }
                yield f"data: {json.dumps(final).decode()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/v1/gemini/gems")
    async def list_gems(include_hidden: bool = False) -> dict[str, Any]:
        async def operation(client):
            return await client.fetch_gems(include_hidden=include_hidden)

        try:
            gems = await rotator.run(
                operation,
                count_usage=False,
                count_failure=False,
                endpoint="/v1/gemini/gems",
                output_type="gems",
            )
        except Exception as exc:
            cached = store.list_gems_cache()
            if cached:
                return {
                    "ok": True,
                    "cached": True,
                    "gems": [item.__dict__ for item in cached],
                    "warning": str(exc),
                }
            return {"ok": False, "cached": True, "gems": [], "warning": str(exc)}
        gem_list = [_gem_dict(gem) for gem in gems]
        store.replace_gems_cache(gem_list)
        return {"ok": True, "cached": False, "gems": gem_list}

    async def _resolve_gem_name_on_current(gem_id: str) -> str | None:
        # Gem id'si yalnızca onu listeleyen (mevcut) hesapta geçerlidir. Diğer hesaplarda aynı gem'i
        # bulabilmek için kararlı anahtar olan adı, mevcut hesaptan çözeriz.
        async def operation(client):
            gems = await client.fetch_gems()
            gem = gems.get(id=gem_id)
            return gem.name if gem else None

        try:
            return await rotator.run(
                operation,
                count_usage=False,
                count_failure=False,
                endpoint="/v1/gemini/gems",
                output_type="gem",
            )
        except Exception:
            return None

    @app.post("/v1/gemini/gems")
    async def create_gem(request: GemRequest) -> dict[str, Any]:
        # Gem'i tüm hesaplarda aynı adla oluştur; böylece rotator hangi hesaba geçerse geçsin gem mevcut olur.
        async def operation(client, account):
            return await client.create_gem(
                name=request.name,
                prompt=request.prompt,
                description=request.description,
            )

        results = await rotator.run_on_each_account(operation)
        created = [r for r in results if r["error"] is None and r["result"] is not None]
        if not created:
            errors = "; ".join(r["error"] for r in results if r["error"]) or "Aktif hesap yok."
            raise HTTPException(status_code=502, detail=f"Gem hiçbir hesapta oluşturulamadı: {errors}")
        return {
            "ok": True,
            "gem": _gem_dict(created[0]["result"]),
            "accounts_total": len(results),
            "accounts_succeeded": len(created),
        }

    @app.patch("/v1/gemini/gems/{gem_id}")
    async def update_gem(gem_id: str, request: GemRequest) -> dict[str, Any]:
        # Güncellenecek gem'i tüm hesaplarda eski adına göre bul ve yeni değerlerle güncelle (yeniden adlandırma dahil).
        old_name = await _resolve_gem_name_on_current(gem_id)

        async def operation(client, account):
            target_id = gem_id
            if old_name:
                gems = await client.fetch_gems()
                match = gems.get(name=old_name)
                if match is None:
                    return None
                target_id = match.id
            return await client.update_gem(
                gem=target_id,
                name=request.name,
                prompt=request.prompt,
                description=request.description,
            )

        results = await rotator.run_on_each_account(operation)
        updated = [r for r in results if r["error"] is None and r["result"] is not None]
        if not updated:
            errors = "; ".join(r["error"] for r in results if r["error"]) or "Eşleşen gem bulunamadı."
            raise HTTPException(status_code=502, detail=f"Gem hiçbir hesapta güncellenemedi: {errors}")
        return {
            "ok": True,
            "gem": _gem_dict(updated[0]["result"]),
            "accounts_total": len(results),
            "accounts_succeeded": len(updated),
        }

    @app.delete("/v1/gemini/gems/{gem_id}")
    async def delete_gem(gem_id: str) -> dict[str, Any]:
        # Gem'i tüm hesaplardan adına göre sil.
        old_name = await _resolve_gem_name_on_current(gem_id)

        async def operation(client, account):
            target_id = gem_id
            if old_name:
                gems = await client.fetch_gems()
                match = gems.get(name=old_name)
                if match is None:
                    return None
                target_id = match.id
            await client.delete_gem(target_id)
            return True

        results = await rotator.run_on_each_account(operation)
        deleted = [r for r in results if r["error"] is None and r["result"]]
        return {
            "ok": True,
            "accounts_total": len(results),
            "accounts_succeeded": len(deleted),
        }

    def _custom_gem_dict(gem: Any) -> dict[str, Any]:
        return {
            "id": gem.id,
            "name": gem.name,
            "prompt": gem.prompt,
            "description": gem.description,
            "is_default": gem.is_default,
            "created_at": gem.created_at,
            "updated_at": gem.updated_at,
        }

    async def _sync_custom_gem_to_accounts(
        *, name: str, prompt: str, description: str, previous_name: str | None = None
    ) -> dict[str, int]:
        # Özel gem'i tüm Gemini hesaplarına ada göre uygular: hesapta varsa günceller, yoksa
        # oluşturur. Böylece rotator hangi hesaba düşerse düşsün, üretimde gem ada göre bulunur.
        async def operation(client, account):
            lookup = previous_name or name
            gems = await client.fetch_gems()
            match = gems.get(name=lookup)
            if match is None and previous_name:
                match = gems.get(name=name)
            if match is not None:
                return await client.update_gem(
                    gem=match.id, name=name, prompt=prompt, description=description
                )
            return await client.create_gem(
                name=name, prompt=prompt, description=description
            )

        results = await rotator.run_on_each_account(operation)
        succeeded = [r for r in results if r["error"] is None and r["result"] is not None]
        return {"accounts_total": len(results), "accounts_succeeded": len(succeeded)}

    @app.get("/v1/custom-gems")
    async def list_custom_gems() -> dict[str, Any]:
        return {
            "ok": True,
            "gems": [_custom_gem_dict(gem) for gem in store.list_custom_gems()],
        }

    @app.post("/v1/custom-gems")
    async def create_custom_gem(request: CustomGemRequest) -> dict[str, Any]:
        if not request.name.strip():
            raise HTTPException(status_code=400, detail="Gem adı boş olamaz.")
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Gem promptu boş olamaz.")
        gem = store.create_custom_gem(
            name=request.name.strip(),
            prompt=request.prompt,
            description=request.description or None,
            is_default=request.is_default,
        )
        sync = await _sync_custom_gem_to_accounts(
            name=gem.name, prompt=gem.prompt, description=gem.description or ""
        )
        return {"ok": True, "gem": _custom_gem_dict(gem), "sync": sync}

    @app.patch("/v1/custom-gems/{gem_id}")
    async def update_custom_gem(gem_id: str, request: CustomGemRequest) -> dict[str, Any]:
        existing = store.get_custom_gem(gem_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Özel gem bulunamadı.")
        if not request.name.strip():
            raise HTTPException(status_code=400, detail="Gem adı boş olamaz.")
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Gem promptu boş olamaz.")
        gem = store.update_custom_gem(
            gem_id,
            name=request.name.strip(),
            prompt=request.prompt,
            description=request.description or None,
            is_default=request.is_default,
        )
        sync = await _sync_custom_gem_to_accounts(
            name=gem.name,
            prompt=gem.prompt,
            description=gem.description or "",
            previous_name=existing.name if existing.name != gem.name else None,
        )
        return {"ok": True, "gem": _custom_gem_dict(gem), "sync": sync}

    @app.delete("/v1/custom-gems/{gem_id}")
    async def delete_custom_gem(gem_id: str) -> dict[str, Any]:
        existing = store.get_custom_gem(gem_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Özel gem bulunamadı.")
        store.delete_custom_gem(gem_id)

        async def operation(client, account):
            gems = await client.fetch_gems()
            match = gems.get(name=existing.name)
            if match is None:
                return False
            await client.delete_gem(match.id)
            return True

        results = await rotator.run_on_each_account(operation)
        deleted = [r for r in results if r["error"] is None and r["result"]]
        return {
            "ok": True,
            "accounts_total": len(results),
            "accounts_succeeded": len(deleted),
        }

    @app.post("/v1/gemini/deep-research/plan")
    async def create_deep_research_plan(
        request: DeepResearchCreateRequest,
    ) -> dict[str, Any]:
        job_id = f"dr-{uuid.uuid4().hex}"

        async def operation(client):
            resolved_model = _resolve_model_arg(request.model) or Model.UNSPECIFIED
            return await client.create_deep_research_plan(
                request.prompt,
                model=resolved_model,
            )

        try:
            plan = await rotator.run(
                operation,
                endpoint="/v1/gemini/deep-research/plan",
                model=request.model or "gemini",
                output_type="deep_research",
                job_id=job_id,
                deep_research_state="planned",
            )
        except Exception as exc:
            store.upsert_job(
                job_id=job_id,
                job_type="deep_research",
                state="failed",
                model=request.model or "gemini",
                prompt=request.prompt,
                error=str(exc),
            )
            raise HTTPException(status_code=_error_status(exc), detail=str(exc)) from exc

        if plan.research_id:
            job_id = plan.research_id
        store.upsert_job(
            job_id=job_id,
            job_type="deep_research",
            state="planned",
            account_id=rotator.status()["current_account_id"],
            model=request.model or "gemini",
            prompt=request.prompt,
            plan=_dump_model(plan),
        )
        return {
            "ok": True,
            "job": _job_dict(store.get_job(job_id)),
            "plan": _dump_model(plan),
        }

    @app.post("/v1/gemini/deep-research/start")
    async def start_deep_research(request: DeepResearchStartRequest) -> dict[str, Any]:
        if request.plan is None and request.job_id is None:
            raise HTTPException(status_code=400, detail="job_id or plan is required.")
        job = store.get_job(request.job_id) if request.job_id else None
        plan_data = request.plan or (job.plan_json if job else None)
        if not plan_data:
            raise HTTPException(status_code=404, detail="Deep research plan not found.")
        plan = DeepResearchPlan(**plan_data)
        job_id = request.job_id or plan.research_id or f"dr-{uuid.uuid4().hex}"

        async def operation(client):
            return await client.start_deep_research(
                plan,
                confirm_prompt=request.confirm_prompt,
            )

        try:
            output = await rotator.run(
                operation,
                endpoint="/v1/gemini/deep-research/start",
                output_type="deep_research",
                job_id=job_id,
                deep_research_state="running",
            )
        except Exception as exc:
            store.upsert_job(
                job_id=job_id,
                job_type="deep_research",
                state="failed",
                plan=plan_data,
                error=str(exc),
            )
            raise HTTPException(status_code=_error_status(exc), detail=str(exc)) from exc
        store.upsert_job(
            job_id=job_id,
            job_type="deep_research",
            state="running",
            account_id=rotator.status()["current_account_id"],
            plan=plan_data,
            result={"start_output": _classified_output(output), "metadata": output.metadata},
        )
        return {
            "ok": True,
            "job": _job_dict(store.get_job(job_id)),
            "output": _classified_output(output),
        }

    @app.get("/v1/gemini/deep-research/{job_id}/status")
    async def deep_research_status(job_id: str) -> dict[str, Any]:
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")

        async def operation(client):
            return await client.get_deep_research_status(job_id)

        try:
            status_obj = await rotator.run(
                operation,
                count_usage=False,
                endpoint="/v1/gemini/deep-research/status",
                output_type="deep_research",
                job_id=job_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=_error_status(exc), detail=str(exc)) from exc
        status_data = _dump_model(status_obj)
        if status_obj:
            store.upsert_job(
                job_id=job_id,
                job_type="deep_research",
                state="done" if status_obj.done else status_obj.state,
                result={"status": status_data, **(job.result_json or {})},
            )
        return {"ok": True, "job": _job_dict(store.get_job(job_id)), "status": status_data}

    @app.post("/v1/gemini/deep-research/wait")
    async def wait_deep_research(request: DeepResearchWaitRequest) -> dict[str, Any]:
        job = store.get_job(request.job_id)
        if not job or not job.plan_json:
            raise HTTPException(status_code=404, detail="Deep research plan not found.")
        plan = DeepResearchPlan(**job.plan_json)

        async def operation(client):
            return await client.wait_for_deep_research(
                plan,
                poll_interval=request.poll_interval,
                timeout=request.timeout,
            )

        try:
            result = await rotator.run(
                operation,
                endpoint="/v1/gemini/deep-research/wait",
                output_type="deep_research",
                job_id=request.job_id,
                deep_research_state="waiting",
            )
        except Exception as exc:
            store.upsert_job(
                job_id=request.job_id,
                job_type="deep_research",
                state="failed",
                error=str(exc),
            )
            raise HTTPException(status_code=_error_status(exc), detail=str(exc)) from exc
        result_data = _dump_model(result)
        store.upsert_job(
            job_id=request.job_id,
            job_type="deep_research",
            state="done" if result.done else "timeout",
            result=result_data,
        )
        return {"ok": True, "job": _job_dict(store.get_job(request.job_id)), "result": result_data}

    @app.get("/v1/accounts")
    async def list_accounts() -> dict[str, Any]:
        """Yönetim paneli ve hata ayıklama betiklerinin doğrudan okuyabileceği hesap havuzu listesini döndürür."""
        status = rotator.status()
        return {
            "ok": True,
            "current_account_id": status["current_account_id"],
            "accounts": status["accounts"],
        }

    @app.post("/v1/accounts")
    async def add_account(request: AccountRequest) -> dict[str, Any]:
        cookies = dict(request.cookies)
        cookies["__Secure-1PSID"] = request.secure_1psid
        if request.secure_1psidts:
            cookies["__Secure-1PSIDTS"] = request.secure_1psidts
        account = store.upsert_account(
            name=request.name,
            secure_1psid=request.secure_1psid,
            secure_1psidts=request.secure_1psidts,
            cookies=cookies,
            enabled=request.enabled,
        )
        validation = await rotator.validate_account(account.id)
        return {
            "ok": True,
            "validation": validation,
            "accounts": rotator.status()["accounts"],
        }

    @app.post("/v1/accounts/import")
    async def import_accounts() -> dict[str, Any]:
        imported = store.import_accounts_file(config.accounts_file)
        return {"ok": True, "imported": imported, "accounts": rotator.status()["accounts"]}

    @app.get("/v1/accounts/export")
    async def export_accounts() -> dict[str, Any]:
        return {
            "accounts": [
                {
                    "name": account.name,
                    "__Secure-1PSID": account.secure_1psid,
                    "__Secure-1PSIDTS": account.secure_1psidts,
                    "cookies": account.cookies,
                    "enabled": account.enabled,
                    "expired": account.expired,
                }
                for account in store.list_accounts()
            ]
        }

    @app.post("/v1/accounts/switch")
    async def switch_account(request: SwitchAccountRequest) -> dict[str, Any]:
        try:
            if request.account_id is None:
                account = await rotator.switch_next()
            else:
                account = await rotator.switch_to(request.account_id)
        except Exception as exc:
            raise HTTPException(status_code=_error_status(exc), detail=str(exc)) from exc
        return {"ok": True, "current_account_id": account.id, "status": rotator.status()}

    @app.post("/v1/accounts/validate")
    async def validate_current_account() -> dict[str, Any]:
        try:
            result = await rotator.validate_account()
        except Exception as exc:
            raise HTTPException(status_code=_error_status(exc), detail=str(exc)) from exc
        return {"ok": True, "validation": result, "accounts": rotator.status()["accounts"]}

    @app.post("/v1/accounts/validate-all")
    async def validate_all_accounts() -> dict[str, Any]:
        results = await rotator.validate_accounts()
        return {"ok": True, "validations": results, "accounts": rotator.status()["accounts"]}

    @app.post("/v1/accounts/{account_id}/validate")
    async def validate_account(account_id: int) -> dict[str, Any]:
        try:
            result = await rotator.validate_account(account_id)
        except Exception as exc:
            raise HTTPException(status_code=_error_status(exc), detail=str(exc)) from exc
        return {"ok": True, "validation": result, "accounts": rotator.status()["accounts"]}

    @app.post("/v1/accounts/{account_id}/media-cooldowns/clear")
    async def clear_account_media_cooldowns(
        account_id: int, request: ClearMediaCooldownRequest
    ) -> dict[str, Any]:
        if store.get_account(account_id) is None:
            raise HTTPException(status_code=404, detail="Account not found.")
        kinds = ["audio", "image", "video"] if request.kind is None else [request.kind]
        cleared: list[str] = []
        for kind in kinds:
            normalized = (kind or "").strip().lower()
            if normalized not in {"audio", "image", "video"}:
                raise HTTPException(
                    status_code=400,
                    detail="kind must be one of: image, video, audio.",
                )
            if store.clear_media_cooldown(account_id, normalized):
                cleared.append(normalized)
        return {
            "ok": True,
            "account_id": account_id,
            "cleared": cleared,
            "accounts": rotator.status()["accounts"],
        }

    @app.patch("/v1/accounts/{account_id}")
    async def update_account(
        account_id: int, request: AccountToggleRequest
    ) -> dict[str, Any]:
        if not store.set_account_enabled(account_id, request.enabled):
            raise HTTPException(status_code=404, detail="Account not found.")
        return {"ok": True, "accounts": rotator.status()["accounts"]}

    @app.delete("/v1/accounts/{account_id}")
    async def delete_account(account_id: int) -> dict[str, Any]:
        if not store.delete_account(account_id):
            raise HTTPException(status_code=404, detail="Account not found.")
        return {"ok": True, "accounts": rotator.status()["accounts"]}

    @app.post("/v1/auth/session")
    async def start_auth_session() -> dict[str, Any]:
        try:
            return await auth_browser.start_session()
        except AuthBrowserUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.delete("/v1/auth/session")
    async def close_auth_session() -> dict[str, Any]:
        await auth_browser.close_session()
        return {"ok": True}

    @app.get("/v1/auth/screenshot")
    async def auth_screenshot() -> Response:
        try:
            image = await auth_browser.screenshot()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=image, media_type="image/png")

    @app.post("/v1/auth/click")
    async def auth_click(request: AuthClickRequest) -> dict[str, Any]:
        try:
            return await auth_browser.click(request.x, request.y)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/auth/type")
    async def auth_type(request: AuthTypeRequest) -> dict[str, Any]:
        try:
            return await auth_browser.type_text(request.text)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/auth/press")
    async def auth_press(request: AuthPressRequest) -> dict[str, Any]:
        try:
            return await auth_browser.press(request.key)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/auth/save")
    async def auth_save(request: AuthSaveRequest) -> dict[str, Any]:
        try:
            result = await auth_browser.save_account(name=request.name)
            validation = await rotator.validate_account(result["account_id"])
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {**result, "validation": validation, "accounts": rotator.status()["accounts"]}

    async def _resolve_custom_gem_for_generation(
        client: Any, gem_id: str | None
    ) -> str | None:
        # Özel gem DB'sindeki id'yi, çalışan hesabın kendi gem id'sine çözer. Hesapta o adda
        # gem yoksa otomatik oluşturur; böylece rotator hangi hesaba düşerse düşsün üretimde gem
        # her zaman mevcut olur.
        if not gem_id:
            record = store.get_default_custom_gem()
            if record is None:
                return None
        else:
            record = store.get_custom_gem(gem_id)
            if record is None:
                raise ValueError(f"Özel gem bulunamadı: {gem_id}")
        gems = await client.fetch_gems()
        match = gems.get(name=record.name)
        if match is not None:
            return match.id
        created = await client.create_gem(
            name=record.name,
            prompt=record.prompt,
            description=record.description or "",
        )
        return created.id

    async def _save_generate_media(
        *,
        request_id: str,
        account_id: int | None,
        output: Any,
        output_format: str,
        store_media: bool,
    ) -> list[dict[str, Any]]:
        # Üretilen her görseli filigransız olarak hem PNG hem JPG biçiminde CDN'e (veya yerel
        # önbelleğe) yazar ve içerik bağlantılarını döndürür.
        image_objs = _generated_image_map(output)
        results: list[dict[str, Any]] = []
        formats = [("png", "PNG"), ("jpeg", "JPEG")]
        # İstenen biçim ilk sırada dönsün.
        if output_format == "jpeg":
            formats = [("jpeg", "JPEG"), ("png", "PNG")]

        for image_index, item in enumerate(_media_entries(output)):
            if item.get("kind") != "image":
                continue
            variant_key = f"{request_id}#{image_index}"
            source: dict[str, Any] = {}
            image_obj = image_objs.get(item.get("image_id"))
            if image_obj is not None:
                source = await _download_full_size_generated_image(image_obj)
                if source and image_obj.url:
                    item["url"] = image_obj.url
            if not source:
                source = _download_media_item(item, full_size=True)
            if not source.get("content"):
                continue

            variants: dict[str, Any] = {}
            for key, fmt in formats:
                cleaned = await asyncio.to_thread(
                    _strip_watermark_bytes, source["content"], fmt
                )
                if not cleaned:
                    continue
                variant_item = {**item}
                storage: dict[str, Any] = {}
                cache: dict[str, Any] = {}
                if store_media:
                    try:
                        storage = await _upload_media_to_object_storage(variant_item, cleaned)
                    except Exception as exc:
                        storage = {"error": str(exc)}
                if not storage.get("url"):
                    cache = _cache_downloaded_media(variant_item, cleaned)
                metadata = {
                    **variant_item,
                    "original_url": item.get("url", ""),
                    "image_format": key,
                    # PNG ve JPG aynı kaynağa ait; geçmişte tek görsel olarak gruplanır.
                    "variant_key": variant_key,
                }
                if storage:
                    metadata["object_storage"] = storage
                media_id = store.add_media_output(
                    request_id=request_id,
                    account_id=account_id,
                    kind="image",
                    title=item.get("title"),
                    url=storage.get("url") or item.get("url", ""),
                    thumbnail=item.get("thumbnail"),
                    local_path=cache.get("path"),
                    local_content_type=cache.get("content_type"),
                    local_size=cache.get("size"),
                    metadata=metadata,
                )
                content_url = storage.get("url")
                if not content_url:
                    saved = store.get_media_output(media_id)
                    if saved and saved.token:
                        content_url = f"/v1/gemini/media/{saved.token}/content"
                variants[key] = content_url

            if variants:
                results.append(
                    {
                        "title": item.get("title"),
                        "png_url": variants.get("png"),
                        "jpg_url": variants.get("jpeg"),
                        "url": variants.get(output_format) or next(iter(variants.values())),
                    }
                )
        return results

    @app.post("/v1/generate")
    async def generate(request: GenerateRequest):
        try:
            generation_mode = _generation_mode_arg(request.mode) or "image"
            resolved_model = _resolve_model_arg(request.model)
            aspect_ratio = _aspect_ratio_arg(request.aspect_ratio)
            output_format = _output_format_arg(request.output_format)
        except Exception as exc:
            raise HTTPException(status_code=_error_status(exc), detail=str(exc)) from exc

        request_id = f"gen-{uuid.uuid4().hex}"
        prompt = _apply_aspect_ratio(request.prompt, aspect_ratio)

        async def event_stream():
            yield f"event: status\ndata: {json.dumps({'stage': 'generating', 'request_id': request_id}).decode()}\n\n"

            async def operation(client):
                kwargs: dict[str, Any] = {
                    "temporary": request.temporary,
                    "generation_mode": generation_mode,
                }
                if resolved_model:
                    kwargs["model"] = resolved_model
                try:
                    gem_arg = await _resolve_custom_gem_for_generation(
                        client, request.gem_id
                    )
                except ValueError:
                    raise
                if gem_arg:
                    kwargs["gem"] = gem_arg
                files = _file_paths(request.file_ids)
                if files:
                    kwargs["files"] = files
                output = await client.generate_content(prompt, **kwargs)
                _ensure_media_generation_result(output, generation_mode)
                return output

            try:
                output = await rotator.run(
                    operation,
                    endpoint="/v1/generate",
                    model=request.model or "gemini",
                    output_type=f"gemini_{generation_mode}",
                    job_id=request_id,
                    require_video_generation=generation_mode == "video",
                    media_generation_mode=generation_mode,
                )
            except Exception as exc:
                payload = {"stage": "error", "detail": str(exc), "status": _error_status(exc)}
                yield f"event: error\ndata: {json.dumps(payload).decode()}\n\n"
                return

            account_id = rotator.status()["current_account_id"]
            try:
                images = await _save_generate_media(
                    request_id=request_id,
                    account_id=account_id,
                    output=output,
                    output_format=output_format,
                    store_media=request.store_media,
                )
            except Exception as exc:
                payload = {"stage": "error", "detail": str(exc), "status": 502}
                yield f"event: error\ndata: {json.dumps(payload).decode()}\n\n"
                return

            if images:
                store.update_request_log_media_count(request_id, len(images))

            result = {
                "stage": "done",
                "request_id": request_id,
                "account": account_id,
                "model": request.model or "gemini",
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
                "text": output.text,
                "images": images,
            }
            yield f"event: result\ndata: {json.dumps(result).decode()}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    config = ServerConfig.from_env()
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
    )


if __name__ == "__main__":
    main()
