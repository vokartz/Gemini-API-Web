"""Gemini watermark remover (reverse alpha blending).

Exposes the helpers from :mod:`gemini_watermark_remover` so the package can be
imported as ``gemini_webapi.utils.remover``. Requires the optional ``pillow`` and
``numpy`` dependencies (installed via the ``server`` extra).
"""

from .gemini_watermark_remover import (
    remove_watermark,
    remove_watermark_bytes,
)

__all__ = ["remove_watermark", "remove_watermark_bytes"]
