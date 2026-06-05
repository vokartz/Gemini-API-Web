class AuthError(Exception):
    """
    Exception for authentication errors caused by invalid credentials/cookies.
    """

    pass


class APIError(Exception):
    """
    Exception for package-level errors which need to be fixed in the future development (e.g. validation errors).
    """

    pass


class VideoGenerationNotSubmitted(APIError):
    """
    Raised when a video request ended before Gemini confirmed a saved chat/job.
    """

    pass


class VideoGenerationFailed(APIError):
    """
    Raised when Gemini saved a video request but ended it without video media.
    """

    pass


class ImageGenerationError(APIError):
    """
    Exception for generated image parsing errors.
    """

    pass


class MediaGenerationEmptyResult(APIError):
    """
    Medya oluşturma isteği üst sunucudan döndü, ancak yanıtta karşılık gelen resim, video veya ses sonucu yok.
    """

    pass


class GeminiError(Exception):
    """
    Exception for errors returned from Gemini server which are not handled by the package.
    """

    pass


class TimeoutError(GeminiError):
    """
    Exception for request timeouts.
    """

    pass


class UsageLimitExceeded(GeminiError):
    """
    Exception for model usage limit exceeded errors.
    """

    pass


class MediaGenerationTemporarilyUnavailable(GeminiError):
    """
    Raised when all accounts are cooling down for a media generation mode.
    """

    pass


class ModelInvalid(GeminiError):
    """
    Exception for invalid model header string errors.
    """

    pass


class TemporarilyBlocked(GeminiError):
    """
    Exception for 429 Too Many Requests when IP is temporarily blocked.
    """

    pass
