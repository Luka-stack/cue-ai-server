from datetime import UTC, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.core.exceptions import InvalidTimezoneError


def resolve_zone(tz: str | None) -> tzinfo:
    """IANA zone name (e.g. ``Europe/Warsaw``) to a tzinfo; UTC when not given."""
    if not tz:
        return UTC
    try:
        return ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError):
        raise InvalidTimezoneError(tz) from None
