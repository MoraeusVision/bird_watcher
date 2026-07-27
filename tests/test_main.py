import platform
from unittest.mock import patch

from main import _create_led


def test_create_led_returns_none_on_non_linux() -> None:
    with patch.object(platform, "system", return_value="Darwin"):
        result = _create_led()

    assert result is None
