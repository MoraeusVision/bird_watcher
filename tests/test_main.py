import importlib
from unittest.mock import patch

from main import _create_led


def test_create_led_returns_none_on_non_linux() -> None:
    with patch("utils.get_platform", return_value="Darwin"):
        import main

        importlib.reload(main)
        result = main._create_led()

    assert result is None


def test_platform_uses_get_platform() -> None:
    with patch("utils.get_platform", return_value="Linux"):
        import main

        importlib.reload(main)

    assert main.PLATFORM == "Linux"
