import logging
import platform
from time import sleep

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _create_led():
    """Create a hardware LED on GPIO pin 17, or None in simulation mode."""
    if platform.system() == "Linux":
        from gpiozero import LED  # noqa: PLC0415

        logger.info("Raspberry Pi mode")
        return LED(17)

    logger.info("Simulation mode")
    return None


def run() -> None:
    led = _create_led()

    try:
        while True:
            if led is not None:
                led.toggle()
            else:
                logger.debug("LED toggle (simulated)")
            sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        if led is not None:
            led.close()

if __name__ == "__main__":
    run()
