# AI Coding Instructions

## Keep It Small

- Prefer the simplest working change.
- Do not add abstractions, callbacks, or extra layers unless the repo already needs them.
- Reuse existing code paths before creating new ones.
- Keep edits narrowly scoped to the file or behavior being changed.

## Python Style

- Use type hints for public functions, methods, and dataclass fields.
- Prefer built-in generic types like `list[str]` and `dict[str, int]`.
- Use `logging`, not `print()`, for runtime output.
- Keep comments in English and only when the code is not self-explanatory.

## Project Shape

- The app is centered in [main.py](../main.py) with camera capture, detection, classification, and app wiring.
- Shared helpers live in [utils.py](../utils.py), including platform checks, image cropping, and LED handling.
- Heavy dependencies such as RF-DETR, Transformers, Picamera2, and gpiozero should be treated as runtime-only and mocked in tests.

## Raspberry Pi Camera Path

- On Linux, the camera path uses Picamera2.
- Follow the working pattern from [simple_picamera_check.py](../simple_picamera_check.py): configure the preview before `start()`.
- If a frame is passed to Transformers, convert the OpenCV BGR array to RGB and wrap it with `PIL.Image.fromarray(...)` first.

## LED Handling

- Keep LED logic in [utils.py](../utils.py).
- Use the `LedManager` helper for turning LED on/off and closing it.
- Do not import `gpiozero` in tests; mock the LED helper instead.

## Testing

- Use `pytest`.
- Keep tests deterministic and isolated from hardware and model downloads.
- For `main.py`, prefer stubbing modules and objects at import time when third-party packages are heavy or unavailable.
- After behavior changes, run the narrowest relevant test first, then the full test suite if needed.

## Change Discipline

- When asked for advice, discuss the approach before editing.
- When asked to make the change, implement the smallest correct fix and verify it.
- If the codebase already has a working pattern, follow it instead of inventing a new one.
