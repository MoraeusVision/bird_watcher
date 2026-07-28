# AI Coding Instructions

## General Principles

- Target Python 3.12 or newer.
- Follow the YAGNI principle (You Aren't Gonna Need It). Do not add abstractions or features until they are required.
- Follow the KISS principle (Keep It Simple, Stupid). Prefer the simplest solution that solves the problem.
- Write clean, readable, and maintainable code.
- Refactor instead of overengineering.
- Avoid duplicated code (DRY), but do not create unnecessary abstractions.

---

## Code Style

- Format code with Black.
- Follow Ruff linting rules.
- Use meaningful variable, function, and class names.
- Keep functions small and focused on a single responsibility.
- Prefer composition over inheritance when appropriate.
- Avoid global mutable state.

---

## Typing

- Use type hints everywhere.
- Prefer built-in generic types (`list[str]`, `dict[str, int]`, etc.).
- Avoid `Any` unless absolutely necessary.
- Return explicit types.
- Use dataclasses for structured data when appropriate.

Example:

```python
def detect_people(frame: np.ndarray) -> list[Person]:
    ...
```

---

## Logging

- Use the standard `logging` module.
- Never use `print()` for debugging or runtime information.
- Log meaningful events.
- Use appropriate log levels:
    - DEBUG
    - INFO
    - WARNING
    - ERROR
    - CRITICAL
- Log exceptions with stack traces.

Example:

```python
logger.exception("Failed to load model")
```

---

## Error Handling

- Fail early.
- Raise meaningful exceptions.
- Do not silently ignore exceptions.
- Catch only exceptions that can be handled.

---

## Comments

- Write comments in English.
- Explain *why*, not *what*.
- Prefer self-documenting code over comments.

---

## Documentation

- Write concise docstrings for public classes and functions.
- Follow Google-style docstrings.

---

## Testing

- Use pytest.
- Prefer Test Driven Development (TDD) when practical.
- Write tests for new functionality.
- Keep tests deterministic.

---

## Design Patterns

Use design patterns only when they simplify the code.

Prefer:
- Strategy
- Factory
- Dependency Injection
- Observer
- Builder (for complex object creation)

Avoid unnecessary abstraction.

---

## Project Structure

Keep modules focused.

Example:

```
project/
    core/
    detection/
    tracking/
    visualization/
    utils/
    tests/
```

---

## Performance

- Prefer readable code over micro-optimizations.
- Optimize only after measuring.
- Avoid premature optimization.

---

## AI Assistant Behaviour

When generating code:

- **Always discuss ideas and proposed solutions with the user before writing any code.** Present your approach, trade-offs, and alternatives first, and wait for confirmation before proceeding.
- Follow existing project architecture.
- Reuse existing code before creating new functionality.
- Do not introduce new dependencies unless necessary.
- Keep changes as small as possible.
- Explain trade-offs when multiple solutions exist.
- If uncertain, choose the simpler implementation.

---

## YAGNI/KISS Decision Rules (Practical)

When multiple implementations are possible, prefer the smallest working design first.

- Do not add extension points (callbacks, dependency injection, strategy classes) unless at least one real second implementation is needed now.
- Keep behavior inside the class/function first. Extract or inject only when current requirements demand variation.
- Avoid optional function parameters that default to `None` when a simple internal method is enough.
- Start with one clear flow. Add abstraction only after repeated code or real branching requirements appear.
- Prefer explicit, concrete types over generic placeholders unless integration constraints require flexibility.
- If a simpler solution and a flexible solution are both valid, implement the simpler one and document how to extend later.

Examples:

- Prefer internal `on_frame` / `on_prediction` methods over injected callbacks until external customization is actually required.
- Prefer one concrete app class over base classes + subclasses until multiple variants exist in production code.
