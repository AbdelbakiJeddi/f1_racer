"""Keyboard teleoperation module for manual driving in Isaac Sim."""

from __future__ import annotations

from typing import Tuple, Set, Optional


class KeyboardTeleop:
    """Listens to Omniverse window keyboard events for manual vehicle driving."""

    def __init__(self):
        import carb.input
        import omni.appwindow

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        self.steer_cmd: float = 0.0      # [-1.0, 1.0]
        self.throttle_cmd: float = -1.0  # [-1.0, 1.0] (-1.0 is stopped / 0 speed)
        self.reset_requested: bool = False

        self._keys_pressed: Set[str] = set()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key_event)

    def _on_key_event(self, event, *args, **kwargs) -> bool:
        import carb.input

        key = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._keys_pressed.add(key)
            if key == "R":
                self.reset_requested = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._keys_pressed.discard(key)
        return True

    def update(self) -> Tuple[float, float, bool]:
        """Returns (steer [-1, 1], throttle [-1, 1], reset_requested)."""
        # Steering input
        steer_target = 0.0
        if "A" in self._keys_pressed or "LEFT" in self._keys_pressed:
            steer_target += 1.0  # Turn left (+steer)
        if "D" in self._keys_pressed or "RIGHT" in self._keys_pressed:
            steer_target -= 1.0  # Turn right (-steer)

        # Smooth steering filter
        self.steer_cmd = 0.7 * self.steer_cmd + 0.3 * steer_target

        # Throttle input: -1.0 = stopped, +1.0 = full forward (2.5 m/s)
        if "SPACE" in self._keys_pressed:
            self.throttle_cmd = -1.0  # Handbrake
        elif "W" in self._keys_pressed or "UP" in self._keys_pressed:
            self.throttle_cmd = min(1.0, self.throttle_cmd + 0.08)
        elif "S" in self._keys_pressed or "DOWN" in self._keys_pressed:
            self.throttle_cmd = max(-1.0, self.throttle_cmd - 0.12)
        else:
            # Coast down to stop when no throttle pressed
            self.throttle_cmd = max(-1.0, self.throttle_cmd - 0.05)

        do_reset = self.reset_requested
        self.reset_requested = False
        return self.steer_cmd, self.throttle_cmd, do_reset

    def close(self) -> None:
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None
