"""Ursina editor camera without per-frame zoom drift (reduces view jitter)."""

from __future__ import annotations

from ursina import EditorCamera, camera


class StableEditorCamera(EditorCamera):
    """
    EditorCamera that only interpolates zoom while scrolling.

    Ursina's default EditorCamera lerps camera.z every frame, which can cause
    shimmering/z-fighting-style jitter when orbiting over large ground planes.
    """

    def __init__(self, **kwargs):
        self._zoom_active = False
        super().__init__(**kwargs)

    def input(self, key):
        if key in ("scroll up", "scroll down"):
            self._zoom_active = True
        super().input(key)

    def update(self):
        super().update()
        if camera.orthographic:
            return
        if self._zoom_active:
            if abs(camera.z - self.target_z) < 0.25:
                camera.z = self.target_z
                self._zoom_active = False
            return
        camera.z = self.target_z
