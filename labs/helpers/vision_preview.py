"""Data sources and display utilities for the lane-detector labs. Contains no detection logic."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Iterator

import cv2
import matplotlib.pyplot as plt
import numpy as np

LAB_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = LAB_DIR / "data"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

YELLOW = (255, 209, 0)
ASPHALT = (31, 31, 31)
BACKGROUND = (62, 68, 82)


# ----------------------------------------------------------------------------- synthetic camera


@dataclass
class SyntheticTrack:
    """A flat track seen by a level pinhole camera.

    Pixel column c covers u in [c, c+1); row r covers v in [r, r+1). The optical centre is u = width/2.
    Track geometry in metres (car frame): x lateral (+ right), z forward.
    """

    width: int = 128
    height: int = 128
    focal: float = 64.0  # px, ~90 deg horizontal field of view at 128 px
    cam_height: float = 0.25  # m above the ground
    horizon: float = 0.28  # horizon row as a fraction of image height
    half_width: float = 0.30  # track half-width, m (centre to line centre)
    offset: float = 0.0  # track centre lateral position at the car, m
    yaw: float = 0.0  # track direction relative to the car heading, rad
    curvature: float = 0.0  # 1/m, + bends right
    line_width: float = 0.03  # painted line width, m
    draw_left: bool = True
    draw_right: bool = True
    brightness: float = 1.0
    noise: float = 0.0  # gaussian noise std, 0..255 units
    asphalt: tuple = ASPHALT
    line_color: tuple = YELLOW
    background: tuple = BACKGROUND

    @property
    def v_horizon(self) -> float:
        return self.horizon * self.height

    def distance(self, v: float) -> float:
        return self.focal * self.cam_height / (v - self.v_horizon)

    def centre_x(self, z: float) -> float:
        """Ground truth: lateral position of the track centre at distance z."""
        return self.offset + math.tan(self.yaw) * z + 0.5 * self.curvature * z * z

    def render(self, seed: int | None = None) -> np.ndarray:
        H, W = self.height, self.width
        img = np.empty((H, W, 3), dtype=np.float32)
        img[:] = self.asphalt
        horizon_row = int(math.ceil(self.v_horizon))
        img[:horizon_row] = self.background
        cols = np.arange(W, dtype=np.float32)
        line = np.array(self.line_color, dtype=np.float32)
        for r in range(horizon_row, H):
            v = r + 0.5
            if v - self.v_horizon < 0.5:
                continue
            z = self.distance(v)
            xc = self.centre_x(z)
            for draw, x in ((self.draw_left, xc - self.half_width), (self.draw_right, xc + self.half_width)):
                if not draw:
                    continue
                u0 = W / 2 + self.focal * (x - self.line_width / 2) / z
                u1 = W / 2 + self.focal * (x + self.line_width / 2) / z
                cover = np.clip(np.minimum(cols + 1, u1) - np.maximum(cols, u0), 0, 1)[:, None]
                img[r] = img[r] * (1 - cover) + line * cover
        img *= self.brightness
        if self.noise > 0:
            img += np.random.default_rng(seed).normal(0, self.noise, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)


def synthetic_drive(n_frames: int = 150, **track_kwargs) -> Iterator[tuple[np.ndarray, SyntheticTrack]]:
    """Yields (frame, track): the car weaves while the track bends left and right."""
    base = SyntheticTrack(**track_kwargs)
    for i in range(n_frames):
        t = 2 * math.pi * i / n_frames
        track = replace(
            base,
            offset=0.10 * math.sin(t),
            yaw=0.12 * math.sin(t + 1.2),
            curvature=0.7 * math.sin(2 * t),
        )
        yield track.render(seed=i), track


# ----------------------------------------------------------------------------- frame sources


def _resize(frame: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    if size is None or frame.shape[1::-1] == tuple(size):
        return frame
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def load_image(path: str | Path, size: tuple[int, int] | None = (128, 128)) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return _resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size)


def iter_frames(
    source: int | str | Path = "synthetic",
    max_frames: int | None = None,
    size: tuple[int, int] | None = (128, 128),
) -> Iterator[np.ndarray]:
    """RGB uint8 frames from "synthetic", a webcam index, a video file, an image folder or one image.

    Relative paths resolve against labs/.
    """
    if source == "synthetic":
        for frame, _ in synthetic_drive(max_frames or 150):
            yield _resize(frame, size)
        return
    if isinstance(source, (str, Path)):
        path = Path(source)
        path = path if path.is_absolute() else LAB_DIR / path
        if path.is_dir():
            files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            for f in files[:max_frames]:
                yield load_image(f, size)
            return
        if path.suffix.lower() in IMAGE_EXTS:
            yield load_image(path, size)
            return
        if not path.exists():
            raise FileNotFoundError(path)
        # video files go through imageio-ffmpeg: the OpenCV build in this env has no FFMPEG
        import imageio.v2 as iio

        reader = iio.get_reader(str(path))
        try:
            for n, frame in enumerate(reader):
                if max_frames is not None and n >= max_frames:
                    break
                yield _resize(np.asarray(frame)[..., :3], size)
        finally:
            reader.close()
        return
    cap = cv2.VideoCapture(source)  # webcam index (V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video source {source!r}")
    try:
        n = 0
        while max_frames is None or n < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            yield _resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size)
            n += 1
    finally:
        cap.release()


# ----------------------------------------------------------------------------- display


def use_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "image.interpolation": "nearest",
        }
    )


def show(images, titles=None, cols: int | None = None, size: float = 3.0, cmap: str = "magma", axes_off=True):
    """Grid of images: (H, W, 3) uint8 shown as colour, (H, W) arrays with `cmap` on a 0..1 scale."""
    images = list(images) if isinstance(images, (list, tuple)) else [images]
    titles = list(titles) if titles is not None else [None] * len(images)
    cols = cols or len(images)
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)
    for ax in axes.flat[len(images):]:
        ax.axis("off")
    for ax, img, title in zip(axes.flat, images, titles):
        img = np.asarray(img)
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(img)
        if title:
            ax.set_title(title)
        if axes_off:
            ax.axis("off")
        else:
            ax.grid(False)
    fig.tight_layout()
    plt.show()


def play(
    frames: Iterable[np.ndarray],
    render: Callable[[np.ndarray], np.ndarray],
    fps: float = 15.0,
    record_to: str | Path | None = None,
) -> int:
    """Live preview in the notebook: render(frame) -> RGB uint8, shown in place. Interrupt the kernel to stop."""
    from IPython.display import Image, display

    handle, writer, n = None, None, 0
    try:
        for frame in frames:
            t0 = time.perf_counter()
            out = render(frame)
            if record_to is not None:
                if writer is None:
                    import imageio.v2 as iio

                    path = Path(record_to)
                    path = path if path.is_absolute() else LAB_DIR / path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    writer = iio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=2)
                writer.append_data(out)
            png = Image(data=cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))[1].tobytes())
            if handle is None:
                handle = display(png, display_id=True)
            else:
                handle.update(png)
            n += 1
            time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.close()
    return n


def table(rows: list[dict], digits: int = 3) -> None:
    """Render a list of dicts as a compact HTML table in the notebook."""
    from IPython.display import HTML, display

    if not rows:
        return
    cols = list(rows[0])

    def fmt(v):
        return f"{v:.{digits}f}" if isinstance(v, float) else str(v)

    head = "".join(f"<th style='text-align:right;padding:2px 10px'>{c}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td style='text-align:right;padding:2px 10px'>{fmt(r[c])}</td>" for c in cols) + "</tr>"
        for r in rows
    )
    display(HTML(f"<table style='border-collapse:collapse;font-size:12px'><tr>{head}</tr>{body}</table>"))
