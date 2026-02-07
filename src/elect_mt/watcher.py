from __future__ import annotations

import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .config import OUTPUT_VARIANTS, SUPPORTED_IMAGE_EXTENSIONS
from .processor import ProcessResult, process_file


def _wait_for_stable_file(path: Path, *, retries: int, sleep_s: float) -> bool:
    """Best-effort check that file writing has finished."""
    last_size = -1
    for _ in range(retries):
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            time.sleep(sleep_s)
            continue
        if size > 0 and size == last_size:
            return True
        last_size = size
        time.sleep(sleep_s)
    return False


class ImageEventHandler(FileSystemEventHandler):
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        *,
        overwrite: bool = False,
        stability_retries: int = 10,
        stability_sleep_s: float = 0.2,
        read_retries: int = 3,
        read_sleep_s: float = 0.2,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.stability_retries = stability_retries
        self.stability_sleep_s = stability_sleep_s
        self.read_retries = read_retries
        self.read_sleep_s = read_sleep_s
        self._seen: set[Path] = set()

    def _maybe_process(self, path: Path) -> ProcessResult | None:
        path = path.resolve()
        if path in self._seen:
            return None
        if not path.is_file():
            return None

        # Only process files inside the configured input directory.
        try:
            path.relative_to(self.input_dir.resolve())
        except ValueError:
            return None

        if not _wait_for_stable_file(path, retries=self.stability_retries, sleep_s=self.stability_sleep_s):
            return None

        last_exc: Exception | None = None
        for _ in range(self.read_retries):
            try:
                result = process_file(path, self.output_dir, overwrite=self.overwrite)
                if result is not None:
                    self._seen.add(path)
                return result
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(self.read_sleep_s)

        if last_exc is not None:
            raise last_exc
        return None

    def _maybe_delete_outputs(self, path: Path) -> None:
        path = path.resolve()
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            return

        try:
            path.relative_to(self.input_dir.resolve())
        except ValueError:
            return

        for variant_name in OUTPUT_VARIANTS:
            output_path = self.output_dir / f"{path.stem}_{variant_name}.png"
            try:
                output_path.unlink()
            except FileNotFoundError:
                continue

    def on_created(self, event):  # type: ignore[override]
        if event.is_directory:
            return
        self._maybe_process(Path(event.src_path))

    def on_deleted(self, event):  # type: ignore[override]
        if event.is_directory:
            return
        self._maybe_delete_outputs(Path(event.src_path))

    def on_moved(self, event):  # type: ignore[override]
        if event.is_directory:
            return
        src_path = Path(event.src_path)
        dest_path = Path(event.dest_path)
        self._maybe_delete_outputs(src_path)
        self._maybe_process(dest_path)


def watch(
    input_dir: Path,
    output_dir: Path,
    *,
    recursive: bool = False,
    overwrite: bool = False,
) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    observer = Observer()
    handler = ImageEventHandler(input_dir, output_dir, overwrite=overwrite)
    observer.schedule(handler, str(input_dir), recursive=recursive)
    observer.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
