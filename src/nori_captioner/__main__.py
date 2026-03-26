from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

import uvicorn
from rich.console import Console

from .captioner import DEFAULT_PROMPT, DEFAULT_SYSTEM_PROMPT
from .server import AppConfig, create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web UI for image/video dataset captioning")
    parser.add_argument("directory", nargs="?", default=".", help="Directory to scan recursively")
    parser.add_argument(
        "--model",
        default=None,
        help="Model alias (qwen3-vl:8b) or HuggingFace model id",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        choices=[4, 8],
        default=None,
        help="Optional quantization",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--frames",
        type=int,
        default=8,
        help="Frames per video for auto-captioning",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for model behavior",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Captioning prompt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.directory).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Invalid directory: {root}")

    app = create_app(
        AppConfig(
            root=root,
            model=args.model,
            quantize=args.quantize,
            device=args.device,
            frames=args.frames,
            system_prompt=args.system_prompt,
            prompt=args.prompt,
        )
    )

    url = f"http://{args.host}:{args.port}"
    Console().print(f"[bold green]nori-captioner[/bold green] scanning: [cyan]{root}[/cyan]")
    Console().print(f"Open: [underline]{url}[/underline]")

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
