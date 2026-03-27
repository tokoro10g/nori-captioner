# nori-captioner

Local Vision Caption Studio for Images and Video.

A web UI for captioning image/video datasets in-place using local VLMs.
Captions are saved as sidecar `.txt` files next to each media file.

## Quick start

```bash
uvx nori-captioner
```

Scans the current directory (or a given path) recursively for images and videos and opens a local web UI.

```bash
uvx nori-captioner /path/to/dataset
```

## Features

- Recursive directory scan — hidden directories are excluded
- Per-file metadata display: resolution, duration, frame count, fps
- Manual caption editing with autosave
- Upload images/videos via file picker or drag-and-drop
- Delete files (removes media and sidecar caption together)
- Auto-captioning queue with single-file and batch modes
- Configurable user prompt — editable in the UI and persisted to disk
- Pagination and filter by caption state (all / captioned / uncaptioned / queued)

## Auto-captioning with local VLMs

Run with VLM extras:

```bash
uvx --from 'nori-captioner[vlm]' nori-captioner
```

> **Note:** Qwen3-VL requires `torchvision`, which is included in the `vlm` extra.
> On Linux x86\_64, CUDA 12.8 wheels for `torch` and `torchvision` are used automatically.

Optional 4-bit / 8-bit quantization:

```bash
uvx --from 'nori-captioner[vlm,quantize]' nori-captioner
```

Run with a built-in model alias:

```bash
uvx --from 'nori-captioner[vlm]' nori-captioner --model qwen3-vl:8b
```

Or pass any Hugging Face model ID directly:

```bash
uvx --from 'nori-captioner[vlm]' nori-captioner --model your-org/your-vlm
```

### Model aliases

| Alias | Model |
|---|---|
| `qwen3-vl:2b` | Qwen/Qwen3-VL-2B-Instruct |
| `qwen3-vl:4b` | Qwen/Qwen3-VL-4B-Instruct |
| `qwen3-vl:8b` | Qwen/Qwen3-VL-8B-Instruct |
| `qwen3-vl:32b` | Qwen/Qwen3-VL-32B-Instruct |
| `qwen3-vl:30b` | Qwen/Qwen3-VL-30B-A3B-Instruct |
| `qwen2.5-vl:3b` | Qwen/Qwen2.5-VL-3B-Instruct |
| `qwen2.5-vl:7b` | Qwen/Qwen2.5-VL-7B-Instruct |
| `qwen2.5-vl:72b` | Qwen/Qwen2.5-VL-72B-Instruct |
| `qwen2-vl:2b` | Qwen/Qwen2-VL-2B-Instruct |
| `qwen2-vl:7b` | Qwen/Qwen2-VL-7B-Instruct |
| `qwen2-vl:72b` | Qwen/Qwen2-VL-72B-Instruct |
| `gemma3:4b` | google/gemma-3-4b-it |
| `gemma3:12b` | google/gemma-3-12b-it |
| `gemma3:27b` | google/gemma-3-27b-it |

## CLI options

| Option | Default | Description |
|---|---|---|
| `directory` | `.` | Directory to scan |
| `--model` | none | Model alias or HF model ID |
| `--quantize` | none | `4` or `8` bit quantization |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--frames` | `8` | Video frames sampled per auto-caption |
| `--system-prompt` | built-in | System prompt for model behavior |
| `--prompt` | built-in | Captioning prompt (also editable in UI) |
| `--host` | `127.0.0.1` | Server bind address |
| `--port` | `8765` | Server port |
| `--no-browser` | false | Suppress automatic browser open |

## Prompt persistence

The user prompt edited in the web UI is saved to `.nori-captioner.settings.json` in the scanned
directory and automatically restored on next launch.
