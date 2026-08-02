---
title: SupraLabs Studio
emoji: ✦
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.22.0
python_version: '3.12'
app_file: app.py
pinned: false
license: gpl-3.0
The complete SupraLabs demo site, powered by ZeroGPU.
---

# SupraLabs Studio

A single Hugging Face **ZeroGPU** Space for the complete public [SupraLabs](https://huggingface.co/SupraLabs) catalog. The interface uses SupraLabs’ purple (`#8F5BFF`) and blue (`#4DB8FF`) visual palette.

## Demos

| Demo | Checkpoint | Input / output contract |
| --- | --- | --- |
| **Instruct** | [`SupraLabs/Supra-1.5-50M-Instruct-exp`](https://huggingface.co/SupraLabs/Supra-1.5-50M-Instruct-exp) | Chat-style instruction following. Uses the checkpoint’s published chat template when it has one. |
| **Next token** | [`SupraLabs/Supra-1.5-50M-Base-exp`](https://huggingface.co/SupraLabs/Supra-1.5-50M-Base-exp) | Raw next-token continuation from a supplied seed; no chat wrapper is added. |
| **Title generator** | [`SupraLabs/supra-title-50m-pre`](https://huggingface.co/SupraLabs/supra-title-50m-pre) | First chat message in; a concise title out. It uses the model card’s training format: `User: <message>\nTitle:`. |
| **Thinking summarizer** | [`SupraLabs/reasoning-summarizer-800m-pre`](https://huggingface.co/SupraLabs/reasoning-summarizer-800m-pre) | Plain-text reasoning chain in; JSON metadata (`title`, `sub_title`, `summary`, `cur_task`) out. |

## Full catalog and version comparison

The **All models** tab makes every public release discoverable, including the current and predecessor releases of Instruct, Base, Title, Router, Mini, Reasoning, weather, and A2A families. The predecessor toggle exposes v1–v5 entries and adds them to both sides of the built-in comparison panel, so a current checkpoint and an older version can be run side by side on exactly the same raw prompt.

Standard Transformers causal-language-model releases are runnable in the playground. GGUF releases are documented in the catalog but require a llama.cpp runtime; A2A is image-capable; and SupraWeather models are tabular classifiers, so their input contracts need their dedicated modality demos rather than the raw text runner. `Supra2-100M` is visible for completeness but its publisher has gated it as an internal artifact, so it cannot be loaded by the Space.

## ZeroGPU design

Each action is independently decorated with `@spaces.GPU`. A checkpoint is loaded for that request only, generated from, then released with CUDA cache cleanup. This keeps the large summarizer and the smaller 50M models from permanently occupying shared VRAM. The Gradio queue is serialized to avoid competing model loads.

The Space intentionally does not retain prompts or outputs. Please do not enter private or sensitive text. These are research/experimental checkpoints, so outputs can be incomplete or malformed—particularly the summarizer’s JSON.

## Local development

```bash
pip install -r requirements.txt
python app.py
python -m unittest discover -s tests
```

For hosted inference, enable **ZeroGPU** on the Hugging Face Space hardware settings.
