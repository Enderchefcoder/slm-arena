---
title: SLM Arena
emoji: ⚔️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.22.0
python_version: '3.12'
app_file: app.py
pinned: false
license: gpl-3.0
short_description: Blindly compare tiny language model outputs and vote.
---

# SLM Arena

A ZeroGPU Hugging Face Space for anonymous head-to-head comparisons of the top 20
Efficiency entries in [Glint Research's Tiny-ML Leaderboard](https://huggingface.co/spaces/Glint-Research/Tiny-ML-Leaderboard).
The model names are concealed until a visitor votes.

## Modes

- **Completion** samples two models from the full top-20 list and continues the supplied text.
- **Reply** samples only the instruction-tuned entries (`Supra-50M-Instruct`,
  `KeyLM-75M-Instruct`, and `Supra-1.5-50M-Instruct-exp`). It uses a model's
  published tokenizer chat template when one exists.

Models are loaded one at a time with `AutoModelForCausalLM`, generated, and
released. This keeps the Space within a shared ZeroGPU allocation rather than
keeping twenty checkpoints in VRAM. Unsupported or gated checkpoints return a
clear per-round error; they are never silently replaced with another model.

## Deploying

1. Create the dataset repository **`Enderchef/slm-arena-votes`** (private is
   recommended if prompts/outputs should not be public).
2. In the Space **Settings → Variables and secrets**, add a secret named
   `HF_TOKEN`. It needs write access to that dataset repo.
3. Enable ZeroGPU hardware for the Space. The generation handler is decorated
   with `@spaces.GPU(duration=120)` and the Gradio queue serializes requests to
   avoid VRAM contention.

Each vote is an immutable `votes/<uuid>.json` commit in the dataset repository.
Using one file per vote avoids lost votes from concurrent Space replicas. Stored
records include the prompt, outputs, anonymous-side choice, models, mode, and
UTC timestamp—do not submit secrets or personal data.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

Run the lightweight checks with `python -m unittest discover -s tests`.
