"""SLM Arena — blind side-by-side comparison for the Tiny-ML Leaderboard."""
import gc
import json
import os
import random
import tempfile
import time
import uuid
from datetime import datetime, timezone

import gradio as gr
import torch
from huggingface_hub import CommitOperationAdd, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import spaces
    GPU = spaces.GPU
except ImportError:  # Makes local development possible.
    GPU = lambda *args, **kwargs: (lambda fn: fn)

VOTE_REPO = "Enderchef/slm-arena-votes"
# Ordered by Efficiency on Glint-Research/Tiny-ML-Leaderboard (31 July 2026).
# `instruct` is deliberately conservative: Reply only offers released instruction models.
MODELS = [
    ("Glint-Research/Glint-2", False), ("AxiomicLabs/GPT-X2-125M", False),
    ("Glint-Research/Glint-1.3", False), ("SupraLabs/Supra-50M-Instruct", True),
    ("AxiomicLabs/GPT-S-5M", False), ("SupraLabs/Supra-50M-Base", False),
    ("SupraLabs/MicroSupra-1k", False), ("OpenGCM/Hydrion-v1-Base", False),
    ("IvmeLabs/Ivme-Conversate-v2-Base", False), ("finnianx/michel-nano-v2", False),
    ("TobiasLogic/TextModel-v1", False), ("finnianx/Gros-Michel-90m-Base-v2", False),
    ("finnianx/Gros-Michel-90m-Base", False), ("SupraLabs/Supra-Mini-v6-1M", False),
    ("wonderfulmonkey/CreekwardGoat-500K", False), ("finnianx/michel-micro", False),
    ("MinimaLabs/KeyLM-75M-Instruct", True), ("GODELEV/Archaea-74M", False),
    ("finnianx/michel-tiny", False), ("SupraLabs/Supra-1.5-50M-Instruct-exp", True),
]
INSTRUCT = {model for model, is_instruct in MODELS if is_instruct}
ALL_MODELS = [model for model, _ in MODELS]


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _format_prompt(tokenizer, text, mode):
    """Use the model's own chat template where provided; never invent one."""
    if mode == "Reply" and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
    # Instruction checkpoints without a template still get a minimal, explicit prompt.
    if mode == "Reply":
        return f"User: {text}\nAssistant:"
    return text


def generate_one(model_id, prompt, mode, max_tokens, temperature):
    """Load, generate, and release one tiny model to fit ZeroGPU's shared VRAM."""
    tokenizer = model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.float16 if _device() == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=False, low_cpu_mem_usage=True
        ).to(_device()).eval()
        formatted = _format_prompt(tokenizer, prompt, mode)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(_device())
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=int(max_tokens), do_sample=temperature > 0,
                temperature=max(float(temperature), 0.01), top_p=0.95,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return text or "(This model returned an empty completion.)"
    except Exception as exc:
        # Do not silently substitute a different model: a valid comparison must be auditable.
        return f"Generation unavailable for this round: {type(exc).__name__}: {str(exc)[:180]}"
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@GPU(duration=120)
def run_arena(prompt, mode, max_tokens, temperature):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Enter a message first.")
    candidates = list(INSTRUCT) if mode == "Reply" else ALL_MODELS
    left_model, right_model = random.sample(candidates, 2)
    # Randomize presentation independently of sampling, so position cannot signal identity.
    if random.choice([True, False]):
        left_model, right_model = right_model, left_model
    left = generate_one(left_model, prompt, mode, max_tokens, temperature)
    right = generate_one(right_model, prompt, mode, max_tokens, temperature)
    round_data = {"id": str(uuid.uuid4()), "prompt": prompt, "mode": mode,
                  "left_model": left_model, "right_model": right_model,
                  "left_output": left, "right_output": right, "created_at": time.time(), "voted": False}
    return left, right, round_data, gr.update(visible=True), gr.update(visible=False), "**Compare anonymously, then vote.**"


def save_vote(round_data, choice):
    if not round_data or round_data.get("voted"):
        raise gr.Error("Generate a new comparison before voting again.")
    record = {key: round_data[key] for key in ("id", "prompt", "mode", "left_model", "right_model", "left_output", "right_output", "created_at")}
    record.update({"choice": choice, "timestamp": datetime.now(timezone.utc).isoformat()})
    warning = ""
    token = os.getenv("HF_TOKEN")
    if token:
        try:
            # One immutable JSON per vote eliminates read/append/write races between Space replicas.
            with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
                path = f.name
            try:
                HfApi(token=token).create_commit(
                    repo_id=VOTE_REPO, repo_type="dataset",
                    operations=[CommitOperationAdd(path_in_repo=f"votes/{record['id']}.json", path_or_fileobj=path)],
                    commit_message=f"Add arena vote {record['id']}",
                )
            finally:
                os.unlink(path)
        except Exception as exc:
            warning = f"\n\n*Vote could not be persisted: {type(exc).__name__}.*"
    else:
        warning = "\n\n*Voting storage is not configured (set the Space secret `HF_TOKEN`).*"
    round_data["voted"] = True
    reveal = f"### Revealed\n**A — {round_data['left_model']}**  \n**B — {round_data['right_model']}**"
    return round_data, reveal, gr.update(visible=False), "Vote recorded. Identities are now revealed." + warning


CSS = """
.gradio-container {max-width: 1180px !important; background: #0b1020;} 
#hero {text-align:center; padding: 20px 0 6px;} #hero h1 {font-size: 2.4rem; margin-bottom: .2rem;}
.output textarea {font-size: 15px !important; line-height: 1.55;} .vote button {min-height: 48px; font-weight: 700;}
"""
with gr.Blocks(title="SLM Arena", theme=gr.themes.Base(), css=CSS) as demo:
    gr.HTML("<div id='hero'><h1>⚔️ SLM Arena</h1><p>Blind-test tiny language models from the Tiny-ML Leaderboard.</p></div>")
    state = gr.State({})
    with gr.Row():
        mode = gr.Radio(["Completion", "Reply"], value="Completion", label="Mode", info="Reply compares instruction-tuned SLMs; Completion uses all top-20 models.")
        max_tokens = gr.Slider(8, 256, value=96, step=8, label="Maximum new tokens")
        temperature = gr.Slider(0, 1.5, value=0.7, step=0.1, label="Temperature")
    prompt = gr.Textbox(label="Your message", placeholder="Write a short story about a robot gardener…", lines=3)
    send = gr.Button("Generate blind comparison", variant="primary")
    with gr.Row():
        left = gr.Textbox(label="Model A", lines=15, interactive=False, elem_classes="output")
        right = gr.Textbox(label="Model B", lines=15, interactive=False, elem_classes="output")
    vote_row = gr.Row(visible=False)
    with vote_row:
        a = gr.Button("A is better", elem_classes="vote")
        tie = gr.Button("Tie", elem_classes="vote")
        b = gr.Button("B is better", elem_classes="vote")
        bad = gr.Button("Both are bad", elem_classes="vote")
    reveal = gr.Markdown(visible=False)
    status = gr.Markdown("Choose a mode, write a prompt, and compare the two anonymous outputs.")
    send.click(run_arena, [prompt, mode, max_tokens, temperature], [left, right, state, vote_row, reveal, status])
    for button, choice in [(a, "left"), (b, "right"), (tie, "tie"), (bad, "both_bad")]:
        button.click(lambda s, c=choice: save_vote(s, c), state, [state, reveal, vote_row, status])
    gr.Markdown("---\nModels are sampled from the **top 20 Efficiency** entries on [Glint Research’s Tiny-ML Leaderboard](https://huggingface.co/spaces/Glint-Research/Tiny-ML-Leaderboard). Votes are saved as immutable records to `Enderchef/slm-arena-votes`. Do not enter sensitive information.")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
