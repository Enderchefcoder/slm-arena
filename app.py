"""SupraLabs Studio — four purpose-built SupraLabs model demos in one ZeroGPU Space."""
import gc
import json

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import spaces
    GPU = spaces.GPU
except ImportError:  # Allows the Space to be imported during local development/tests.
    GPU = lambda *args, **kwargs: (lambda fn: fn)


# Official SupraLabs checkpoints, selected for their distinct intended tasks.
INSTRUCT_MODEL = "SupraLabs/Supra-1.5-50M-Instruct-exp"
NTP_MODEL = "SupraLabs/Supra-1.5-50M-Base-exp"
TITLE_MODEL = "SupraLabs/supra-title-50m-pre"
THINKING_SUMMARIZER_MODEL = "SupraLabs/reasoning-summarizer-800m-pre"


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_and_generate(model_id, prompt, max_new_tokens, temperature=0.0,
                       top_p=0.95, repetition_penalty=1.0, trust_remote_code=False):
    """Load one checkpoint, run it, and immediately free memory for ZeroGPU users."""
    tokenizer = model = None
    try:
        device = _device()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        ).to(device).eval()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        sampling = float(temperature) > 0
        generation_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": sampling,
            "repetition_penalty": float(repetition_penalty),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if sampling:
            generation_kwargs.update({"temperature": max(float(temperature), 0.01), "top_p": float(top_p)})
        with torch.inference_mode():
            output = model.generate(**inputs, **generation_kwargs)
        generated = output[0][inputs.input_ids.shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip() or "(No text was generated.)"
    except Exception as exc:
        return f"**Generation unavailable**  \n`{type(exc).__name__}: {str(exc)[:240]}`"
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _chat_prompt(tokenizer, message):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True
        )
    return f"User: {message}\nAssistant:"


@GPU(duration=75)
def run_instruct(message, max_tokens, temperature):
    message = (message or "").strip()
    if not message:
        raise gr.Error("Write a message for Supra Instruct first.")
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_MODEL, trust_remote_code=False)
        prompt = _chat_prompt(tokenizer, message)
    finally:
        del tokenizer
        gc.collect()
    return _load_and_generate(INSTRUCT_MODEL, prompt, max_tokens, temperature)


@GPU(duration=75)
def run_ntp(seed_text, max_tokens, temperature):
    seed_text = (seed_text or "").strip()
    if not seed_text:
        raise gr.Error("Enter text for the model to continue.")
    return _load_and_generate(NTP_MODEL, seed_text, max_tokens, temperature)


@GPU(duration=75)
def run_title(message):
    message = (message or "").strip()
    if not message:
        raise gr.Error("Paste the first chat message to title.")
    # This is the exact raw format published by SupraLabs; no system prompt is used.
    title = _load_and_generate(
        TITLE_MODEL, f"User: {message}\nTitle: ", 16,
        temperature=0.4, top_p=0.85, repetition_penalty=1.2,
    )
    return title.replace("\n", " ").strip()


@GPU(duration=120)
def run_thinking_summarizer(reasoning, max_tokens):
    reasoning = (reasoning or "").strip()
    if not reasoning:
        raise gr.Error("Paste a reasoning chain or work log to summarize.")
    # The model card specifies trust_remote_code=True for its Qwen 3.5 base architecture.
    raw = _load_and_generate(
        THINKING_SUMMARIZER_MODEL, reasoning + "\n", max_tokens, trust_remote_code=True
    )
    try:
        parsed = json.loads(raw)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, TypeError):
        return raw


CSS = """
:root { --supra-purple: #8f5bff; --supra-blue: #4db8ff; --ink: #10182e; }
.gradio-container { max-width: 1180px !important; background: #f7f8ff;
  background-image: radial-gradient(circle at 5% 0%, #e5ddff 0, transparent 30%), radial-gradient(circle at 100% 4%, #d9f4ff 0, transparent 27%); }
#supra-hero { padding: 32px 18px 24px; text-align: center; border-radius: 22px; color: white;
  background: linear-gradient(120deg, #43208b, var(--supra-purple) 54%, #328dcc); box-shadow: 0 14px 35px #6f57b844; }
#supra-hero h1 { font-size: 2.55rem; margin: 0; letter-spacing: -.06rem; }
#supra-hero p { margin: 8px 0 0; font-size: 1.08rem; opacity: .94; }
#supra-hero .eyebrow { color: #d9f6ff; font-weight: 700; font-size: .78rem; letter-spacing: .16em; text-transform: uppercase; }
.tab-nav button { border-radius: 12px 12px 0 0 !important; font-weight: 700; }
.primary-btn { background: linear-gradient(105deg, var(--supra-purple), var(--supra-blue)) !important; border: 0 !important; color: white !important; font-weight: 700 !important; }
.model-card { border-left: 4px solid var(--supra-purple); padding: 4px 0 4px 16px; margin-bottom: 12px; }
.model-card h2 { margin: 0 0 5px; color: var(--ink); } .model-card p { margin: 0; color: #526077; }
footer { text-align:center; color: #69758d; }
"""

with gr.Blocks(title="SupraLabs Studio", theme=gr.themes.Base(), css=CSS) as demo:
    gr.HTML("""
    <section id="supra-hero"><div class="eyebrow">SupraLabs • ZeroGPU Studio</div>
    <h1>Small models. Focused tools.</h1><p>Try four specialized SupraLabs checkpoints in one shared, on-demand GPU Space.</p></section>
    """)
    with gr.Tabs():
        with gr.Tab("✦ Instruct", id="instruct"):
            gr.HTML("<div class='model-card'><h2>Supra-1.5 50M Instruct</h2><p>Experimental instruction-tuned chat model. Ask a direct question or give it a task.</p></div>")
            instruct_input = gr.Textbox(label="Message", lines=5, placeholder="Explain why leaves change color in autumn.")
            with gr.Row():
                instruct_tokens = gr.Slider(16, 256, value=128, step=8, label="Maximum new tokens")
                instruct_temp = gr.Slider(0, 1.2, value=0.7, step=0.1, label="Temperature")
            instruct_button = gr.Button("Ask Supra Instruct", variant="primary", elem_classes="primary-btn")
            instruct_output = gr.Markdown(label="Response")
            instruct_button.click(run_instruct, [instruct_input, instruct_tokens, instruct_temp], instruct_output)

        with gr.Tab("→ Next token", id="ntp"):
            gr.HTML("<div class='model-card'><h2>Supra-1.5 50M Base</h2><p>Raw next-token prediction (NTP). Provide a seed and inspect how the base model continues it.</p></div>")
            ntp_input = gr.Textbox(label="Seed text", lines=5, placeholder="The observatory opened its dome just as the sky turned violet,")
            with gr.Row():
                ntp_tokens = gr.Slider(8, 256, value=96, step=8, label="Tokens to continue")
                ntp_temp = gr.Slider(0, 1.5, value=0.8, step=0.1, label="Temperature")
            ntp_button = gr.Button("Continue text", variant="primary", elem_classes="primary-btn")
            ntp_output = gr.Markdown(label="Continuation")
            ntp_button.click(run_ntp, [ntp_input, ntp_tokens, ntp_temp], ntp_output)

        with gr.Tab("⌁ Title generator", id="title"):
            gr.HTML("<div class='model-card'><h2>Supra Title 50M</h2><p>Purpose-built for concise conversation titles. Enter the first user message—no system prompt is added.</p></div>")
            title_input = gr.Textbox(label="First chat message", lines=5, placeholder="My WiFi disconnects every ten minutes. What should I check?")
            title_button = gr.Button("Generate title", variant="primary", elem_classes="primary-btn")
            title_output = gr.Textbox(label="Suggested title", interactive=False)
            title_button.click(run_title, title_input, title_output)

        with gr.Tab("{} Thinking summarizer", id="summarizer"):
            gr.HTML("<div class='model-card'><h2>Reasoning Summarizer 0.8B</h2><p>Turns a plain-text reasoning chain into structured JSON: <code>title</code>, <code>sub_title</code>, <code>summary</code>, and <code>cur_task</code>.</p></div>")
            thinking_input = gr.Textbox(label="Reasoning chain / work log", lines=14, placeholder="I need to diagnose why the API returns 401...\nFirst I should inspect the authorization header.")
            thinking_tokens = gr.Slider(64, 320, value=192, step=16, label="Maximum JSON tokens")
            thinking_button = gr.Button("Create structured summary", variant="primary", elem_classes="primary-btn")
            thinking_output = gr.Code(label="Structured metadata", language="json", interactive=False)
            thinking_button.click(run_thinking_summarizer, [thinking_input, thinking_tokens], thinking_output)
    gr.HTML("<footer>Models are loaded only for your request and released afterwards to share ZeroGPU capacity. Experimental models may produce imperfect output—do not submit sensitive information.</footer>")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
