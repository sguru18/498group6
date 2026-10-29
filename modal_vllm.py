"""Modal app: serves Qwen2.5-Coder-32B-Instruct-AWQ via vLLM OpenAI-compatible API."""

import modal

app = modal.App("cs498-vllm")

# Separate volume for 32B weights (~16GB AWQ)
model_volume = modal.Volume.from_name("qwen-weights-32b", create_if_missing=True)

MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"  # 4-bit quantized, A100 80GB
MODEL_DIR = "/models"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.6.0", "huggingface_hub[hf_transfer]", "autoawq")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=3600,
)
@modal.concurrent(max_inputs=4)
@modal.web_server(8000, startup_timeout=900)  # 15 min: first run downloads ~16GB
def serve():
    """Download model to volume if needed, then start vLLM server."""
    import os
    import subprocess

    # Download only on first run — volume persists between sessions
    if not os.path.exists(f"{MODEL_DIR}/config.json"):
        print("Downloading model weights to volume (one-time, ~16GB AWQ)...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR)
        model_volume.commit()
        print("Download complete, weights cached in volume.")

    env = {**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_DIR,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", "16384",
        "--enforce-eager",            # skip FlashInfer JIT — avoids nvcc
        "--enable-auto-tool-choice",  # allow tool_choice="required"
        "--tool-call-parser", "hermes",  # Qwen2.5-Coder uses Hermes tool format
        "--quantization", "awq",      # AWQ 4-bit quantization
    ], env=env)
