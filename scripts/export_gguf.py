import argparse
import os
import subprocess
import time
from pathlib import Path

from neurograph.config import git_hash, load_config

LLAMA_CPP = "https://github.com/ggml-org/llama.cpp"


def sh(cmd: str, cwd: str | None = None) -> str:
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def merge(base: str, adapter: Path, out: Path) -> None:
    # LoRA weights folded into the fp16 base, so llama.cpp sees one ordinary model
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16)
    PeftModel.from_pretrained(model, str(adapter)).merge_and_unload().save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="default: <derivatives>/qlora")
    args = ap.parse_args()
    t0 = time.perf_counter()
    cfg = load_config("finetune")
    g = cfg["gguf"]
    out = Path(args.dir) if args.dir else Path(load_config("paths")["paths"]["derivatives_dir"]) / cfg["out"]
    merged, lcpp = Path("/tmp/merged"), Path("/tmp/llama.cpp")
    merge(cfg["base_model"], out / "adapter", merged)
    if not lcpp.exists():
        sh(f"git clone -q --depth 1 {LLAMA_CPP} {lcpp}")
    sh(f"pip install -q {lcpp / 'gguf-py'}")  # only the gguf writer; llama.cpp's own requirements would pin torch
    f16 = Path("/tmp/model-f16.gguf")
    sh(f"python {lcpp / 'convert_hf_to_gguf.py'} {merged} --outtype f16 --outfile {f16}")
    sh("cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > /dev/null && "
       "cmake --build build --target llama-quantize llama-cli -j 4 > /dev/null", cwd=str(lcpp))
    gguf = out / g["file"]
    sh(f"{lcpp / 'build/bin/llama-quantize'} {f16} {gguf} {g['quant']}")
    # smoke test on CPU, like the demo Space: one short greedy answer
    t = time.perf_counter()
    reply = sh(f"{lcpp / 'build/bin/llama-cli'} -m {gguf} -p 'What is the lowest passing grade at the university?' -n 40 "
               f"--temp 0 -no-cnv -t 4 2>/dev/null")
    secs = time.perf_counter() - t

    print(f"== step 7.4 merge + GGUF export | commit {git_hash()} ==")
    print(f"merged {cfg['base_model']} + adapter -> {g['quant']} GGUF {gguf.stat().st_size / 1e9:.2f} GB "
          f"(f16 {f16.stat().st_size / 1e9:.2f} GB)")
    print(f"llama.cpp CPU smoke test (4 threads, 40 tokens, {secs:.0f} s incl. load): {reply.strip()[-200:]!r}")
    repo = os.environ.get("HF_REPO")
    if os.environ.get("HF_TOKEN") and repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=str(gguf), path_in_repo=g["file"], repo_id=repo)
        api.upload_file(path_or_fileobj=str(out / "model_card.md"), path_in_repo="README.md", repo_id=repo)
        print(f"uploaded to https://huggingface.co/{repo}")
    else:
        print("not uploaded (HF_TOKEN / HF_REPO not set)")
    f16.unlink()
    print(f"time {time.perf_counter() - t0:.0f} s | saved: {gguf}")


if __name__ == "__main__":
    main()
