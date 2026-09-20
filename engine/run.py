"""Entry point used by GitHub Actions: python -m engine.run"""
import json
import os
import sys
import time
import traceback

from engine import deliver
from engine.prompt import UserError, build


def write_meta(job_id, meta):
    os.makedirs("out", exist_ok=True)
    with open(f"out/{job_id}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def main():
    cfg = json.load(open("config.json", encoding="utf-8"))
    raw = os.environ.get("PROMPT", "").strip()
    job_id = os.environ.get("JOB_ID") or time.strftime("%Y%m%d%H%M%S")
    platform = os.environ.get("PLATFORM", "web")
    chat_id = os.environ.get("CHAT_ID", "")
    t0 = time.time()
    meta = {"job_id": job_id, "status": "error", "message": "خطای داخلی"}
    os.makedirs("out", exist_ok=True)

    try:
        p = build(raw, cfg)
        from engine.generate import generate  # heavy import after cheap validation
        img = generate(cfg, p)
        path = f"out/{job_id}.jpg"
        img.convert("RGB").save(path, "JPEG", quality=95, optimize=True)
        meta = {
            "job_id": job_id, "status": "done", "prompt_en": p["prompt_en"], "style": p["style"],
            "seed": p["seed"], "width": img.width, "height": img.height,
            "seconds": round(time.time() - t0, 1),
        }
        caption = f"🎨 {p['style']} • seed {p['seed']}\n{p['prompt_en'][:300]}"
        try:
            deliver.send_image(platform, chat_id, path, caption, also_document=p["hd"])
        except Exception:
            traceback.print_exc()
    except UserError as e:
        meta = {"job_id": job_id, "status": "error", "message": str(e)}
        deliver.send_text(platform, chat_id, "⚠️ " + str(e))
    except Exception:
        traceback.print_exc()
        deliver.send_text(platform, chat_id, "⚠️ ساخت تصویر ناموفق بود. دوباره تلاش کنید.")
        write_meta(job_id, meta)
        sys.exit(1)
    write_meta(job_id, meta)


if __name__ == "__main__":
    main()
