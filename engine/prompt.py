"""Prompt handling: inline flags, Persian detection + translation, styles, safety."""
import random
import re

PERSIAN_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
FLAG_RE = re.compile(r"(?:^|\s)--(style|ratio|seed|steps|hd)\b(?:[=\s]+([^\s-]\S*))?", re.I)

RATIOS = {
    "1:1": (512, 512), "3:2": (576, 384), "2:3": (384, 576),
    "4:3": (576, 448), "3:4": (448, 576), "16:9": (640, 360), "9:16": (360, 640),
}

BASE_NEG = (
    "lowres, blurry, bad anatomy, bad hands, extra fingers, missing fingers, deformed, "
    "disfigured, watermark, text, signature, jpeg artifacts, worst quality, low quality, "
    "cropped, nsfw, nude"
)

STYLES = {
    "auto": ("masterpiece, best quality, highly detailed, sharp focus, professional lighting", ""),
    "realistic": ("photorealistic, ultra detailed, sharp focus, natural lighting, 85mm lens, "
                  "high dynamic range, professional photography", "cartoon, painting, illustration, 3d render"),
    "cinematic": ("cinematic still, dramatic lighting, anamorphic lens, film grain, color graded, "
                  "volumetric light, highly detailed", ""),
    "anime": ("anime style, vibrant colors, clean lineart, detailed, studio quality, key visual",
              "photo, realistic, 3d"),
    "art": ("digital painting, concept art, intricate details, artstation, masterpiece", ""),
    "3d": ("3d render, octane render, unreal engine 5, soft lighting, highly detailed, pixar style", ""),
    "pixel": ("pixel art, voxel art, blocky, minecraft style, isometric, vibrant colors",
              "smooth, realistic, blurry"),
    "fantasy": ("epic fantasy art, magical atmosphere, intricate details, dramatic lighting, masterpiece", ""),
}

BLOCK_EN = re.compile(
    r"\b(nude|naked|nsfw|porn\w*|sex\w*|erotic\w*|topless|genital\w*|nipple\w*|hentai|lolicon|gore)\b", re.I)
BLOCK_FA = re.compile(r"(برهنه|لخت|سکس|پورن|شهوانی|عریان)")


class UserError(Exception):
    """Error whose message is safe to show to the end user (Persian)."""


def parse_flags(text):
    opts = {}

    def repl(m):
        key, val = m.group(1).lower(), m.group(2)
        if key == "hd":
            opts["hd"] = True
            return " " + (val or "")
        if val:
            opts[key] = val.translate(DIGITS)
        return " "

    clean = FLAG_RE.sub(repl, text)
    return re.sub(r"\s+", " ", clean).strip(), opts


def is_blocked(text):
    return bool(BLOCK_EN.search(text) or BLOCK_FA.search(text))


_tr = None


def translate_fa_en(text, model_name):
    """Translate Persian to English, phrase by phrase (better for prompts)."""
    global _tr
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    if _tr is None:
        tok = MarianTokenizer.from_pretrained(model_name)
        mdl = MarianMTModel.from_pretrained(model_name).eval()
        _tr = (tok, mdl)
    tok, mdl = _tr
    parts = [p.strip() for p in re.split(r"[\n.!؟?،,;؛]+", text) if p.strip()][:12]
    if not parts:
        return text
    batch = tok(parts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.inference_mode():
        out = mdl.generate(**batch, num_beams=4, max_new_tokens=96)
    return ", ".join(tok.decode(o, skip_special_tokens=True).strip(" .") for o in out)


def build(raw, cfg):
    text, opts = parse_flags(raw)
    if not text:
        raise UserError("لطفاً توضیح تصویر را بنویسید.")
    if len(text) > cfg.get("max_prompt_chars", 500):
        raise UserError("متن خیلی طولانی است؛ کوتاه‌ترش کنید.")
    if is_blocked(text):
        raise UserError("این درخواست پشتیبانی نمی‌شود.")

    en = translate_fa_en(text, cfg["translator"]) if PERSIAN_RE.search(text) else text
    if is_blocked(en):
        raise UserError("این درخواست پشتیبانی نمی‌شود.")

    style = opts.get("style", "auto").lower()
    if style not in STYLES:
        style = "auto"
    pos_add, neg_add = STYLES[style]
    width, height = RATIOS.get(opts.get("ratio", "1:1"), RATIOS["1:1"])

    try:
        seed = int(opts["seed"]) if "seed" in opts else random.randint(0, 2**31 - 1)
    except ValueError:
        seed = random.randint(0, 2**31 - 1)
    try:
        steps = max(3, min(8, int(opts.get("steps", cfg.get("steps", 5)))))
    except ValueError:
        steps = cfg.get("steps", 5)

    return {
        "prompt_en": en,
        "prompt": f"{en}, {pos_add}",
        "negative": BASE_NEG + (", " + neg_add if neg_add else ""),
        "style": style, "width": width, "height": height,
        "seed": seed, "steps": steps, "hd": bool(opts.get("hd")),
    }
