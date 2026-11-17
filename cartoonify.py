"""Photo -> illustration. Standalone: needs only this folder.

    python cartoonify.py --input photo.jpg
    python cartoonify.py --input folder_of_photos --out results --device cpu

For every photo it writes <name>.jpg (the result) into the output folder.
With --compare it also writes <name>_compare.jpg (photo | result).

Steps: main model draws the whole picture, a face detector finds small faces,
the face refiner redraws each of them, and they are blended back in.
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from model import UNetGenerator

HERE = os.path.dirname(os.path.abspath(__file__))
W_MAIN = os.path.join(HERE, "weights", "main_v7.pth")
W_FACE = os.path.join(HERE, "weights", "face_refiner_v7.pth")
W_DET = os.path.join(HERE, "weights", "face_detection_yunet_2023mar.onnx")
EXTS = (".png", ".jpg", ".jpeg", ".webp")
_WEIGHTS_URL = "https://www.thearifoffice.com"


def to_tensor(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1)


def to_pil(t):
    a = ((t.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(a)


def load_model(path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Weight file not found: {path}\n"
            f"Download weights from {_WEIGHTS_URL} and place them in the weights/ folder."
        )
    sd = torch.load(path, map_location="cpu")
    ngf = next(v for v in sd.values() if v.ndim == 4).shape[0]  # network width, read from the file
    G = UNetGenerator(ngf, depth=4 if any(k.startswith("down4") for k in sd) else 3)
    G.load_state_dict(sd)
    return G.to(device).eval()


def feather_mask(side, inner=0.55, outer=0.92):
    """1 in the middle, fading to 0 before the crop edge (soft ellipse)."""
    y, x = np.mgrid[0:side, 0:side]
    r = np.hypot((x - side / 2) / (side / 2), (y - side / 2) / (side / 2))
    return np.clip((outer - r) / (outer - inner), 0, 1)[..., None].astype(np.float32)


def prepare(photo, long_side=1024):
    """Respect phone rotation, bring the longer side to `long_side`, make both sides a multiple of 16."""
    photo = ImageOps.exif_transpose(photo).convert("RGB")
    W, H = photo.size
    if long_side and max(W, H) != long_side:
        k = long_side / max(W, H)
        photo = photo.resize((max(16, round(W * k)), max(16, round(H * k))), Image.LANCZOS)
    W, H = photo.width - photo.width % 16, photo.height - photo.height % 16
    x0, y0 = (photo.width - W) // 2, (photo.height - H) // 2
    return photo.crop((x0, y0, x0 + W, y0 + H))


class Cartoonifier:
    """Load once, then call it on as many photos as you like.

        c = Cartoonifier(device="cpu")
        result = c(Image.open("photo.jpg"))      # a PIL image
        result.save("out.jpg")
    """

    def __init__(self, device=None, refine_faces=True, max_face=320, long_side=1024):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_face, self.long_side = max_face, long_side
        self.G = load_model(W_MAIN, self.device)
        self.R = load_model(W_FACE, self.device) if refine_faces else None
        self.det = cv2.FaceDetectorYN.create(W_DET, "", (1024, 672), 0.6, 0.3, 50) if refine_faces else None
        self.faces_refined = 0

    @torch.no_grad()
    def __call__(self, photo, min_face=20, box_scale=2.0):
        if isinstance(photo, (str, os.PathLike)):
            photo = Image.open(photo)
        photo = prepare(photo, self.long_side)
        W, H = photo.size
        main = to_pil(self.G(to_tensor(photo)[None].to(self.device))[0])
        self.faces_refined = 0
        if self.R is None:
            return main
        out = np.asarray(main, dtype=np.float32).copy()
        self.det.setInputSize((W, H))
        _, faces = self.det.detect(cv2.cvtColor(np.asarray(photo), cv2.COLOR_RGB2BGR))
        faces = [] if faces is None else list(faces)
        for f in sorted(faces, key=lambda f: -f[3]):  # largest first, smaller pasted on top
            x, y, w, h = f[:4]
            if not (min_face <= h <= self.max_face):
                continue  # very large faces are already drawn well by the main model
            side = int(min(box_scale * max(w, h), W, H))
            cx, cy = x + w / 2, y + h / 2 - 0.08 * h
            x0 = int(np.clip(cx - side / 2, 0, W - side)); y0 = int(np.clip(cy - side / 2, 0, H - side))
            crop = photo.crop((x0, y0, x0 + side, y0 + side)).resize((256, 256), Image.BICUBIC)
            face = to_pil(self.R(to_tensor(crop)[None].to(self.device))[0]).resize((side, side), Image.LANCZOS)
            face = np.asarray(face, dtype=np.float32)
            base = out[y0:y0 + side, x0:x0 + side]
            m = feather_mask(side)
            ring = ((m > 0.02) & (m < 0.98))[..., 0]  # match colours where the two pictures meet
            if ring.sum() > 50:
                face = face + (base[ring].mean(0) - face[ring].mean(0))
            out[y0:y0 + side, x0:x0 + side] = base * (1 - m) + np.clip(face, 0, 255) * m
            self.faces_refined += 1
        return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def cartoonify(photo, device=None, refine_faces=True):
    """One-call convenience: cartoonify a photo and return the result as a PIL Image.

        from cartoonify import cartoonify
        cartoonify("photo.jpg").save("cartoon.jpg")
    """
    if isinstance(photo, (str, os.PathLike)):
        photo = Image.open(photo)
    return Cartoonifier(device=device, refine_faces=refine_faces)(photo)


def main():
    p = argparse.ArgumentParser(description="Turn photos into illustrations.")
    p.add_argument("--input", required=True, help="one image file, or a folder of images")
    p.add_argument("--out", default="results", help="folder for the results")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if there is one)")
    p.add_argument("--threads", type=int, default=4, help="CPU threads to use when running on cpu")
    p.add_argument("--long_side", type=int, default=1024, help="longer side of the result in px (trained at 1024)")
    p.add_argument("--max_face", type=int, default=320, help="redraw faces up to this height in px")
    p.add_argument("--no_face_refiner", action="store_true", help="main model only")
    p.add_argument("--compare", action="store_true", help="also save photo and result side by side")
    opt = p.parse_args()

    torch.set_num_threads(opt.threads)
    os.makedirs(opt.out, exist_ok=True)
    t0 = time.time()
    c = Cartoonifier(opt.device, not opt.no_face_refiner, opt.max_face, opt.long_side)
    print(f"Running on {c.device}. Models loaded in {time.time() - t0:.1f}s", flush=True)

    files = ([os.path.join(opt.input, f) for f in sorted(os.listdir(opt.input))]
             if os.path.isdir(opt.input) else [opt.input])
    files = [f for f in files if f.lower().endswith(EXTS)]
    if not files:
        raise SystemExit(f"No images found in {opt.input}")
    for path in files:
        photo = Image.open(path)
        t = time.time()
        result = c(photo)
        dt = time.time() - t
        stem = os.path.splitext(os.path.basename(path))[0]
        result.save(os.path.join(opt.out, f"{stem}.jpg"), quality=93)
        if opt.compare:
            src = prepare(photo, opt.long_side)
            both = Image.new("RGB", (src.width * 2, src.height))
            both.paste(src, (0, 0)); both.paste(result, (src.width, 0))
            both.save(os.path.join(opt.out, f"{stem}_compare.jpg"), quality=90)
        print(f"{stem}: {result.width}x{result.height}, {c.faces_refined} face(s) redrawn, {dt:.1f}s", flush=True)
    print(f"Done. {len(files)} photo(s) saved in {opt.out}")


if __name__ == "__main__":
    main()
