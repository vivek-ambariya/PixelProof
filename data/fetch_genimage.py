"""Fetch a capped native-resolution sample from GenImage, without downloading it all.

Why this exists
    Every image PixelProof trained on is 32x32-native (CIFAKE), so "sharp,
    detailed image" was a perfect stand-in for "AI-generated" inside the
    training distribution -- no real training example was ever sharp. On a
    genuine native-resolution photograph the shortcut inverts and the detector
    calls real photos fake with high confidence. That is not fixable with
    augmentation: blurring or re-compressing a 32x32 image cannot add real
    high-frequency detail to the *real* class. It needs real photographs at
    native resolution, and generated images at comparable resolution, in
    training.

    GenImage supplies both: ImageNet photographs (``nature``) paired with
    generated images (``ai``) from eight generators, all at native resolution.
    The challenge brief names GenImage explicitly as permitted extra training
    data (§4.1), provided it is cited -- see README §3.

Why it streams instead of downloading
    The per-generator archives run from ~0.9 GB (BigGAN) to ~7.7 GB
    (Midjourney), and we want a few thousand images, not all of them. The Hugging
    Face CDN honours HTTP Range requests, so this reads the zip's central
    directory from the tail of the file and then fetches only the byte ranges
    holding the entries it actually wants. Pulling 4,000 images costs roughly
    the size of those 4,000 images, not the size of the archive.

Output layout matches the convention prepare.py already understands, so no
change to prepare.py is needed:

    data/raw/real/genimage_imagenet/*        <- native-resolution photographs
    data/raw/ai/<generator>_genimage/*       <- native-resolution generated

Usage:
    python data/fetch_genimage.py --list                    # inspect structure only
    python data/fetch_genimage.py                           # 2000 real + 2000 ai
    python data/fetch_genimage.py --generator SD_v15 --per-class 1000
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE = "https://huggingface.co/datasets/shimei123/Genimage/resolve/main"

# Archive per generator. Sizes are the full archive; streaming means we pay only
# for the entries pulled, so the default is chosen for usefulness, not size.
# Midjourney is the brief's own example of a "newer diffusion model /
# Midjourney-class output" (§4.1), which makes it the most valuable addition.
GENERATORS = {
    "Midjourney": "Midjourney.zip",
    "SD_v14": "SD_v14.zip",
    "SD_v15": "SD_v15.zip",
    "ADM": "ADM.zip",
    "glide": "glide.zip",
    "wukong": "wukong.zip",
    "VQDM": "VQDM.zip",
    "BigGAN": "BigGAN.zip",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPEG", ".JPG", ".PNG")

# 8 MiB cache blocks. Entries are picked contiguously (see --stride), so one
# block typically serves a run of consecutive images: the dominant cost is the
# number of range requests, not the bytes, and this keeps that count low.
BLOCK = 8 << 20


class HTTPRangeFile(io.RawIOBase):
    """A seekable read-only file over HTTP Range requests, with block caching.

    ``zipfile.ZipFile`` only needs read/seek/tell, and it reads the tail first
    (to find the central directory) then seeks to individual entry offsets. Both
    patterns are served here from 1 MiB blocks, so extracting many small entries
    costs roughly one request per megabyte touched rather than one per read.
    """

    def __init__(self, url: str, timeout: int = 60) -> None:
        self.url = url
        self.timeout = timeout
        self._pos = 0
        self._blocks: dict[int, bytes] = {}
        self.requests = 0
        self.bytes_fetched = 0
        self.size = self._content_length()

    def _content_length(self) -> int:
        req = urllib.request.Request(self.url, method="HEAD")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            n = r.headers.get("content-length")
            if n is None:
                raise RuntimeError("server did not report a content-length")
            return int(n)

    def _fetch(self, index: int) -> bytes:
        blk = self._blocks.get(index)
        if blk is not None:
            return blk
        start = index * BLOCK
        end = min(start + BLOCK, self.size) - 1
        if start > end:
            return b""
        req = urllib.request.Request(self.url,
                                     headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    blk = r.read()
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))
        self.requests += 1
        self.bytes_fetched += len(blk)
        self._blocks[index] = blk
        return blk

    # --- file protocol ---------------------------------------------------
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.size + offset
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self._pos
        n = max(0, min(n, self.size - self._pos))
        out = bytearray()
        while n > 0:
            index = self._pos // BLOCK
            off = self._pos - index * BLOCK
            blk = self._fetch(index)
            if not blk:
                break
            take = blk[off:off + n]
            if not take:
                break
            out += take
            self._pos += len(take)
            n -= len(take)
        return bytes(out)

    def readinto(self, b) -> int:  # RawIOBase contract, used by BufferedReader
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


# Directory names the two halves go by. GenImage's own release uses
# ``{ai,nature}``; this mirror repackages it in the ForenSynths convention
# ``{1_fake,0_real}``. Both are accepted so either layout works unchanged.
AI_DIRS = {"ai", "1_fake", "fake"}
REAL_DIRS = {"nature", "0_real", "real"}


def classify(name: str) -> str | None:
    """'ai', 'nature', or None -- read from the entry's path, not its filename.

    Keying on the directory rather than the extension matters here: in this
    mirror the real half is ``.JPEG`` and the generated half is ``.png``, so
    extension and label are perfectly correlated *inside the archive* and must
    not be allowed to stand in for the label anywhere downstream.
    """
    if name.endswith("/") or not name.endswith(IMAGE_EXTS):
        return None
    parts = set(name.split("/"))
    if parts & AI_DIRS:
        return "ai"
    if parts & REAL_DIRS:
        return "nature"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generator", default="Midjourney", choices=sorted(GENERATORS),
                    help="which GenImage generator subset to stream from")
    ap.add_argument("--per-class", type=int, default=2000,
                    help="images to pull per class (real and ai each)")
    ap.add_argument("--list", action="store_true",
                    help="print the archive's structure and exit, fetching no images")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth entry. 1 (default) is contiguous and "
                         "far faster; raise it only if you specifically want "
                         "the sample spread across the whole archive.")
    ap.add_argument("--no-reencode", dest="reencode", action="store_false",
                    help="keep each entry's original bytes. Off by default "
                         "because this mirror stores reals as .JPEG and fakes "
                         "as .png, making format a perfect proxy for the label.")
    ap.add_argument("--real-dir", default=None)
    ap.add_argument("--ai-dir", default=None)
    args = ap.parse_args()

    url = f"{BASE}/{GENERATORS[args.generator]}"
    print(f"opening {args.generator}  ({GENERATORS[args.generator]})")
    fp = HTTPRangeFile(url)
    print(f"  archive size {fp.size / 1e9:.2f} GB -- streaming, not downloading")

    zf = zipfile.ZipFile(io.BufferedReader(fp, buffer_size=BLOCK))
    names = zf.namelist()
    print(f"  {len(names):,} entries in the central directory "
          f"({fp.bytes_fetched / 1e6:.1f} MB fetched to read it)")

    buckets: dict[str, list[str]] = {"ai": [], "nature": []}
    for n in names:
        kind = classify(n)
        if kind:
            buckets[kind].append(n)
    for kind, items in buckets.items():
        print(f"  {kind:7} {len(items):,} images")

    if args.list:
        print("\nsample entries:")
        for kind, items in buckets.items():
            for n in items[:4]:
                print(f"  {kind:7} {n}")
        return

    if not buckets["ai"] or not buckets["nature"]:
        sys.exit("archive did not contain both an 'ai' and a 'nature' folder")

    real_dir = Path(args.real_dir) if args.real_dir else ROOT / "data" / "raw" / "real" / "genimage_imagenet"
    ai_dir = Path(args.ai_dir) if args.ai_dir else ROOT / "data" / "raw" / "ai" / f"{args.generator.lower()}_genimage"

    targets = [("nature", real_dir, "genimage_real"),
               ("ai", ai_dir, f"{args.generator.lower()}")]

    from PIL import Image  # local import: only needed when actually writing

    for kind, out_dir, prefix in targets:
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = [p for p in out_dir.iterdir() if p.is_file()]
        if len(existing) >= args.per_class:
            print(f"\n{kind}: {len(existing)} already in {out_dir}, skipping")
            continue

        # Contiguous by default. Spacing the picks out across the archive
        # sounds fairer but is ~15x slower here: every image then lands in its
        # own cache block, so a 4,000-image sample would stream several GB.
        # Contiguity costs nothing in diversity because both halves are stored
        # in an order uncorrelated with content -- ImageNet validation indices
        # are class-shuffled, and the generated half is numbered by prompt id.
        pool = buckets[kind]
        picks = pool[::args.stride][:args.per_class]

        print(f"\n{kind}: writing {len(picks)} images -> {out_dir}")
        sizes: list[tuple[int, int]] = []
        written = 0
        t0 = time.time()
        for i, name in enumerate(picks):
            try:
                raw = zf.read(name)
                with Image.open(io.BytesIO(raw)) as im:
                    im.load()
                    sizes.append(im.size)
                    im = im.convert("RGB")
                    if args.reencode:
                        # In this archive the real half is .JPEG and the
                        # generated half is .png, so container format is a
                        # perfect stand-in for the label. Left alone, a model
                        # can score "JPEG artefacts present" instead of
                        # "photographic", which fails in deployment exactly the
                        # way the 32x32 resolution shortcut does. Writing both
                        # halves through one encoder at one quality removes the
                        # shortcut. It does cost some of the PNG-preserved
                        # high-frequency generator fingerprint, which is the
                        # deliberate trade: a weaker but honest signal beats a
                        # strong confounded one.
                        im.save(out_dir / f"{prefix}_{written:05d}.jpg",
                                format="JPEG", quality=95)
                    else:
                        ext = Path(name).suffix.lower()
                        if ext not in (".jpg", ".jpeg", ".png"):
                            ext = ".png"
                        (out_dir / f"{prefix}_{written:05d}{ext}").write_bytes(raw)
                    written += 1
            except Exception as e:
                print(f"  skip {name}: {type(e).__name__}")
                continue
            if written and written % 250 == 0:
                rate = written / max(time.time() - t0, 1e-6)
                print(f"  {written}/{len(picks)}  {rate:.1f} img/s  "
                      f"{fp.bytes_fetched / 1e6:.0f} MB fetched", flush=True)

        if sizes:
            w = sorted(s[0] for s in sizes)
            h = sorted(s[1] for s in sizes)
            mid = len(sizes) // 2
            print(f"  wrote {written}; median resolution {w[mid]}x{h[mid]}, "
                  f"min {min(w)}x{min(h)}, max {max(w)}x{max(h)}")

    print(f"\ntotal fetched {fp.bytes_fetched / 1e6:.0f} MB in {fp.requests} range "
          f"requests (archive is {fp.size / 1e9:.2f} GB)")
    print("\nNext: rebuild the splits so these are indexed --")
    print("  python data/prepare.py --holdout-generators progan")


if __name__ == "__main__":
    main()
