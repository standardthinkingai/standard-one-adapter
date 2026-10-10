"""Smoke-test a running adapter, optionally including actual image bytes."""

import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30120")
    parser.add_argument(
        "--request", type=Path, default=Path(__file__).with_name("request.json")
    )
    parser.add_argument("--image", type=Path, action="append", default=[])
    args = parser.parse_args()
    body = json.loads(args.request.read_text())
    for path in args.image:
        mime = mimetypes.guess_type(path.name)[0]
        if not mime or not mime.startswith("image/"):
            parser.error(f"Cannot identify image type: {path.name}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        body.setdefault("images", []).append(f"data:{mime};base64,{encoded}")
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("JEV_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    request = Request(
        args.base_url.rstrip("/") + "/v1/systemone",
        data=json.dumps(body).encode(),
        headers=headers,
    )
    started = time.perf_counter()
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    elapsed = (time.perf_counter() - started) * 1000
    if result.get("usage", {}).get("output_tokens") != 0:
        raise RuntimeError("Expected zero generated tokens")
    if set(result.get("answers", {})) != set(body["questions"]):
        raise RuntimeError("Missing decision answers")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"End-to-end HTTP latency: {elapsed:.1f} ms (not a GPU benchmark)")


if __name__ == "__main__":
    main()
