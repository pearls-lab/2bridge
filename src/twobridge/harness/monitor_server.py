"""Local Qwen completion server. Run in a separate Qwen-compatible environment.

This module never imports the policy backend or its patched Transformers modules.
"""

import argparse
import base64
import io
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from .clients import SYSTEM, parse_yes

logger = logging.getLogger(__name__)


class QwenMonitor:
    def __init__(self, model_path, device):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16).to(device).eval()
        self.device = device

    def judge(self, data):
        import torch

        images = []
        for encoded in data["images"]:
            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                images.append(image.convert("RGB"))
        n_ref = data["n_ref"]
        content = [{"type": "text", "text": "START images:"}]
        content += [{"type": "image", "image": x} for x in images[:n_ref]]
        content += [{"type": "text", "text": "NOW images:"}]
        content += [{"type": "image", "image": x} for x in images[n_ref:]]
        content += [{"type": "text", "text": (
            f"The robot's current subtask is: {data['subgoal']}\n"
            "Comparing the start images to the current images, has the robot "
            "completed this subtask? Answer with exactly one word: yes or no.")}]
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True)
        raw = self.processor.batch_decode(output[:, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)[0].strip()
        return {"complete": parse_yes(raw), "raw": raw}


def make_handler(model):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200 if self.path == "/healthz" else 404, {"ready": self.path == "/healthz"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                n_ref = data.get("n_ref")
                if (data.get("query_type") != "completion" or type(n_ref) is not int
                        or not 1 <= n_ref <= 3 or not isinstance(data.get("images"), list)
                        or len(data["images"]) != 2 * n_ref
                        or not isinstance(data.get("subgoal"), str) or not data["subgoal"].strip()):
                    raise ValueError("Expected matching START/NOW images and a subgoal")
                started = time.monotonic()
                with lock:
                    result = model.judge(data)
                result["generation_time_ms"] = (time.monotonic() - started) * 1000
                self.reply(200, result)
            except (ValueError, KeyError, TypeError) as exc:
                self.reply(400, {"error": str(exc)})
            except Exception:
                logger.exception("Monitor inference failed")
                self.reply(500, {"error": "Monitor inference failed"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    model = QwenMonitor(args.model_path, args.device)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model))
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
