#!/usr/bin/env python3
"""
Flask + TinyLlama chatbot for Isvaryam.
Works entirely on CPU and deploys on Render Free Tier.
"""
from __future__ import annotations

import os
import time
import json
import random
import logging
from collections import defaultdict
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
from bson import ObjectId

# ─────────────────────────────────  LLM  ───────────────────────────────── #
from llama_cpp import Llama, ChatMessage

LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/tinyllama.gguf")
LLAMA_TEMP       = float(os.getenv("LLAMA_TEMP", "0.7"))
LLAMA_TOP_P      = float(os.getenv("LLAMA_TOP_P", "0.95"))
CTX_SIZE         = 2048   # keep within 1 GB RAM

llm = Llama(
    model_path = LLAMA_MODEL_PATH,
    n_ctx      = CTX_SIZE,
    n_threads  = os.cpu_count() or 2,
    temperature= LLAMA_TEMP,
    top_p      = LLAMA_TOP_P,
)

# ───────────────────────────────  Flask & Mongo  ───────────────────────── #
app = Flask(__name__,
            static_url_path='/static',
            static_folder='static',
            template_folder='templates')

logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MONGO_URI = os.environ.get("MONGO_URI", "")
client    = MongoClient(MONGO_URI) if MONGO_URI else None
db        = client["isvaryam"] if client else None
products  = db["products"]  if db else None
reviews   = db["reviews"]   if db else None

# ─────────────────────────────  In‑memory state  ───────────────────────── #
conversation_context: dict[str, dict] = defaultdict(dict)

# ───────────────────────  Domain data (abridged sample)  ────────────────── #
with open("ingredients.json") as f:
    ingredients_data = json.load(f)
with open("contact.json") as f:
    contact_data = json.load(f)
with open("faqs.json") as f:
    faqs_data = json.load(f)

product_data = {
    "groundnut oil": {"description": "...", "benefits": ["..."], "attributes": {"rating": 4.7}},
    "coconut oil":   {"description": "...", "benefits": ["..."], "attributes": {"rating": 4.9}},
    # 🔎 add remaining products…
}
alias_map = {"combo pack": "super pack"}   # etc.

# ──────────────────────────  Intent detection (stub)  ───────────────────── #
def detect_intent(text: str) -> str | None:
    t = text.lower()
    if any(greet in t for greet in ("hi", "hello", "hey")):
        return "greet"
    if "price" in t:
        return "price"
    # expand with your full keyword lists…
    return None

def handle_intents(text: str, user_id: str) -> str | None:
    intent = detect_intent(text)
    if intent == "greet":
        hour = datetime.now().hour
        sal  = "Good morning ☀️" if hour < 12 else (
               "Good afternoon 🌤️" if hour < 17 else "Good evening 🌙")
        return f"{sal} I'm Isvaryam's assistant. How can I help you?"
    if intent == "price":
        return "Our price list is coming right up! (demo reply)"
    return None

# ───────────────────────────────  Llama helper  ─────────────────────────── #
def llama_chat(user_msg: str, history: list[ChatMessage]) -> str:
    messages = history + [{"role": "user", "content": user_msg}]
    res = llm.create_chat_completion(messages=messages, stream=False)
    return res["choices"][0]["message"]["content"].strip()

# ───────────────────────────────  Routes  ───────────────────────────────── #
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_msg = request.json.get("message", "")
    user_id  = request.json.get("user_id", "anon")

    # 1. keyword intents
    intent_reply = handle_intents(user_msg, user_id)
    if intent_reply:
        return jsonify(response=intent_reply)

    # 2. fallback → TinyLlama
    hist = conversation_context[user_id].get("history", [])
    llama_reply = llama_chat(user_msg, hist)

    # store limited history (keep last 10 exchanges)
    hist.extend([
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": llama_reply}
    ])
    conversation_context[user_id]["history"] = hist[-10:]

    return jsonify(response=llama_reply)

@app.route("/feedback", methods=["POST"])
def feedback():
    logging.info(f"Feedback: {request.json}")
    return jsonify(status="success")

# ───────────────────────────────  Main  ─────────────────────────────────── #
if __name__ == "__main__":
    app.run(debug=True)
