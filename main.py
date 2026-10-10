"""
WhatsApp Personal Assistant - MVP
WhatsApp Cloud API -> FastAPI -> (Whisper transcription) -> Claude -> WhatsApp
"""
import os
import hmac
import hashlib
import logging

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, BackgroundTasks, Query, HTTPException
from fastapi.responses import PlainTextResponse
from anthropic import Anthropic
from openai import OpenAI

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("assistant")

# ---------- Config ----------
WA_TOKEN = os.environ["WA_TOKEN"]
PHONE_NUMBER_ID = os.environ["WA_PHONE_NUMBER_ID"]
VERIFY_TOKEN = os.environ["WA_VERIFY_TOKEN"]
OWNER_NUMBER = os.environ["OWNER_NUMBER"]          # e.g. 9725XXXXXXXX (no +)
APP_SECRET = os.getenv("WA_APP_SECRET")            # optional, enables signature check
GRAPH_VERSION = os.getenv("GRAPH_VERSION", "v21.0")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-1")
GRAPH = f"https://graph.facebook.com/{GRAPH_VERSION}"

SYSTEM_PROMPT = """אתה העוזר האישי של לידור, ומדבר איתו בוואטסאפ.
ענה בעברית, קצר וענייני, בפורמט שנוח לקרוא בטלפון.

כשההודעה מתחילה ב-[תמלול הקלטה], זו הקלטה של פגישה או הודעה קולית. החזר:
*סיכום* - 3 עד 6 שורות
*החלטות* - אם היו
*משימות* - מי, מה, עד מתי (אם נאמר)
*פולואפ מוצע* - פגישות או הודעות שכדאי לקבוע
אם ההקלטה קצרה ופשוטה (בקשה או תזכורת), פשוט ענה עליה כמו על הודעה רגילה.

אל תמציא פרטים שלא נאמרו. אם משהו לא ברור, שאל."""

app = FastAPI()
claude = Anthropic()   # reads ANTHROPIC_API_KEY
oai = OpenAI()         # reads OPENAI_API_KEY
http = httpx.Client(timeout=60)

seen_ids: set[str] = set()
history: list[dict] = []      # in-memory for MVP; move to a DB later
MAX_HISTORY = 20


# ---------- WhatsApp helpers ----------
def wa_headers() -> dict:
    return {"Authorization": f"Bearer {WA_TOKEN}"}


def send_text(to: str, text: str) -> None:
    # WhatsApp text limit is 4096 chars
    for i in range(0, len(text), 4000):
        r = http.post(
            f"{GRAPH}/{PHONE_NUMBER_ID}/messages",
            headers=wa_headers(),
            json={
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": {"body": text[i:i + 4000]},
            },
        )
        if r.status_code >= 300:
            log.error("send failed: %s %s", r.status_code, r.text)


def download_media(media_id: str) -> tuple[bytes, str]:
    meta = http.get(f"{GRAPH}/{media_id}", headers=wa_headers()).json()
    data = http.get(meta["url"], headers=wa_headers()).content
    return data, meta.get("mime_type", "audio/ogg")


# ---------- AI ----------
def transcribe(audio: bytes, mime: str) -> str:
    ext = "ogg" if "ogg" in mime else mime.split("/")[-1].split(";")[0]
    result = oai.audio.transcriptions.create(
        model=TRANSCRIBE_MODEL,
        file=(f"audio.{ext}", audio, mime),
        language="he",
    )
    return result.text


def ask_claude(user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    msgs = history[-MAX_HISTORY:]
    while msgs and msgs[0]["role"] != "user":
        msgs = msgs[1:]
    resp = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=msgs,
    )
    answer = "".join(b.text for b in resp.content if b.type == "text").strip()
    history.append({"role": "assistant", "content": answer})
    del history[:-MAX_HISTORY * 2]
    return answer


# ---------- Message handling ----------
def handle_message(msg: dict) -> None:
    sender = msg["from"]
    try:
        mtype = msg.get("type")
        if mtype == "text":
            reply = ask_claude(msg["text"]["body"])
        elif mtype == "audio":
            send_text(sender, "🎧 מתמלל...")
            audio, mime = download_media(msg["audio"]["id"])
            transcript = transcribe(audio, mime)
            log.info("transcript: %s", transcript[:200])
            reply = ask_claude(f"[תמלול הקלטה]\n{transcript}")
        else:
            reply = "כרגע אני מבין רק טקסט והקלטות קוליות."
        send_text(sender, reply)
    except Exception as e:
        log.exception("handler error")
        send_text(sender, f"⚠️ שגיאה: {e}")


def valid_signature(body: bytes, header: str | None) -> bool:
    if not APP_SECRET:
        return True
    if not header:
        return False
    expected = "sha256=" + hmac.new(APP_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)


# ---------- Routes ----------
@app.get("/webhook")
def verify(
    mode: str = Query(None, alias="hub.mode"),
    token: str = Query(None, alias="hub.verify_token"),
    challenge: str = Query(None, alias="hub.challenge"),
):
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403)


@app.post("/webhook")
async def receive(request: Request, background: BackgroundTasks):
    body = await request.body()
    if not valid_signature(body, request.headers.get("X-Hub-Signature-256")):
        raise HTTPException(status_code=401)

    payload = await request.json()
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            for msg in change.get("value", {}).get("messages", []):
                if msg["id"] in seen_ids:          # Meta retries webhooks
                    continue
                seen_ids.add(msg["id"])
                if msg["from"] != OWNER_NUMBER:    # only you can use the bot
                    log.warning("ignored message from %s", msg["from"])
                    continue
                background.add_task(handle_message, msg)
    return {"ok": True}                            # reply 200 fast


@app.get("/")
def health():
    return {"status": "alive"}
