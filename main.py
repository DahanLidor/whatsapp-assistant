"""
WhatsApp Personal Assistant
WhatsApp Cloud API -> FastAPI -> (Whisper) -> Claude (+ Google Calendar tools) -> WhatsApp
"""
import os
import hmac
import html
import json
import time
import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, BackgroundTasks, Query, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse, RedirectResponse
from anthropic import Anthropic
from openai import OpenAI

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("assistant")

# ---------- Config ----------
WA_TOKEN = os.environ["WA_TOKEN"]
PHONE_NUMBER_ID = os.environ["WA_PHONE_NUMBER_ID"]
VERIFY_TOKEN = os.environ["WA_VERIFY_TOKEN"]
OWNER_NUMBER = os.environ["OWNER_NUMBER"]
APP_SECRET = os.getenv("WA_APP_SECRET")
GRAPH_VERSION = os.getenv("GRAPH_VERSION", "v26.0")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-1")
GRAPH = f"https://graph.facebook.com/{GRAPH_VERSION}"

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
PUBLIC_URL = os.getenv("PUBLIC_URL", "").rstrip("/")
TIMEZONE = os.getenv("TIMEZONE", "Asia/Jerusalem")
TZ = ZoneInfo(TIMEZONE)
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "google_token.json")
REDIRECT_URI = f"{PUBLIC_URL}/oauth/callback"
GCAL = "https://www.googleapis.com/calendar/v3/calendars/primary"

app = FastAPI()
claude = Anthropic()
oai = OpenAI()
http = httpx.Client(timeout=60)

seen_ids: set[str] = set()
history: list[dict] = []
MAX_HISTORY = 20


def system_prompt() -> str:
    now = datetime.now(TZ)
    return f"""אתה העוזר האישי של לידור, ומדבר איתו בוואטסאפ.
ענה בעברית, קצר וענייני, בפורמט שנוח לקרוא בטלפון.

עכשיו: {now:%A %d/%m/%Y %H:%M} (אזור זמן {TIMEZONE}).

יומן:
- יש לך כלים לקרוא, ליצור, לעדכן ולמחוק אירועים ביומן Google של לידור.
- כשמבקשים לקבוע פגישה בלי משך, קבע שעה אחת.
- אם חסר יום או שעה, שאל לפני שאתה יוצר.
- לפני מחיקה, ודא שזיהית את האירוע הנכון. אם יש כמה מתאימים, שאל.
- אחרי יצירה או שינוי, אשר בשורה אחת: מה, מתי.
- זמנים בכלים: בפורמט YYYY-MM-DDTHH:MM:SS בשעון מקומי.
{"- היומן עדיין לא מחובר. אם מבקשים משהו ביומן, אמור שצריך לחבר אותו קודם." if not google_connected() else ""}

כשההודעה מתחילה ב-[תמלול הקלטה], זו הקלטה של פגישה או הודעה קולית.
אם זו פגישה, החזר: *סיכום*, *החלטות*, *משימות* (מי, מה, עד מתי), *פולואפ מוצע*.
אם זו בקשה קצרה, טפל בה כמו בהודעה רגילה (כולל פעולות ביומן).

אל תמציא פרטים שלא נאמרו. אם משהו לא ברור, שאל."""


# ---------- WhatsApp ----------
def wa_headers() -> dict:
    return {"Authorization": f"Bearer {WA_TOKEN}"}


def send_text(to: str, text: str) -> None:
    for i in range(0, len(text), 4000):
        r = http.post(
            f"{GRAPH}/{PHONE_NUMBER_ID}/messages",
            headers=wa_headers(),
            json={"messaging_product": "whatsapp", "to": to, "type": "text",
                  "text": {"body": text[i:i + 4000]}},
        )
        if r.status_code >= 300:
            log.error("send failed: %s %s", r.status_code, r.text)


def download_media(media_id: str) -> tuple[bytes, str]:
    meta = http.get(f"{GRAPH}/{media_id}", headers=wa_headers()).json()
    data = http.get(meta["url"], headers=wa_headers()).content
    return data, meta.get("mime_type", "audio/ogg")


def transcribe(audio: bytes, mime: str) -> str:
    ext = "ogg" if "ogg" in mime else mime.split("/")[-1].split(";")[0]
    result = oai.audio.transcriptions.create(
        model=TRANSCRIBE_MODEL, file=(f"audio.{ext}", audio, mime), language="he")
    return result.text


# ---------- Google Calendar ----------
_google = {"access_token": None, "expires_at": 0.0}
_oauth_states: dict[str, float] = {}


def google_connected() -> bool:
    return os.path.exists(TOKEN_FILE)


def google_access_token() -> str:
    if _google["access_token"] and time.time() < _google["expires_at"] - 60:
        return _google["access_token"]
    with open(TOKEN_FILE) as f:
        refresh = json.load(f)["refresh_token"]
    r = http.post("https://oauth2.googleapis.com/token", data={
        "client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET,
        "refresh_token": refresh, "grant_type": "refresh_token"})
    if r.status_code >= 300:
        raise RuntimeError(f"Google token refresh failed: {r.text[:200]}")
    d = r.json()
    _google.update(access_token=d["access_token"], expires_at=time.time() + d.get("expires_in", 3600))
    return d["access_token"]


def gcal(method: str, path: str = "", **kw) -> dict:
    r = http.request(method, f"{GCAL}{path}",
                     headers={"Authorization": f"Bearer {google_access_token()}"}, **kw)
    if r.status_code >= 300:
        raise RuntimeError(f"Calendar API {r.status_code}: {r.text[:300]}")
    return r.json() if r.content else {}


def to_rfc3339(s: str) -> str:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt.isoformat()


def ev_time(s: str) -> dict:
    return {"dateTime": to_rfc3339(s), "timeZone": TIMEZONE}


def slim(ev: dict) -> dict:
    return {"id": ev.get("id"), "title": ev.get("summary", "(ללא כותרת)"),
            "start": ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date"),
            "end": ev.get("end", {}).get("dateTime") or ev.get("end", {}).get("date"),
            "location": ev.get("location"), "description": ev.get("description")}


def tool_list_events(start: str, end: str, query: str | None = None) -> dict:
    params = {"timeMin": to_rfc3339(start), "timeMax": to_rfc3339(end),
              "singleEvents": "true", "orderBy": "startTime", "maxResults": 50}
    if query:
        params["q"] = query
    items = gcal("GET", "/events", params=params).get("items", [])
    return {"events": [slim(e) for e in items]}


def tool_create_event(title: str, start: str, end: str,
                      description: str | None = None, location: str | None = None) -> dict:
    body = {"summary": title, "start": ev_time(start), "end": ev_time(end)}
    if description:
        body["description"] = description
    if location:
        body["location"] = location
    return {"created": slim(gcal("POST", "/events", json=body))}


def tool_update_event(event_id: str, title: str | None = None, start: str | None = None,
                      end: str | None = None, description: str | None = None,
                      location: str | None = None) -> dict:
    body = {}
    if title:
        body["summary"] = title
    if start:
        body["start"] = ev_time(start)
    if end:
        body["end"] = ev_time(end)
    if description is not None:
        body["description"] = description
    if location is not None:
        body["location"] = location
    return {"updated": slim(gcal("PATCH", f"/events/{event_id}", json=body))}


def tool_delete_event(event_id: str) -> dict:
    gcal("DELETE", f"/events/{event_id}")
    return {"deleted": event_id}


TOOL_FUNCS = {
    "list_events": tool_list_events,
    "create_event": tool_create_event,
    "update_event": tool_update_event,
    "delete_event": tool_delete_event,
}

_T = {"type": "string", "description": "YYYY-MM-DDTHH:MM:SS local time"}
TOOLS = [
    {"name": "list_events", "description": "List calendar events in a time range, optionally filtered by text.",
     "input_schema": {"type": "object", "properties": {
         "start": _T, "end": _T, "query": {"type": "string"}}, "required": ["start", "end"]}},
    {"name": "create_event", "description": "Create a calendar event.",
     "input_schema": {"type": "object", "properties": {
         "title": {"type": "string"}, "start": _T, "end": _T,
         "description": {"type": "string"}, "location": {"type": "string"}},
         "required": ["title", "start", "end"]}},
    {"name": "update_event", "description": "Update fields of an existing event (by id from list_events).",
     "input_schema": {"type": "object", "properties": {
         "event_id": {"type": "string"}, "title": {"type": "string"}, "start": _T, "end": _T,
         "description": {"type": "string"}, "location": {"type": "string"}},
         "required": ["event_id"]}},
    {"name": "delete_event", "description": "Delete an event by id (from list_events).",
     "input_schema": {"type": "object", "properties": {"event_id": {"type": "string"}},
                      "required": ["event_id"]}},
]


# ---------- Claude ----------
def ask_claude(user_text: str) -> str:
    history.append({"role": "user", "content": user_text})
    msgs = list(history[-MAX_HISTORY:])
    while msgs and msgs[0]["role"] != "user":
        msgs = msgs[1:]

    kwargs = {"model": CLAUDE_MODEL, "max_tokens": 2000, "system": system_prompt()}
    if google_connected():
        kwargs["tools"] = TOOLS

    answer = ""
    for _ in range(8):
        resp = claude.messages.create(messages=msgs, **kwargs)
        if resp.stop_reason != "tool_use":
            answer = "".join(b.text for b in resp.content if b.type == "text").strip()
            break
        msgs.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type != "tool_use":
                continue
            try:
                out, err = TOOL_FUNCS[b.name](**b.input), False
            except Exception as e:
                log.exception("tool %s failed", b.name)
                out, err = {"error": str(e)}, True
            results.append({"type": "tool_result", "tool_use_id": b.id,
                            "content": json.dumps(out, ensure_ascii=False), "is_error": err})
        msgs.append({"role": "user", "content": results})
    else:
        answer = "לא הצלחתי להשלים את הפעולה, נסה לנסח שוב."

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
def verify(mode: str = Query(None, alias="hub.mode"),
           token: str = Query(None, alias="hub.verify_token"),
           challenge: str = Query(None, alias="hub.challenge")):
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
            value = change.get("value", {})
            for st in value.get("statuses", []):
                if st.get("status") == "failed":
                    log.error("delivery failed: %s", st.get("errors"))
            for msg in value.get("messages", []):
                if msg["id"] in seen_ids:
                    continue
                seen_ids.add(msg["id"])
                if msg["from"] != OWNER_NUMBER:
                    log.warning("ignored message from %s", msg["from"])
                    continue
                background.add_task(handle_message, msg)
    return {"ok": True}


def _page(title: str, text: str) -> HTMLResponse:
    return HTMLResponse(f"""<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title></head><body dir="rtl" style="font-family:sans-serif;max-width:600px;margin:40px auto;padding:0 16px;line-height:1.6">
<h2>{title}</h2><p>{text}</p></body></html>""")


@app.get("/oauth/start")
def oauth_start(key: str = ""):
    if not hmac.compare_digest(key.encode(), VERIFY_TOKEN.encode()):
        raise HTTPException(status_code=403)
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and PUBLIC_URL):
        raise HTTPException(status_code=500, detail="Google is not configured in .env")
    state = secrets.token_urlsafe(24)
    _oauth_states[state] = time.time()
    params = {"client_id": GOOGLE_CLIENT_ID, "redirect_uri": REDIRECT_URI, "response_type": "code",
              "scope": "https://www.googleapis.com/auth/calendar", "access_type": "offline",
              "prompt": "consent", "state": state}
    return RedirectResponse("https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params))


@app.get("/oauth/callback")
def oauth_callback(code: str = "", state: str = "", error: str = ""):
    if error:
        return _page("❌ החיבור נכשל", html.escape(error))
    ts = _oauth_states.pop(state, None)
    if not ts or time.time() - ts > 600:
        return _page("❌ הקישור פג", "פתח שוב את קישור החיבור ונסה מחדש.")
    r = http.post("https://oauth2.googleapis.com/token", data={
        "code": code, "client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI, "grant_type": "authorization_code"})
    if r.status_code >= 300:
        return _page("❌ החיבור נכשל", html.escape(r.text[:300]))
    d = r.json()
    if "refresh_token" not in d:
        return _page("❌ חסר אישור קבוע", "גוגל לא החזירו refresh token. נסה שוב את קישור החיבור.")
    with open(TOKEN_FILE, "w") as f:
        json.dump({"refresh_token": d["refresh_token"]}, f)
    os.chmod(TOKEN_FILE, 0o600)
    _google.update(access_token=d["access_token"], expires_at=time.time() + d.get("expires_in", 3600))
    return _page("✅ היומן חובר בהצלחה", "אפשר לסגור את הדף. העוזר יכול עכשיו לקרוא ולקבוע פגישות.")


@app.get("/calendar/test")
def calendar_test(key: str = ""):
    if not hmac.compare_digest(key.encode(), VERIFY_TOKEN.encode()):
        raise HTTPException(status_code=403)
    if not google_connected():
        return {"connected": False}
    now = datetime.now(TZ).replace(microsecond=0)
    week = now + timedelta(days=7)
    return {"connected": True, **tool_list_events(now.replace(tzinfo=None).isoformat(),
                                                  week.replace(tzinfo=None).isoformat())}


@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return """<html><head><meta charset="utf-8"><title>Privacy Policy</title></head>
<body style="font-family:sans-serif;max-width:700px;margin:40px auto;line-height:1.6">
<h1>Privacy Policy - Lidor Assistant</h1>
<p>Lidor Assistant is a private, single-user assistant operated by Lidor Dahan. It only processes messages sent by its owner.</p>
<p>Messages and voice notes are processed solely to generate replies, using the Anthropic and OpenAI APIs. Calendar access is used only to read and manage the owner's own events. Data is not sold or shared for any other purpose.</p>
<p>Messages from any other number are ignored and not stored.</p>
<p>Contact: dahanlid555@gmail.com</p>
</body></html>"""


@app.get("/")
def health():
    return {"status": "alive", "calendar": google_connected()}
