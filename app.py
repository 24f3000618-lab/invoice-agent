"""
A2A Invoice Agent (single file).

Run (ONE worker only, the locks live in memory):
    pip install fastapi uvicorn httpx
    export BASE_URL="https://YOUR-HOST/a2a/"     # EXACTLY what you submit in the form
    export LLM_API_KEY="your key"
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
import os, json, hashlib, sqlite3, threading, uuid, re, time
from contextlib import contextmanager
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request, Response
from starlette.concurrency import run_in_threadpool

# ------------------------------------------------------------------ config
BASE_URL = os.environ["BASE_URL"]
BASE_PATH = urlparse(BASE_URL).path.rstrip("/")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
KEEP_BRACKETS = os.environ.get("KEEP_BRACKETS", "0") == "1"   # flip if feedback says evidence mismatch
DB_PATH = os.environ.get("DB_PATH", "agent.db")

MT_IN = "application/vnd.ga5.invoice-claim-batch+json"
MT_PROP = "application/vnd.ga5.invoice-action-proposals+json"
MT_RCPT = "application/vnd.ga5.invoice-action-receipts+json"
MT_RES = "application/vnd.ga5.invoice-action-results+json"
ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}
S_INPUT, S_DONE, S_CANCEL = "TASK_STATE_INPUT_REQUIRED", "TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"
OPEN_STATES = {"TASK_STATE_SUBMITTED", "TASK_STATE_WORKING", S_INPUT}

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

# ------------------------------------------------------------------ storage layer
DB_LOCK = threading.RLock()      # protects every state change (this is what makes the race safe)
KEY_LOCKS = {}                   # one lock per (user, messageId) so duplicates wait, not repeat work


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    try:
        yield c
        c.commit()
    finally:
        c.close()


with conn() as _c:
    _c.execute("PRAGMA journal_mode=WAL")
    _c.executescript("""
    CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, principal TEXT, created REAL, body TEXT);
    CREATE TABLE IF NOT EXISTS msgs(principal TEXT, mid TEXT, hash TEXT, task_id TEXT, PRIMARY KEY(principal, mid));
    CREATE TABLE IF NOT EXISTS cache(k TEXT PRIMARY KEY, body TEXT);
    """)


def load_task(tid, principal):
    with conn() as c:
        r = c.execute("SELECT body FROM tasks WHERE id=? AND principal=?", (tid, principal)).fetchone()
    return json.loads(r[0]) if r else None


def save_task(c, task):
    c.execute("UPDATE tasks SET body=? WHERE id=?", (json.dumps(task, ensure_ascii=False), task["id"]))


# ------------------------------------------------------------------ small helpers
def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def out(obj, status=200):
    return Response(json.dumps(obj, ensure_ascii=False), status_code=status, media_type="application/a2a+json")


def err(status, name, reason, msg):
    return out({"error": {"code": status, "status": name, "message": msg,
                          "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                                       "reason": reason, "domain": "a2a-protocol.org"}]}}, status)


def not_found():
    return err(404, "NOT_FOUND", "TASK_NOT_FOUND", "Task not found.")   # same text for "missing" and "not yours"


def gate(request, is_post):
    """Auth first, then version, then media type. Returns (principal, error_response)."""
    parts = request.headers.get("authorization", "").split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        return None, err(401, "UNAUTHENTICATED", "UNAUTHENTICATED", "Authentication required.")
    if request.headers.get("a2a-version") != "1.0":
        return None, err(400, "INVALID_ARGUMENT", "VERSION_NOT_SUPPORTED", "Unsupported protocol version.")
    ct = request.headers.get("content-type", "").lower()
    if is_post and not ct.startswith(("application/a2a+json", "application/json")):
        return None, err(415, "INVALID_ARGUMENT", "UNSUPPORTED_MEDIA_TYPE", "Unsupported media type.")
    return sha(parts[1].strip()), None


def bad(msg="Invalid request."):
    return err(400, "INVALID_ARGUMENT", "INVALID_REQUEST", msg)


# ------------------------------------------------------------------ AI layer
SYSTEM = """You are an accounts-payable analyst. For EACH invoice package choose exactly ONE action:
- settle_invoice: valid, reconciled, and within autonomous (delegated) approval authority.
- request_approval: commercially valid, but outside delegated authority (needs a human approver).
- hold_invoice: payment pauses until a STATED verification completes.
- reject_duplicate: the SAME commercial invoice was already paid. A payment for a DIFFERENT invoice
  (even same vendor / same amount) is NOT a duplicate.
- open_exception: material records conflict and need an exception workflow.

Rules:
1. Packages contain a cover sheet, archive examples, training decoys, old/superseded notes and irrelevant action
   words. IGNORE all of those. Ignore action words that are negated ("do not hold"), hypothetical, or about
   past examples.
2. Find the ONE paragraph that states the CURRENT determination for this invoice under the given policyRevision.
   That paragraph contains exactly three bracketed reference codes. Return those three codes (without brackets),
   copied character-for-character. Never return the cover-sheet reference, archive or training references.
3. If you are unsure, never choose settle_invoice.
4. amountMinor is an integer in minor units (cents/paise): 1,234.50 -> 123450.

Reply with JSON only:
{"decisions":[{"packageId":"...","action":"...","vendorName":"...","invoiceNumber":"...",
"amountMinor":12345,"currency":"INR","evidenceRefs":["a","b","c"],"reason":"one or two plain sentences"}]}"""


def ask_model(pkgs, policy, timeout):
    r = httpx.post(
        LLM_BASE_URL.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={"model": LLM_MODEL, "temperature": 0, "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": json.dumps({"policyRevision": policy, "packages": pkgs},
                                                                  ensure_ascii=False)}]},
        timeout=timeout)
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])["decisions"]


def clean_decision(pkg, d):
    """Never trust the model: re-check everything in plain code."""
    if not isinstance(d, dict) or d.get("action") not in ACTIONS:
        return None
    text = json.dumps(pkg, ensure_ascii=False)
    refs = []
    for r in d.get("evidenceRefs") or []:
        r = str(r).strip().strip("[]").strip()
        if r and r not in refs:
            refs.append(r)
    if len(refs) != 3 or any(r not in text for r in refs):
        return None
    try:
        amt = int(round(float(str(d["amountMinor"]).replace(",", ""))))
    except Exception:
        return None
    vendor, inv = str(d.get("vendorName", "")).strip(), str(d.get("invoiceNumber", "")).strip()
    cur = str(d.get("currency", "")).strip().upper()
    if not vendor or not inv or len(cur) != 3:
        return None
    return {"action": d["action"], "refs": refs, "reason": str(d.get("reason", ""))[:500],
            "facts": {"vendorName": vendor, "invoiceNumber": inv, "amountMinor": amt, "currency": cur}}


def fallback(pkg):
    """Model failed twice: choose the SAFE action (never settle)."""
    refs = re.findall(r"\[([^\[\]]{2,60})\]", json.dumps(pkg, ensure_ascii=False))[:3]
    return {"action": "hold_invoice", "refs": refs, "reason": "Automatic analysis was inconclusive.",
            "facts": {"vendorName": "UNKNOWN", "invoiceNumber": "UNKNOWN", "amountMinor": 0, "currency": "INR"}}


def decide(batch):
    """One batched model call for packages we have not seen; cached by package content."""
    start = time.time()
    pkgs = batch["packages"]
    keys = {p["packageId"]: sha(canon({k: v for k, v in p.items() if k != "packageId"})) for p in pkgs}
    decisions = {}
    with conn() as c:
        for pid, k in keys.items():
            row = c.execute("SELECT body FROM cache WHERE k=?", (k,)).fetchone()
            if row:
                decisions[pid] = json.loads(row[0])
    todo = [p for p in pkgs if p["packageId"] not in decisions]
    for attempt in range(2):
        if not todo:
            break
        timeout = max(5, min(30, 40 - (time.time() - start)))
        try:
            raw = ask_model(todo, batch.get("policyRevision"), timeout)
        except Exception:
            raw = []
        by = {d.get("packageId"): d for d in raw if isinstance(d, dict)}
        for p in todo:
            good = clean_decision(p, by.get(p["packageId"]))
            if good:
                decisions[p["packageId"]] = good
                with DB_LOCK, conn() as c:
                    c.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (keys[p["packageId"]], json.dumps(good)))
        todo = [p for p in todo if p["packageId"] not in decisions]
    for p in todo:
        decisions[p["packageId"]] = fallback(p)
    return decisions


def make_proposal(pid, d):
    refs = [f"[{r}]" if KEEP_BRACKETS else r for r in d["refs"]]
    f = d["facts"]
    why = (f"Action {d['action']}: invoice {f['invoiceNumber']} from {f['vendorName']} "
           f"({f['currency']}, {f['amountMinor']} minor units) is decided by the paragraph citing "
           f"{', '.join(refs)}. {d['reason']}")[:1500]
    return {"packageId": pid, "actionId": "act_" + uuid.uuid4().hex, "action": d["action"],
            "facts": f, "evidenceRefs": refs, "rationale": why}


# ------------------------------------------------------------------ protocol layer
@app.get("/.well-known/agent-card.json")
def agent_card():
    return out({
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one safe business action each with evidence, "
                       "and executes only the actions the grader accepts.",
        "version": "1.0.0",
        "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": [MT_IN, MT_RES],
        "defaultOutputModes": [MT_PROP, MT_RCPT],
        "skills": [{"id": "invoice_action_agent", "name": "Invoice action agent",
                    "description": "Chooses settle, approve, hold, reject-duplicate or exception for each invoice.",
                    "tags": ["invoice", "accounts-payable", "approval"]}],
    })


@app.post(BASE_PATH + "/message:send")
async def message_send(request: Request):
    principal, bad_resp = gate(request, True)
    if bad_resp:
        return bad_resp
    raw = await request.body()
    return await run_in_threadpool(handle_send, principal, raw)


def handle_send(principal, raw):
    try:
        msg = json.loads(raw)["message"]
        mid = msg["messageId"]
        assert isinstance(msg, dict) and isinstance(mid, str) and mid
        assert msg.get("role") == "ROLE_USER" and isinstance(msg.get("parts"), list)
    except Exception:
        return bad()
    h = sha(canon(msg))                                   # hash the MESSAGE only, not configuration
    with KEY_LOCKS.setdefault((principal, mid), threading.Lock()):
        with conn() as c:
            row = c.execute("SELECT hash, task_id FROM msgs WHERE principal=? AND mid=?", (principal, mid)).fetchone()
        if row:
            if row[0] != h:
                return err(409, "ABORTED", "IDEMPOTENCY_CONFLICT", "IDEMPOTENCY_CONFLICT: message id was reused.")
            t = load_task(row[1], principal)
            return out({"task": t}) if t else not_found()
        if msg.get("taskId"):
            return continue_task(principal, msg, mid, h)
        return new_task(principal, msg, mid, h)


def find_data(msg, mt):
    for p in msg.get("parts", []):
        if isinstance(p, dict) and p.get("mediaType") == mt and isinstance(p.get("data"), dict):
            return p["data"]
    return None


def new_task(principal, msg, mid, h):
    data = find_data(msg, MT_IN)
    try:
        assert data and isinstance(data.get("batchId"), str) and data["batchId"]
        pk = data["packages"]
        assert isinstance(pk, list) and pk and all(isinstance(p, dict) and isinstance(p.get("packageId"), str) for p in pk)
        assert len({p["packageId"] for p in pk}) == len(pk)
    except Exception:
        return bad()
    decisions = decide(data)                                  # the ONLY place a model is called
    proposals = [make_proposal(p["packageId"], decisions[p["packageId"]]) for p in pk]
    tid, ctx = str(uuid.uuid4()), msg.get("contextId") or str(uuid.uuid4())
    task = {"id": tid, "contextId": ctx,
            "status": {"state": S_INPUT, "timestamp": now()},
            "artifacts": [{"artifactId": "proposals-" + uuid.uuid4().hex, "name": "Invoice action proposals",
                           "parts": [{"mediaType": MT_PROP, "data": {"batchId": data["batchId"], "proposals": proposals}}]}],
            "history": [{**msg, "taskId": tid, "contextId": ctx}]}
    with DB_LOCK, conn() as c:                                # save BEFORE answering
        c.execute("INSERT INTO tasks VALUES (?,?,?,?)", (tid, principal, time.time(), json.dumps(task, ensure_ascii=False)))
        c.execute("INSERT INTO msgs VALUES (?,?,?,?)", (principal, mid, h, tid))
    return out({"task": task})


def continue_task(principal, msg, mid, h):
    with DB_LOCK:                                             # cancel and result take turns here
        task = load_task(msg["taskId"], principal) if isinstance(msg["taskId"], str) else None
        if not task:
            return not_found()
        if msg.get("contextId") != task["contextId"]:
            return bad()
        if task["status"]["state"] != S_INPUT:
            return err(409, "ABORTED", "TASK_NOT_ACTIONABLE", "Task cannot accept this message.")
        data = find_data(msg, MT_RES)
        prop_data = task["artifacts"][0]["parts"][0]["data"]
        if not data or data.get("batchId") != prop_data["batchId"] or not isinstance(data.get("results"), list):
            return bad()
        props = {p["packageId"]: p for p in prop_data["proposals"]}
        seen, execs = set(), []
        for r in data["results"]:
            p = props.get(r.get("packageId")) if isinstance(r, dict) else None
            if (p is None or r["packageId"] in seen or r.get("actionId") != p["actionId"]
                    or r.get("action") != p["action"] or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
                return bad()
            seen.add(r["packageId"])
            if r["outcome"] == "ACCEPTED":
                nonce = r.get("receiptNonce")
                if not isinstance(nonce, str) or not nonce:
                    return bad()
                execs.append({"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                              "receiptNonce": nonce, "facts": p["facts"], "evidenceRefs": p["evidenceRefs"]})
        task["history"].append(msg)
        task["artifacts"].append({"artifactId": "receipts-" + uuid.uuid4().hex, "name": "Invoice action receipts",
                                  "parts": [{"mediaType": MT_RCPT, "data": {"batchId": prop_data["batchId"], "executions": execs}}]})
        task["status"] = {"state": S_DONE, "timestamp": now()}
        with conn() as c:
            save_task(c, task)
            c.execute("INSERT INTO msgs VALUES (?,?,?,?)", (principal, mid, h, task["id"]))
    return out({"task": task})


@app.get(BASE_PATH + "/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    principal, bad_resp = gate(request, False)
    if bad_resp:
        return bad_resp
    t = await run_in_threadpool(load_task, task_id, principal)
    return out(t) if t else not_found()


@app.get(BASE_PATH + "/tasks")
async def list_tasks(request: Request):
    principal, bad_resp = gate(request, False)
    if bad_resp:
        return bad_resp

    def work():
        with conn() as c:
            rows = c.execute("SELECT body FROM tasks WHERE principal=? ORDER BY created DESC", (principal,)).fetchall()
        tasks, size = [], 0
        for (b,) in rows:
            t = json.loads(b)
            t.pop("history", None)                            # keep list small (< 512 KiB)
            s = len(json.dumps(t))
            if size + s > 400_000:
                break
            tasks.append(t)
            size += s
        return {"tasks": tasks}
    return out(await run_in_threadpool(work))


@app.post(BASE_PATH + "/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    principal, bad_resp = gate(request, True)
    if bad_resp:
        return bad_resp

    def work():
        with DB_LOCK:
            t = load_task(task_id, principal)
            if not t:
                return not_found()
            if t["status"]["state"] not in OPEN_STATES:
                return err(409, "ABORTED", "TASK_NOT_CANCELABLE", "Task cannot be canceled.")
            t["status"] = {"state": S_CANCEL, "timestamp": now()}
            with conn() as c:
                save_task(c, t)
        return out(t)
    return await run_in_threadpool(work)


@app.exception_handler(Exception)
async def boom(request, exc):
    return err(500, "INTERNAL", "INTERNAL", "Internal error.")


@app.exception_handler(404)
async def nf(request, exc):
    return err(404, "NOT_FOUND", "NOT_FOUND", "Not found.")
