import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

# Step 3 of training (gpt-oss-20b version): turn each filing's text into numeric features.
# Drop-in replacement for claude_labeler.py: same input files, same text_features.csv output,
# so train_models.py doesn't change.
#
# Works with any OpenAI-compatible host. Set OSS_API_KEY, and pick a provider below:
#   OpenRouter: base https://openrouter.ai/api/v1      model openai/gpt-oss-20b
#   Groq:       base https://api.groq.com/openai/v1    model openai/gpt-oss-20b
#   Together:   base https://api.together.xyz/v1       model openai/gpt-oss-20b
#   Ollama (free, on your own GPU): base http://localhost:11434/v1  model gpt-oss:20b  (any OSS_API_KEY)
# Safe to stop and re-run; it skips filings already in text_features.csv.

BASE_URL = os.environ.get("OSS_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.environ.get("OSS_MODEL", "openai/gpt-oss-20b")
FILINGS_CSV = "historical_sec_8k_2016_2024.csv"
LABELS_CSV = "labels.csv"
OUTPUT_CSV = "text_features.csv"
WORKERS = 8          # requests in flight at once (NaviGator allows 10); slows down automatically if rate-limited
RPM_LIMIT = 110      # requests per minute (NaviGator allows 120)
CHUNK_ROWS = 1000    # filings read from the CSV (and saved) at a time; keeps memory low
FILINGS_PER_REQUEST = 5  # several filings per request: the per-minute limit counts requests, not tokens
MAX_TOKENS = 3000    # room for the model's short reasoning plus the JSON answers for the whole request
MAX_TEXT_CHARS = 6000   # first ~1,500 tokens: the 8-K items and press-release lead; the rest is boilerplate
ONLY_TRADABLE = True    # True: score only filings in labels.csv (liquid stocks), about 16k instead of 37k
                        # False: score every filing, so this can run at the same time as build_labels.py

EVENT_TYPES = [
    "patent_litigation", "settlement", "fda_decision", "clinical_trial_result", "earnings",
    "licensing_or_partnership", "merger_or_acquisition", "financing", "restructuring",
    "management_change", "delisting_or_compliance", "other",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "event_type": {"type": "string", "enum": EVENT_TYPES},
        "sentiment": {"type": "number", "description": "-1 very bad for the stock, 0 neutral, +1 very good"},
        "surprise": {"type": "number", "description": "0 fully expected/routine, 1 completely unexpected"},
        "materiality": {"type": "number", "description": "0 immaterial, 1 affects the company's main product or survival"},
        "binary_catalyst_ahead": {"type": "boolean", "description": "Does the filing announce a dated upcoming decision (PDUFA date, trial readout, ruling)?"},
    },
    "required": ["event_type", "sentiment", "surprise", "materiality", "binary_catalyst_ahead"],
    "additionalProperties": False,
}

# One request carries several filings, so the answer is a list with one entry per filing
BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, **SCHEMA["properties"]},
                "required": ["id"] + SCHEMA["required"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["results"],
    "additionalProperties": False,
}

# The schema is also spelled out in the prompt, because not every host enforces response_format.
SYSTEM = (
    "You score SEC 8-K filings for an event-driven trading model. The company, ticker and dates "
    "have been replaced with placeholders. Judge only from the text in front of you. Do not try to "
    "identify the company or use anything you know about what happened afterwards.\n\n"
    "You get several filings, each starting with '### Filing <id>'. Score each one on its own. "
    'Answer with only a JSON object {"results": [...]} holding one entry per filing, '
    "each with the filing's id and these fields:\n"
    f"- event_type: one of {', '.join(EVENT_TYPES)}\n"
    "- sentiment: number from -1 (very bad for the stock) to 1 (very good), 0 neutral\n"
    "- surprise: number from 0 (fully expected/routine) to 1 (completely unexpected)\n"
    "- materiality: number from 0 (immaterial) to 1 (affects the main product or survival)\n"
    "- binary_catalyst_ahead: true if the filing announces a dated upcoming decision "
    "(PDUFA date, trial readout, ruling), else false"
)

STOPWORDS = {"INC", "CORP", "CO", "LTD", "PLC", "LLC", "THE", "AND", "HOLDINGS", "GROUP", "COMPANY", "INTERNATIONAL"}
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"

# Some hosts reject one of these; any that cause a 400 are dropped automatically.
optional_params = {
    "reasoning_effort": "low",   # less thinking = fewer output tokens and lower cost
    "response_format": {"type": "json_schema", "json_schema": {"name": "filing_scores", "strict": True, "schema": BATCH_SCHEMA}},
}


def anonymize(text, company, ticker):
    """Hides who and when, so the model can't fall back on remembering the outcome."""
    for word in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", str(company)):
        if word.upper() not in STOPWORDS:
            text = re.sub(rf"\b{re.escape(word)}\b", "[COMPANY]", text, flags=re.IGNORECASE)
    if isinstance(ticker, str) and ticker:
        text = re.sub(rf"\b{re.escape(ticker)}\b", "[TICKER]", text)
    text = re.sub(rf"\b({MONTHS})\s+\d{{1,2}},?\s+(19|20)\d{{2}}\b", "[DATE]", text)
    text = re.sub(r"\b\d{1,2}/\d{1,2}/(19|20)?\d{2}\b", "[DATE]", text)
    text = re.sub(r"\b(19|20)\d{2}\b", "[YEAR]", text)
    return text


def check_fields(data):
    """Checks one filing's scores and clamps them into range; None if unusable."""
    try:
        clamp = lambda x, lo, hi: max(lo, min(hi, float(x)))
        catalyst = data["binary_catalyst_ahead"]
        return {
            "event_type": data["event_type"] if data["event_type"] in EVENT_TYPES else "other",
            "sentiment": clamp(data["sentiment"], -1, 1),
            "surprise": clamp(data["surprise"], 0, 1),
            "materiality": clamp(data["materiality"], 0, 1),
            "binary_catalyst_ahead": catalyst if isinstance(catalyst, bool) else str(catalyst).lower() == "true",
        }
    except (ValueError, KeyError, TypeError):
        return None


def parse_answers(text):
    """Pulls {"results": [...]} out of the reply; returns {filing id: scores}."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return {}
    try:
        results = json.loads(match.group(0)).get("results", [])
    except (ValueError, AttributeError):
        return {}
    out = {}
    for item in results if isinstance(results, list) else []:
        scores = check_fields(item) if isinstance(item, dict) else None
        if scores is not None and str(item.get("id", "")).isdigit():
            out[int(item["id"])] = scores
    return out


class Throttle:
    """Keeps requests under the host's rate limit: halves the number in flight on every
    rate-limit error, pauses everyone briefly, then speeds back up while requests succeed."""

    def __init__(self, limit):
        self.max, self.limit, self.active = limit, float(limit), 0
        self.cond = threading.Condition()
        self.pause_until = self.last_cut = self.next_start = 0.0
        self.ok = 0

    def __enter__(self):
        with self.cond:
            while self.active >= int(self.limit):
                self.cond.wait()
            self.active += 1
            # Space request starts evenly to stay under the per-minute limit
            self.next_start = max(self.next_start + 60.0 / RPM_LIMIT, time.time(), self.pause_until)
            start_at = self.next_start
        time.sleep(max(0.0, start_at - time.time()))

    def __exit__(self, *exc):
        with self.cond:
            self.active -= 1
            self.cond.notify_all()

    def rate_limited(self, retry_after):
        with self.cond:
            self.pause_until = max(self.pause_until, time.time() + retry_after)
            if time.time() - self.last_cut > retry_after:  # cut once per pause, not once per request
                self.last_cut = time.time()
                self.limit = max(1.0, self.limit / 2)
                print(f"  [rate-limited: pausing {retry_after:.0f}s, now {int(self.limit)} at a time]", flush=True)

    def succeeded(self):
        with self.cond:
            self.ok += 1
            if self.ok % 25 == 0 and self.limit < self.max:
                self.limit += 1
                self.cond.notify_all()


throttle = Throttle(WORKERS)
errors = Counter()  # why requests failed, shown with the progress lines


def score_batch(rows, session):
    """Scores a few filings in one request. Returns (scores for each filing that worked, tokens in, tokens out)."""
    headers = {"Authorization": f"Bearer {os.environ.get('OSS_API_KEY', 'none')}"}
    texts = {i: anonymize(str(r["Text"])[:MAX_TEXT_CHARS], r["Company"], r["Ticker"]) for i, r in enumerate(rows, 1)}
    pending = {i: rows[i - 1] for i in texts}  # filings still without a usable answer
    answers, tokens_in, tokens_out, max_tokens = [], 0, 0, MAX_TOKENS

    for attempt in range(8):
        if not pending:
            break
        prompt = "\n\n".join(f"### Filing {i}\nItems: {r['Items']}\n\n{texts[i]}" for i, r in pending.items())
        body = {
            "model": MODEL,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            **optional_params,
        }
        try:
            with throttle:
                res = session.post(f"{BASE_URL}/chat/completions", json=body, headers=headers, timeout=300)
        except requests.RequestException as e:
            errors[type(e).__name__] += 1
            time.sleep(5 * (attempt + 1))
            continue
        if res.status_code != 200:
            errors[f"HTTP {res.status_code}"] += 1
        if res.status_code == 400:
            # Drop an optional setting this host doesn't support, then retry
            sent = [k for k in ("reasoning_effort", "response_format") if k in body]
            if any(k not in optional_params for k in sent):
                continue  # another request already dropped a setting; retry without it
            dropped = next((k for k in sent if k in res.text), sent[0] if sent else None)
            if dropped is None:
                print(f"  [request failed: {res.text[:200]}]")
                break
            if optional_params.pop(dropped, None) is not None:
                print(f"  [host rejected {dropped}; continuing without it]")
            continue
        if res.status_code == 429:
            try:
                wait = float(res.headers.get("Retry-After", 10))
            except ValueError:
                wait = 10.0
            throttle.rate_limited(min(120.0, max(2.0, wait)))
            continue
        if res.status_code in (500, 502, 503, 504):
            time.sleep(min(60, 5 * (attempt + 1)) * random.uniform(0.5, 1.5))
            continue
        if res.status_code != 200:
            print(f"  [request failed: HTTP {res.status_code} {res.text[:200]}]")
            break
        data = res.json()
        usage = data.get("usage") or {}
        tokens_in += usage.get("prompt_tokens", 0)
        tokens_out += usage.get("completion_tokens", 0)
        found = parse_answers(data["choices"][0]["message"].get("content"))
        for i, scores in found.items():
            if i in pending:
                scores["Accession"] = pending.pop(i)["Accession"]
                answers.append(scores)
        if found:
            throttle.succeeded()
        if pending:
            # Some filings got no answer, usually because the model ran out of room: ask again for just those
            errors["filings missing from an answer"] += len(pending)
            max_tokens = min(3 * MAX_TOKENS, max_tokens * 2)
    return answers, tokens_in, tokens_out


def filing_chunks(done):
    """Reads the big filings CSV a few thousand rows at a time, so it never has to fit in memory."""
    tradable = set(pd.read_csv(LABELS_CSV, usecols=["Accession"])["Accession"]) if ONLY_TRADABLE else None
    for chunk in pd.read_csv(FILINGS_CSV, usecols=["Accession", "Company", "Ticker", "Items", "Text"],
                             dtype=str, chunksize=CHUNK_ROWS):
        chunk = chunk.dropna(subset=["Ticker"]).fillna({"Text": "", "Items": ""})
        chunk = chunk[(chunk["Text"].str.len() > 200) & ~chunk["Accession"].isin(done)]  # skip empty and finished
        if tradable is not None:
            chunk = chunk[chunk["Accession"].isin(tradable)]
        chunk["Text"] = chunk["Text"].str.slice(0, MAX_TEXT_CHARS)
        rows = chunk.to_dict("records")
        yield [rows[i:i + FILINGS_PER_REQUEST] for i in range(0, len(rows), FILINGS_PER_REQUEST)]


def main():
    if "OSS_API_KEY" not in os.environ and "localhost" not in BASE_URL:
        raise SystemExit("Set OSS_API_KEY to your provider's API key first.")
    session = requests.Session()

    # One quick test call first, so a wrong key, URL or model name shows its real error
    print(f"Testing {MODEL} at {BASE_URL}...", flush=True)
    try:
        res = session.post(f"{BASE_URL}/chat/completions", timeout=60,
                           headers={"Authorization": f"Bearer {os.environ.get('OSS_API_KEY', 'none')}"},
                           json={"model": MODEL, "max_tokens": 50, "messages": [{"role": "user", "content": "Say ok"}]})
    except requests.RequestException as e:
        raise SystemExit(f"Can't reach {BASE_URL}: {e}")
    if res.status_code != 200:
        raise SystemExit(f"Test call to {BASE_URL} with model {MODEL} failed: HTTP {res.status_code} {res.text[:500]}")
    print("Test call OK", flush=True)

    done = set()
    if os.path.exists(OUTPUT_CSV):
        done = set(pd.read_csv(OUTPUT_CSV, usecols=["Accession"])["Accession"])
    print(f"Scoring filings ({len(done)} already done)...", flush=True)

    n, failed, tokens_in, tokens_out, start = 0, 0, 0, 0, time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for groups in filing_chunks(done):
            buffer = []
            futures = {pool.submit(score_batch, g, session): len(g) for g in groups}
            for fut in as_completed(futures):
                answers, t_in, t_out = fut.result()
                prev = n
                n += futures[fut]
                failed += futures[fut] - len(answers)
                tokens_in, tokens_out = tokens_in + t_in, tokens_out + t_out
                buffer.extend(answers)
                if n // 500 > prev // 500:
                    mins = (time.time() - start) / 60
                    print(f"  {n} done in {mins:.0f} min, {failed} failed, "
                          f"tokens so far {tokens_in / 1e6:.1f}M in / {tokens_out / 1e6:.1f}M out, "
                          f"{int(throttle.limit)} at a time", flush=True)
                    if errors:
                        print(f"    retried/failed requests so far: {dict(errors)}", flush=True)
            if buffer:  # save after every chunk
                pd.DataFrame(buffer).to_csv(OUTPUT_CSV, mode="a", index=False, header=not os.path.exists(OUTPUT_CSV))

    print(f"\nDone: {n - failed} scored, {failed} failed. Re-run to retry the failed ones.")
    print(f"Tokens used: {tokens_in / 1e6:.1f}M input, {tokens_out / 1e6:.1f}M output")
    if errors:
        print(f"Request errors along the way (most were retried): {dict(errors)}")


if __name__ == "__main__":
    main()
