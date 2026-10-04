import json
import os
import re
import time

import anthropic
import pandas as pd

# Step 3 of training: turn each filing's text into numeric features with Claude.
# Input:  historical_sec_8k_2016_2024.csv (text) + labels.csv (which filings are tradable)
# Output: text_features.csv
#
# Runs through the Message Batches API: half price, no rate-limit handling, and results
# usually come back within an hour. Safe to stop and re-run; it resumes.

MODEL = "claude-opus-5-5"   # "claude-haiku-4-5" is about a quarter of the cost if budget matters
FILINGS_CSV = "historical_sec_8k_2016_2024.csv"
LABELS_CSV = "labels.csv"
OUTPUT_CSV = "text_features.csv"
BATCH_LOG = "claude_batches.txt"   # batch ids, so a re-run picks up submitted batches
BATCH_SIZE = 5000                  # keeps each batch well under the 256 MB request limit

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

SYSTEM = (
    "You score SEC 8-K filings for an event-driven trading model. The company, ticker and dates "
    "have been replaced with placeholders. Judge only from the text in front of you. Do not try to "
    "identify the company or use anything you know about what happened afterwards."
)

STOPWORDS = {"INC", "CORP", "CO", "LTD", "PLC", "LLC", "THE", "AND", "HOLDINGS", "GROUP", "COMPANY", "INTERNATIONAL"}
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def anonymize(text, company, ticker):
    """Hides who and when, so Claude can't fall back on remembering the outcome.
    Drug names are left in; strip them too if you have a list."""
    for word in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", str(company)):
        if word.upper() not in STOPWORDS:
            text = re.sub(rf"\b{re.escape(word)}\b", "[COMPANY]", text, flags=re.IGNORECASE)
    if isinstance(ticker, str) and ticker:
        text = re.sub(rf"\b{re.escape(ticker)}\b", "[TICKER]", text)
    text = re.sub(rf"\b({MONTHS})\s+\d{{1,2}},?\s+(19|20)\d{{2}}\b", "[DATE]", text)
    text = re.sub(r"\b\d{1,2}/\d{1,2}/(19|20)?\d{2}\b", "[DATE]", text)
    text = re.sub(r"\b(19|20)\d{2}\b", "[YEAR]", text)
    return text


def build_request(row):
    output_config = {"format": {"type": "json_schema", "schema": SCHEMA}}
    if not MODEL.startswith("claude-haiku"):  # Haiku 4.5 rejects the effort setting
        output_config["effort"] = "low"
    content = anonymize(row["Text"], row["Company"], row["Ticker"])
    return {
        "custom_id": row["Accession"],
        "params": {
            "model": MODEL,
            "max_tokens": 4096,
            "system": SYSTEM,
            "output_config": output_config,
            "messages": [{"role": "user", "content": f"Items: {row['Items']}\n\n{content}"}],
        },
    }


def collect(client, batch_id):
    rows, failed = [], 0
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            failed += 1
            continue
        msg = result.result.message
        if msg.stop_reason != "end_turn":  # refusal or max_tokens: no usable answer
            failed += 1
            continue
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            failed += 1
            continue
        data["Accession"] = result.custom_id
        rows.append(data)
    return rows, failed


def main():
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    filings = pd.read_csv(FILINGS_CSV, usecols=["Accession", "Company", "Ticker", "Items", "Text"])
    tradable = set(pd.read_csv(LABELS_CSV, usecols=["Accession"])["Accession"])
    filings = filings[filings["Accession"].isin(tradable)].fillna({"Text": "", "Items": ""})

    done = set()
    if os.path.exists(OUTPUT_CSV):
        done = set(pd.read_csv(OUTPUT_CSV, usecols=["Accession"])["Accession"])
    submitted = open(BATCH_LOG).read().split() if os.path.exists(BATCH_LOG) else []

    # 1. Submit batches for filings not yet scored or already in flight
    if not submitted:
        todo = filings[~filings["Accession"].isin(done)]
        print(f"Submitting {len(todo)} filings to {MODEL} in batches of {BATCH_SIZE}...")
        for start in range(0, len(todo), BATCH_SIZE):
            chunk = todo.iloc[start:start + BATCH_SIZE]
            batch = client.messages.batches.create(requests=[build_request(r) for _, r in chunk.iterrows()])
            submitted.append(batch.id)
            with open(BATCH_LOG, "a") as fh:
                fh.write(batch.id + "\n")
            print(f"  batch {batch.id}: {len(chunk)} filings")

    # 2. Wait for each batch and save its results
    for batch_id in submitted:
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                break
            print(f"  {batch_id}: {batch.request_counts.processing} still processing...", flush=True)
            time.sleep(60)
        rows, failed = collect(client, batch_id)
        rows = [r for r in rows if r["Accession"] not in done]
        if rows:
            pd.DataFrame(rows).to_csv(OUTPUT_CSV, mode="a", index=False, header=not os.path.exists(OUTPUT_CSV))
            done.update(r["Accession"] for r in rows)
        print(f"  {batch_id}: saved {len(rows)}, failed {failed}")

    os.remove(BATCH_LOG)
    missing = len(set(filings["Accession"]) - done)
    print(f"\nDone: {len(done)} filings scored, {missing} missing. Re-run to retry the missing ones.")


if __name__ == "__main__":
    main()
