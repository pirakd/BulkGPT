#!/usr/bin/env python3
"""Scrape DiSignAtlas control/case GSM IDs for all qualifying RNA-Seq datasets."""
import asyncio
import csv
import re
import sys
import time
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm

DATASETS_CSV = Path("data/disign_atlas/Disease_information_Datasets.csv")
OUTPUT_CSV = Path("data/disign_atlas/control_case_gsms.csv")
BASE_URL = "http://www.inbirg.com/disignatlas/detail/{}"
CONCURRENCY = 80
HEADERS = {"User-Agent": "Mozilla/5.0"}
MAX_RETRIES = 10
RETRY_BACKOFF = 2.0  # seconds, doubled each attempt

import scanpy as sp 

disease_annot_data = sp.read_h5ad("/Users/daniel/Downloads/disease_annot_data.h5ad", backed="r")
annotation_df = disease_annot_data.obs
scraped_gsms = pd.read_csv("data/disign_atlas/control_case_gsms.csv")
# filter for covid-19
bulkformer_covid_19_df = annotation_df[annotation_df["label"] == "COVID-19"]
bulkformer_covid_19_df = bulkformer_covid_19_df[bulkformer_covid_19_df["GSM_ID"].str.startswith("GSM")]
scraped_gsms_csv = scraped_gsms[scraped_gsms["disease"] == "COVID-19"]

bulkformer_unique_gsms = bulkformer_covid_19_df["GSM_ID"].unique()
scraped_unique_gsms = scraped_gsms_csv["gsm"].unique()
missing_gsms_in_scraped = bulkformer_unique_gsms[~bulkformer_unique_gsms.isin(scraped_unique_gsms)]

def load_qualifying_ids() -> tuple[list[str], dict[str, dict]]:
    rows = list(csv.DictReader(DATASETS_CSV.open(encoding="latin-1")))
    ids = []
    meta = {}
    for r in rows:
        if r["library_strategy"] != "RNA-Seq":
            continue
        ids.append(r["dsaid"])
        meta[r["dsaid"]] = {"disease": r["disease"], "diseaseid": r["diseaseid"]}
    return ids, meta


def parse_gsms(dsa_id: str, html: str) -> list[dict]:
    m = re.search(r"<tr[^>]*>.*?Control\s*\|\s*Case.*?</tr>", html, re.I | re.S)
    if not m:
        return []
    row = m.group(0)
    cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.I | re.S)
    cells = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", c)).strip() for c in cells]

    idx = next((i for i, c in enumerate(cells) if "Control" in c and "Case" in c), None)
    if idx is None or idx + 2 >= len(cells):
        return []

    control_gsms = re.findall(r"GSM\d+", cells[idx + 1])
    case_gsms = re.findall(r"GSM\d+", cells[idx + 2])
    return (
        [{"dsa_id": dsa_id, "gsm": g, "status": "control"} for g in control_gsms]
        + [{"dsa_id": dsa_id, "gsm": g, "status": "case"} for g in case_gsms]
    )


async def fetch(session: aiohttp.ClientSession, sem: asyncio.Semaphore, dsa_id: str) -> list[dict]:
    url = BASE_URL.format(dsa_id)
    delay = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        async with sem:
            try:
                async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        html = await resp.text(errors="replace")
                        return parse_gsms(dsa_id, html)
                    err = f"HTTP {resp.status}"
            except Exception as e:
                err = str(e)

        print(f"  WARN {dsa_id} attempt {attempt}/{MAX_RETRIES}: {err}", file=sys.stderr)
        if attempt < MAX_RETRIES:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)

    print(f"  ERROR {dsa_id}: giving up after {MAX_RETRIES} attempts", file=sys.stderr)
    return []


async def main() -> None:
    ids, meta = load_qualifying_ids()
    print(f"Fetching {len(ids)} datasets with concurrency={CONCURRENCY}...")
    t0 = time.monotonic()

    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch(session, sem, dsa_id) for dsa_id in ids]
        results = await tqdm.gather(*tasks, total=len(ids), desc="scraping")

    rows = [
        {**row, "disease": meta[row["dsa_id"]]["disease"], "diseaseid": meta[row["dsa_id"]]["diseaseid"]}
        for batch in results for row in batch
    ]
    failed = [ids[i] for i, r in enumerate(results) if not r]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dsa_id", "disease", "diseaseid", "gsm", "status"])
        w.writeheader()
        w.writerows(rows)

    elapsed = time.monotonic() - t0
    print(f"Done in {elapsed:.1f}s — {len(rows)} GSM records written to {OUTPUT_CSV}")
    if failed:
        print(f"WARNING: {len(failed)} datasets returned no records: {failed}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
