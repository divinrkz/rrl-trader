"""Download SUBSTITUTE FX data for Experiment 3 (paper Sec. IV-B) into data/raw/dukascopy/.

The paper used Olsen & Associates HFDF96 half-hourly USD/GBP quotes (Jan-Aug 1996),
which are commercial. We use Dukascopy's free historical GBP/USD tick data, which
has real bid AND ask quotes, for:
  * Jan-Aug 2004: the test period (closest available year to 1996)
  * Sep-Dec 2003: pilot data for choosing settings (never used in the test)

One file per hour: https://datafeed.dukascopy.com/datafeed/GBPUSD/YYYY/MM/DD/HHh_ticks.bi5
(MM is 0-based in the URL). Hours with no trading (weekends) are empty files.
Files are cached; re-running only fetches what is missing. Weekend hours
(Fri 22:00 - Sun 20:59 UTC) are skipped: the market is closed and those files were empty
in all 17 weeks downloaded first (src/data_fx.market_closed).
The server rate-limits (HTTP 429): we use one connection and an adaptive pause
between requests (class Pacer).

Usage: python experiments/fetch_fx.py
"""
import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.data_fx import market_closed  # noqa: E402

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "dukascopy",
                   "GBPUSD")
URL = "https://datafeed.dukascopy.com/datafeed/GBPUSD/{y}/{m0:02d}/{d:02d}/{h:02d}h_ticks.bi5"
START, END = datetime(2003, 9, 1), datetime(2004, 9, 1)     # END exclusive


def local_path(t):
    return os.path.join(RAW, f"{t:%Y}", f"{t:%m}", f"{t:%d}", f"{t:%H}h_ticks.bi5")


class Pacer:
    """Adaptive delay between requests (one connection). After a 429 the delay doubles
    (up to 120 s); after each success it shrinks by 10% (down to 0.5 s), so the
    downloader settles near the rate the server allows."""
    def __init__(self, delay=5.0):
        self.delay = delay

    def wait(self):
        time.sleep(self.delay)

    def ok(self):
        self.delay = max(0.5, self.delay * 0.9)

    def limited(self):
        self.delay = min(120.0, self.delay * 2)


def fetch(t, pacer, retries=10):
    path = local_path(t)
    if os.path.exists(path):
        return "cached"
    url = URL.format(y=t.year, m0=t.month - 1, d=t.day, h=t.hour)
    err = None
    for k in range(retries):
        pacer.wait()
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
            pacer.ok()
            return "ok"
        except urllib.error.HTTPError as e:
            err = e
            if e.code == 429:
                pacer.limited()
            else:
                time.sleep(2 ** k)
        except Exception as e:  # noqa: BLE001  (network errors: retry with backoff)
            err = e
            time.sleep(2 ** k)
    return f"FAILED {url}: {err}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args()
    hours = []
    t = START
    while t < END:
        if not market_closed(t):
            hours.append(t)
        t += timedelta(hours=1)
    pacer, results = Pacer(), []
    for i, t in enumerate(hours):
        r = fetch(t, pacer)
        results.append(r)
        if r != "cached" and (i % 25 == 0 or r.startswith("FAILED")):
            print(f"{datetime.now():%H:%M} {i}/{len(hours)} {t} {r[:100]} "
                  f"(delay {pacer.delay:.1f}s)", flush=True)
    failed = [r for r in results if r.startswith("FAILED")]
    print(f"{len(hours)} hours: {results.count('ok')} downloaded, "
          f"{results.count('cached')} cached, {len(failed)} failed")
    for f in failed[:20]:
        print(f)


if __name__ == "__main__":
    main()
