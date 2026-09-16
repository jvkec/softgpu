#!/usr/bin/env python3
"""Render results/*.json as markdown tables, optionally side by side.

  scripts/report.py results/baseline.json                       # all metrics
  scripts/report.py results/baseline.json results/ring.json -m p50_ns ops_per_s
"""
import argparse, json, sys
from collections import OrderedDict

def load(path):
    d = json.load(open(path))
    rows = OrderedDict()
    for r in d["rows"]:
        rows[(r["bench"], r["variant"])] = {k: v for k, v in r.items() if k not in ("bench", "variant")}
    return d["tag"], rows

def fmt(v):
    if v is None: return ""
    if abs(v) >= 1000: return f"{v:,.0f}"
    if abs(v) >= 100: return f"{v:.0f}"
    if abs(v) >= 1: return f"{v:.2f}"
    return f"{v:.3f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-m", "--metrics", nargs="*", help="metrics to show (default: all)")
    ap.add_argument("-b", "--bench", help="only this benchmark")
    a = ap.parse_args()

    runs = [load(f) for f in a.files]
    keys = OrderedDict()
    for _, rows in runs:
        for k in rows: keys[k] = None
    if a.bench: keys = OrderedDict((k, None) for k in keys if k[0] == a.bench)

    for bench in OrderedDict((k[0], None) for k in keys):
        variants = [k for k in keys if k[0] == bench]
        metrics = a.metrics or sorted({m for _, rows in runs for k in variants for m in rows.get(k, {})})
        print(f"\n### {bench}\n")
        if len(runs) == 1:
            print("| variant | " + " | ".join(metrics) + " |")
            print("|---|" + "---:|" * len(metrics))
            for k in variants:
                r = runs[0][1].get(k, {})
                print(f"| {k[1]} | " + " | ".join(fmt(r.get(m)) for m in metrics) + " |")
        else:
            for m in metrics:
                print(f"**{m}**\n")
                print("| variant | " + " | ".join(t for t, _ in runs) + " | change |")
                print("|---|" + "---:|" * (len(runs) + 1))
                for k in variants:
                    vals = [rows.get(k, {}).get(m) for _, rows in runs]
                    ch = ""
                    if vals[0] and vals[-1] is not None and vals[0] != 0:
                        ch = f"{(vals[-1] / vals[0] - 1) * 100:+.0f}%"
                    print(f"| {k[1]} | " + " | ".join(fmt(v) for v in vals) + f" | {ch} |")
                print()

if __name__ == "__main__":
    main()
