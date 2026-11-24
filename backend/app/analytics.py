"""Effort analytics from interaction logs. Shared by the in-app Insights panel and eval/effort.py,
so the numbers in the app and in the report come from the same code."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np


def per_image_effort(events: list[dict], idle_s: float) -> list[dict]:
    events = sorted(events, key=lambda e: e["t"])
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("image_id") is None:
            continue
        who = e.get("payload", {}).get("participant") or e["session_id"][:8]
        by_key[(who, e["image_id"])].append(e)

    rows = []
    for (who, image_id), evs in by_key.items():
        done = [e for e in evs if e["event"] == "image_done"]
        if not done:
            continue  # never finished, not a valid measurement
        end_t = done[0]["t"]
        evs = [e for e in evs if e["t"] <= end_t]
        active, last = 0.0, None
        open_ts = [e["t"] for e in evs if e["event"] == "image_open"]
        for e in evs:
            if e["event"] == "image_open":
                last = e["t"]
                continue
            if last is not None:
                gap = (e["t"] - last) / 1000
                if gap <= idle_s:
                    active += gap
            last = e["t"]
            if e["event"] in ("image_leave", "tab_hidden"):
                last = None  # time away from the image does not count
        count = lambda name: sum(1 for e in evs if e["event"] == name)  # noqa: E731
        p = done[0].get("payload", {})
        rows.append({
            "participant": who, "image_id": image_id, "mode": done[0]["mode"],
            "active_s": round(active, 1), "wall_s": round((end_t - min(open_ts)) / 1000, 1) if open_ts else None,
            "clicks": count("pointer_down"), "shortcuts": count("shortcut"),
            "objects": p.get("n_annotations", 0), "by_source": json.dumps(p.get("by_source", {})),
            "sam_clicks": count("sam_click"), "proposals_accepted": count("proposal_accepted"),
            "undo": count("undo"), "deleted": count("ann_deleted"),
        })
    return rows


def acceptance(events: list[dict]) -> list[dict]:
    """Share of detector suggestions the user kept, per model version."""
    agg: dict[int, dict] = defaultdict(lambda: {"shown": 0, "accepted": 0, "rejected": 0, "discarded": 0})
    for e in events:
        p, v = e.get("payload", {}), e.get("payload", {}).get("version")
        if v is None:
            continue
        if e["event"] == "proposals_shown":
            agg[v]["shown"] += p.get("n", 0)
        elif e["event"] == "proposal_accepted":
            agg[v]["accepted"] += 1
        elif e["event"] == "proposals_accept_all":
            agg[v]["accepted"] += p.get("n", 0)
        elif e["event"] == "proposal_rejected":
            agg[v]["rejected"] += 1
        elif e["event"] == "proposals_discarded":
            agg[v]["discarded"] += p.get("n", 0)
    return [{"version": v, **d, "acceptance_rate": round(d["accepted"] / d["shown"], 3) if d["shown"] else None}
            for v, d in sorted(agg.items())]



def summarize_by_mode(rows: list[dict]) -> list[dict]:
    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)
    out = []
    for mode, rs in sorted(by_mode.items()):
        cpo = [r["clicks"] / r["objects"] for r in rs if r["objects"]]
        out.append({"mode": mode, "images": len(rs),
                    "active_s_mean": round(float(np.mean([r["active_s"] for r in rs])), 1),
                    "active_s_sd": round(float(np.std([r["active_s"] for r in rs], ddof=1)), 1) if len(rs) > 1 else 0.0,
                    "clicks_per_object": round(float(np.mean(cpo)), 2) if cpo else None,
                    "objects_per_image": round(float(np.mean([r["objects"] for r in rs])), 1)})
    return out
