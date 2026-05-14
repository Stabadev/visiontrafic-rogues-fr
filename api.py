#!/usr/bin/env python3
import os
import csv
import json
import sqlite3
from pathlib import Path
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, request, send_from_directory, abort, send_file, render_template

BASE_DIR = Path(__file__).resolve().parent

OUT_DIR = Path(os.environ.get("INFOROUTE_OUT_DIR", str(BASE_DIR / "out_cycle"))).resolve()
DB_PATH = Path(os.environ.get("INFOROUTE_DB_PATH", str(OUT_DIR / "events.sqlite3"))).resolve()
CSV_PATH = Path(os.environ.get("INFOROUTE_CSV_PATH", str(BASE_DIR / "webcams.csv"))).resolve()
STATUS_PATH = Path(os.environ.get("INFOROUTE_STATUS_PATH", str(OUT_DIR / "status.json"))).resolve()

TZ_LOCAL = ZoneInfo("Europe/Paris")

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)


# -------------------
# DB + helpers
# -------------------
def db_connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def load_cams():
    cams = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cams.append({
                "cam_id": r["id_webcam"].strip(),
                "title": r["titre"].strip(),
                "lat": float(r["latitude"]),
                "lon": float(r["longitude"]),
                "delay_ms": int(r["delai_ms"]),
                "url_image": r["url_image"].strip(),
            })
    return cams


def parse_ts_utc(ts_utc: str) -> datetime:
    return datetime.strptime(ts_utc, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def fmt_ts_utc(dt_utc: datetime) -> str:
    return dt_utc.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_out_filename(path_str: str) -> str:
    return Path(path_str).name


def parse_local_date(date_str: str | None):
    if date_str:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    return datetime.now(TZ_LOCAL).date()


def local_day_bounds_utc(local_day):
    start_local = datetime.combine(local_day, time(0, 0), tzinfo=TZ_LOCAL)
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)
    return fmt_ts_utc(start_utc), fmt_ts_utc(end_utc)


def read_status():
    try:
        if STATUS_PATH.exists():
            return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {
        "ok": False,
        "paused": None,
        "reason": "no_status_file",
        "updated_at_utc": None,
        "window_local": None,
        "next_wake_local": None,
        "next_wake_utc": None,
    }


def hhmm_to_time(hhmm: str) -> time:
    return datetime.strptime(hhmm, "%H:%M").time()


# -------------------
# Pages vitrine
# -------------------
@app.get("/")
def page_index():
    return render_template("index.html")


@app.get("/dashboard")
def page_dashboard():
    return render_template("dashboard_full.html")


@app.get("/technique")
def page_technique():
    return render_template("technique.html")


@app.get("/debug")
def page_debug():
    debug_path = (BASE_DIR / "debug.html").resolve()
    if not debug_path.exists():
        return "<h1>debug.html introuvable</h1>", 404
    return send_file(debug_path)

@app.get("/contact")
def page_contact():
    return render_template("contact.html")


# -------------------
# API
# -------------------
@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "db": str(DB_PATH),
        "out_dir": str(OUT_DIR),
        "csv": str(CSV_PATH),
        "status": str(STATUS_PATH),
    })


@app.get("/api/status")
def api_status():
    return jsonify(read_status())


@app.get("/api/cams")
def api_cams():
    return jsonify(load_cams())


@app.get("/api/events")
def api_events():
    since_id = request.args.get("since_id", default=0, type=int)
    limit = request.args.get("limit", default=100, type=int)
    limit = max(1, min(limit, 500))

    con = db_connect()
    frames = con.execute(
        """
        SELECT id, cam_id, ts_utc, raw_path, annotated_path, vehicles_count
        FROM frames
        WHERE id > ?
        ORDER BY id ASC
        LIMIT ?
        """,
        (since_id, limit)
    ).fetchall()

    frame_ids = [row["id"] for row in frames]
    det_by_frame = {fid: [] for fid in frame_ids}
    if frame_ids:
        q_marks = ",".join(["?"] * len(frame_ids))
        dets = con.execute(
            f"""
            SELECT frame_id, cls, conf, x1, y1, x2, y2
            FROM detections
            WHERE frame_id IN ({q_marks})
            ORDER BY frame_id ASC, conf DESC
            """,
            frame_ids
        ).fetchall()
        for d in dets:
            det_by_frame[d["frame_id"]].append({
                "cls": d["cls"],
                "conf": d["conf"],
                "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
            })

    events = []
    max_id = since_id
    for fr in frames:
        fid = fr["id"]
        max_id = max(max_id, fid)

        raw_name = safe_out_filename(fr["raw_path"])
        ann_name = safe_out_filename(fr["annotated_path"])

        dt_local = parse_ts_utc(fr["ts_utc"]).astimezone(TZ_LOCAL)
        ts_local = dt_local.strftime("%Y-%m-%d %H:%M:%S")

        events.append({
            "id": fid,
            "cam_id": fr["cam_id"],
            "ts_utc": fr["ts_utc"],
            "ts_local": ts_local,
            "vehicles_count": fr["vehicles_count"],
            "raw_url": f"/out/{raw_name}",
            "annotated_url": f"/out/{ann_name}",
            "detections": det_by_frame.get(fid, []),
        })

    return jsonify({
        "since_id": since_id,
        "max_id": max_id,
        "count": len(events),
        "events": events,
    })


@app.get("/api/events/latest")
def api_events_latest():
    limit = request.args.get("limit", default=50, type=int)
    limit = max(1, min(limit, 200))
    cam_id = request.args.get("cam_id")
    date_str = request.args.get("date")  # local date

    con = db_connect()

    where = []
    params = []

    if cam_id:
        where.append("cam_id = ?")
        params.append(cam_id)

    if date_str:
        local_day = parse_local_date(date_str)
        start_utc_str, end_utc_str = local_day_bounds_utc(local_day)
        where.append("ts_utc >= ? AND ts_utc < ?")
        params.extend([start_utc_str, end_utc_str])

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    frames = con.execute(
        f"""
        SELECT id, cam_id, ts_utc, raw_path, annotated_path, vehicles_count
        FROM frames
        {where_sql}
        ORDER BY id DESC
        LIMIT ?
        """,
        (*params, limit)
    ).fetchall()

    frame_ids = [row["id"] for row in frames]
    det_by_frame = {fid: [] for fid in frame_ids}
    if frame_ids:
        q_marks = ",".join(["?"] * len(frame_ids))
        dets = con.execute(
            f"""
            SELECT frame_id, cls, conf, x1, y1, x2, y2
            FROM detections
            WHERE frame_id IN ({q_marks})
            ORDER BY frame_id DESC, conf DESC
            """,
            frame_ids
        ).fetchall()
        for d in dets:
            det_by_frame[d["frame_id"]].append({
                "cls": d["cls"],
                "conf": d["conf"],
                "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
            })

    events = []
    for fr in frames:
        raw_name = safe_out_filename(fr["raw_path"])
        ann_name = safe_out_filename(fr["annotated_path"])
        dt_local = parse_ts_utc(fr["ts_utc"]).astimezone(TZ_LOCAL)
        ts_local = dt_local.strftime("%Y-%m-%d %H:%M:%S")
        events.append({
            "id": fr["id"],
            "cam_id": fr["cam_id"],
            "ts_utc": fr["ts_utc"],
            "ts_local": ts_local,
            "vehicles_count": fr["vehicles_count"],
            "raw_url": f"/out/{raw_name}",
            "annotated_url": f"/out/{ann_name}",
            "detections": det_by_frame.get(fr["id"], []),
        })

    return jsonify({"count": len(events), "events": events})


@app.get("/api/stats")
def api_stats():
    date_str = request.args.get("date")
    bucket = request.args.get("bucket", "10m")
    cam_id = request.args.get("cam_id")

    local_day = parse_local_date(date_str)
    bucket_minutes = 10 if bucket == "10m" else 60

    now_local = datetime.now(TZ_LOCAL)
    is_today = (local_day == now_local.date())

    start_local = datetime.combine(local_day, time(6, 0), tzinfo=TZ_LOCAL)
    end_local_fixed = datetime.combine(local_day, time(22, 0), tzinfo=TZ_LOCAL)

    status = read_status()
    if is_today and status.get("ok") and status.get("window_local"):
        wl = status["window_local"] or {}
        start_hhmm = wl.get("start_effective") or wl.get("start")
        end_hhmm = wl.get("end_effective") or wl.get("end")
        try:
            if start_hhmm:
                start_local = datetime.combine(local_day, hhmm_to_time(start_hhmm), tzinfo=TZ_LOCAL)
            if end_hhmm:
                end_local_fixed = datetime.combine(local_day, hhmm_to_time(end_hhmm), tzinfo=TZ_LOCAL)
        except Exception:
            pass

    end_local = min(now_local, end_local_fixed) if is_today else end_local_fixed

    if end_local <= start_local:
        return jsonify({
            "date_local": local_day.strftime("%Y-%m-%d"),
            "window_local": {"start": start_local.strftime("%H:%M"), "end": end_local.strftime("%H:%M")},
            "bucket_minutes": bucket_minutes,
            "filter": {"cam_id": cam_id},
            "totals": {"events": 0, "vehicles": 0},
            "podium": [],
            "buckets": [],
            "status": status,
        })

    def floor_to_bucket(dt: datetime, minutes: int) -> datetime:
        discard = dt.minute % minutes
        return dt.replace(minute=dt.minute - discard, second=0, microsecond=0)

    end_bucket_start = floor_to_bucket(end_local, bucket_minutes)
    end_local_rounded = end_bucket_start + timedelta(minutes=bucket_minutes)
    end_local_rounded = min(end_local_rounded, end_local_fixed)

    start_utc_str = fmt_ts_utc(start_local.astimezone(timezone.utc))
    end_utc_str = fmt_ts_utc(end_local_rounded.astimezone(timezone.utc))

    con = db_connect()

    if cam_id:
        rows_hist = con.execute(
            """
            SELECT ts_utc, vehicles_count
            FROM frames
            WHERE ts_utc >= ? AND ts_utc < ? AND cam_id = ?
            ORDER BY ts_utc ASC
            """,
            (start_utc_str, end_utc_str, cam_id)
        ).fetchall()
    else:
        rows_hist = con.execute(
            """
            SELECT ts_utc, vehicles_count
            FROM frames
            WHERE ts_utc >= ? AND ts_utc < ?
            ORDER BY ts_utc ASC
            """,
            (start_utc_str, end_utc_str)
        ).fetchall()

    rows_global = con.execute(
        """
        SELECT cam_id, vehicles_count
        FROM frames
        WHERE ts_utc >= ? AND ts_utc < ?
        """,
        (start_utc_str, end_utc_str)
    ).fetchall()

    buckets = []
    cursor = start_local
    while cursor < end_local_rounded:
        buckets.append({"start_local": cursor.strftime("%H:%M"), "count": 0})
        cursor += timedelta(minutes=bucket_minutes)

    def bucket_index(dt_local: datetime) -> int:
        if dt_local < start_local or dt_local >= end_local_rounded:
            return -1
        delta = dt_local - start_local
        minutes = int(delta.total_seconds() // 60)
        return minutes // bucket_minutes

    total_vehicles = 0
    total_events = 0
    for r in rows_hist:
        dt_local = parse_ts_utc(r["ts_utc"]).astimezone(TZ_LOCAL)
        idx = bucket_index(dt_local)
        if 0 <= idx < len(buckets):
            buckets[idx]["count"] += int(r["vehicles_count"])
        total_vehicles += int(r["vehicles_count"])
        total_events += 1

    by_cam_global = {}
    for r in rows_global:
        cid = r["cam_id"]
        by_cam_global[cid] = by_cam_global.get(cid, 0) + int(r["vehicles_count"])

    podium = sorted(
        [{"cam_id": k, "vehicles": v} for k, v in by_cam_global.items()],
        key=lambda x: x["vehicles"],
        reverse=True
    )[:3]

    return jsonify({
        "date_local": local_day.strftime("%Y-%m-%d"),
        "window_local": {"start": start_local.strftime("%H:%M"), "end": end_local.strftime("%H:%M")},
        "bucket_minutes": bucket_minutes,
        "filter": {"cam_id": cam_id},
        "totals": {"events": total_events, "vehicles": total_vehicles},
        "podium": podium,
        "buckets": buckets,
        "status": status,
    })


@app.get("/out/<path:filename>")
def serve_out_file(filename):
    safe_path = (OUT_DIR / filename).resolve()
    if OUT_DIR not in safe_path.parents and safe_path != OUT_DIR:
        abort(404)
    if not safe_path.exists():
        abort(404)
    return send_from_directory(OUT_DIR, filename, conditional=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
