#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU

import warnings
warnings.filterwarnings("ignore", message="CUDA initialization")

import csv
import json
import time
import hashlib
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone, date as date_type
from zoneinfo import ZoneInfo

import requests
from PIL import Image, ImageFilter
from ultralytics import YOLO


# -------------------
# Config
# -------------------
BASE_DIR = Path(__file__).resolve().parent

# Allow override by env, but keep sensible defaults
CSV_PATH = os.environ.get("INFOROUTE_CSV_PATH", str(BASE_DIR / "webcams.csv"))

OUT_DIR = Path(os.environ.get("INFOROUTE_OUT_DIR", "/var/lib/visiontrafic"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_HASH_PATH   = OUT_DIR / "state_last_hash.json"
STATE_TRACKS_PATH = OUT_DIR / "state_tracks.json"
STATUS_PATH       = OUT_DIR / "status.json"          # consumed by API/front

DB_PATH = Path(os.environ.get("INFOROUTE_DB_PATH", str(OUT_DIR / "events.sqlite3")))

CYCLE_SECONDS = 100
MAX_DOWNLOAD_WORKERS = 8
REQUEST_TIMEOUT_S = 10

# YOLO COCO vehicle classes: 2=car, 3=motorcycle, 5=bus, 7=truck
COCO_VEHICLE_CLASS_IDS = [2, 3, 5, 7]
COCO_ID_TO_NAME = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Matching / parking logic
IOU_MATCH_THRESHOLD = 0.65
PARKED_SEEN_COUNT = 2
TRACK_TTL_SECONDS = 30 * 60
MAX_TRACKS_PER_CAM = 30

# Night pause logic (civil twilight)
TZ_LOCAL = ZoneInfo("Europe/Paris")
CIVIL_MARGIN_SECONDS = 180  # +3 min after dawn, +3 min after dusk (safety)
FALLBACK_DAY_START = (6, 0)  # if astral missing
FALLBACK_DAY_END   = (22, 0)

USER_AGENT = "Mozilla/5.0 (compatible; Inforoute43Bot/1.0)"

# Privacy/minimization (post-processing for web display)
WEB_MAX_SIZE = 450          # max width/height (aspect ratio preserved)
RAW_BLUR_RADIUS = 0.7       # light blur on raw
ANN_BLUR_RADIUS = 0.5       # slightly less blur so boxes remain readable
RAW_JPEG_QUALITY = 60
ANN_JPEG_QUALITY = 62


# -------------------
# Utils
# -------------------
def now_utc_dt() -> datetime:
    return datetime.now(timezone.utc)

def now_local_dt() -> datetime:
    return datetime.now(TZ_LOCAL)

def now_ts_id():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def load_webcams(csv_path: str):
    cams = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cams.append(row)
    return cams

def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default

def save_json_atomic(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def downscale_and_blur_inplace_jpeg(
    path: Path,
    max_size: int = WEB_MAX_SIZE,
    blur_radius: float = 0.6,
    quality: int = 60
):
    """
    Downscale (keep aspect ratio) + light blur + JPEG recompress, IN-PLACE.

    IMPORTANT:
    - Called AFTER YOLO + annotation + DB insert
    - Keeps file naming / paths unchanged => no need to modify api.py
    """
    try:
        img = Image.open(path).convert("RGB")

        # Keep aspect ratio, shrink to fit in max_size x max_size
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        if blur_radius and blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # atomic replace
        tmp = path.with_suffix(path.suffix + ".tmp")
        img.save(tmp, format="JPEG", quality=quality, optimize=True, progressive=True)
        tmp.replace(path)
    except Exception as e:
        print(f"  WARN: downscale/blur failed for {path.name}: {e}")


# -------------------
# Daylight (civil twilight)
# -------------------
def mean_lat_lon(cams: list[dict]) -> tuple[float, float]:
    lats, lons = [], []
    for c in cams:
        try:
            lats.append(float(c["latitude"]))
            lons.append(float(c["longitude"]))
        except Exception:
            pass
    if not lats or not lons:
        # fallback roughly Haute-Loire
        return (45.05, 4.05)
    return (sum(lats) / len(lats), sum(lons) / len(lons))

def get_civil_window_local(day: date_type, lat: float, lon: float) -> tuple[datetime, datetime, str]:
    """
    Returns (dawn_local, dusk_local, source)
    Uses astral if available, else fallback 06:00-22:00.
    """
    try:
        from astral import Observer
        from astral.sun import sun

        obs = Observer(latitude=lat, longitude=lon)
        s = sun(observer=obs, date=day, tzinfo=TZ_LOCAL)  # astral uses depression=6° by default => civil
        # s['dawn'] / s['dusk'] correspond to civil twilight
        dawn = s["dawn"]
        dusk = s["dusk"]
        return dawn, dusk, "astral(civil)"
    except Exception:
        dawn = datetime(day.year, day.month, day.day, FALLBACK_DAY_START[0], FALLBACK_DAY_START[1], tzinfo=TZ_LOCAL)
        dusk = datetime(day.year, day.month, day.day, FALLBACK_DAY_END[0], FALLBACK_DAY_END[1], tzinfo=TZ_LOCAL)
        return dawn, dusk, "fallback(06-22)"

def compute_run_state(lat: float, lon: float):
    """
    Determines if we should run now. If not, returns next wake time.
    """
    now_l = now_local_dt()
    today = now_l.date()

    dawn, dusk, src = get_civil_window_local(today, lat, lon)

    # safety margins
    dawn_m = dawn + timedelta(seconds=CIVIL_MARGIN_SECONDS)
    dusk_m = dusk - timedelta(seconds=CIVIL_MARGIN_SECONDS)

    if dawn_m <= now_l < dusk_m:
        return {
            "can_run": True,
            "reason": "daylight",
            "window_local": {
                "date": today.isoformat(),
                "start": dawn.strftime("%H:%M"),
                "end": dusk.strftime("%H:%M"),
                "source": src,
                "start_effective": dawn_m.strftime("%H:%M"),
                "end_effective": dusk_m.strftime("%H:%M"),
            },
            "next_wake_local": None,
        }

    # compute next wake: if before dawn => today dawn_m, else tomorrow dawn_m
    if now_l < dawn_m:
        next_wake = dawn_m
        reason = "night_before_dawn"
    else:
        tomorrow = today + timedelta(days=1)
        dawn2, _, src2 = get_civil_window_local(tomorrow, lat, lon)
        next_wake = dawn2 + timedelta(seconds=CIVIL_MARGIN_SECONDS)
        reason = "night_after_dusk"

        # keep source for window on "today" but mention tomorrow source too
        src = src2

    return {
        "can_run": False,
        "reason": reason,
        "window_local": {
            "date": today.isoformat(),
            "start": dawn.strftime("%H:%M"),
            "end": dusk.strftime("%H:%M"),
            "source": src,
            "start_effective": dawn_m.strftime("%H:%M"),
            "end_effective": dusk_m.strftime("%H:%M"),
        },
        "next_wake_local": next_wake,
    }

def write_status(paused: bool, reason: str, lat: float, lon: float, window_local: dict, next_wake_local: datetime | None):
    nowu = now_utc_dt()
    payload = {
        "ok": True,
        "service": "run_yolo_cycle",
        "updated_at_utc": nowu.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "paused": paused,
        "reason": reason,
        "lat_ref": lat,
        "lon_ref": lon,
        "window_local": window_local,
        "next_wake_local": next_wake_local.strftime("%Y-%m-%d %H:%M:%S") if next_wake_local else None,
        "next_wake_utc": next_wake_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if next_wake_local else None,
        "cycle_seconds": CYCLE_SECONDS,
    }
    save_json_atomic(STATUS_PATH, payload)


# -------------------
# DB
# -------------------
def db_connect():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def db_init(con: sqlite3.Connection):
    con.execute("""
    CREATE TABLE IF NOT EXISTS frames (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cam_id TEXT NOT NULL,
        ts_utc TEXT NOT NULL,
        raw_path TEXT NOT NULL,
        annotated_path TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        vehicles_count INTEGER NOT NULL,
        created_at_utc TEXT NOT NULL
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        frame_id INTEGER NOT NULL,
        cls TEXT NOT NULL,
        conf REAL NOT NULL,
        x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
        FOREIGN KEY(frame_id) REFERENCES frames(id)
    );
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_frames_ts ON frames(ts_utc);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_frames_cam ON frames(cam_id);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_det_frame ON detections(frame_id);")
    con.commit()

def db_insert_frame_and_detections(
    con: sqlite3.Connection,
    cam_id: str,
    ts_utc: str,
    raw_path: Path,
    ann_path: Path,
    sha256: str,
    vehicles
) -> int:
    created_at = now_ts_id()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO frames(cam_id, ts_utc, raw_path, annotated_path, sha256, vehicles_count, created_at_utc)
        VALUES (?,?,?,?,?,?,?)
        """,
        (str(cam_id), ts_utc, str(raw_path), str(ann_path), sha256, len(vehicles), created_at),
    )
    frame_id = cur.lastrowid

    cur.executemany(
        """
        INSERT INTO detections(frame_id, cls, conf, x1, y1, x2, y2)
        VALUES (?,?,?,?,?,?,?)
        """,
        [(frame_id, cls, conf, x1, y1, x2, y2) for (cls, conf, x1, y1, x2, y2) in vehicles],
    )

    con.commit()
    return frame_id


# -------------------
# Download
# -------------------
def download_one(cam: dict):
    cam_id = cam["id_webcam"].strip()
    url = cam["url_image"].strip()

    sep = "&" if "?" in url else "?"
    url_nc = f"{url}{sep}t={int(time.time())}"

    headers = {"User-Agent": USER_AGENT}
    try:
        t0 = time.time()
        r = requests.get(url_nc, timeout=REQUEST_TIMEOUT_S, headers=headers)
        r.raise_for_status()
        return {"cam_id": cam_id, "ok": True, "bytes": r.content, "dt": time.time() - t0}
    except Exception as e:
        return {"cam_id": cam_id, "ok": False, "err": str(e)}


# -------------------
# Tracks (parking filter)
# -------------------
@dataclass
class Track:
    x1: float
    y1: float
    x2: float
    y2: float
    seen_count: int
    parked: bool
    last_seen_epoch: float

def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union

def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))

def to_rel_bbox(x1, y1, x2, y2, w, h):
    return (
        clamp01(x1 / w),
        clamp01(y1 / h),
        clamp01(x2 / w),
        clamp01(y2 / h),
    )

def ema_bbox(old_bbox, new_bbox, alpha=0.4):
    ox1, oy1, ox2, oy2 = old_bbox
    nx1, ny1, nx2, ny2 = new_bbox
    return (
        ox1 * (1 - alpha) + nx1 * alpha,
        oy1 * (1 - alpha) + ny1 * alpha,
        ox2 * (1 - alpha) + nx2 * alpha,
        oy2 * (1 - alpha) + ny2 * alpha,
    )

def load_tracks_state():
    raw = load_json(STATE_TRACKS_PATH, default={})
    state = {}
    now = time.time()
    for cam_id, tlist in raw.items():
        tracks = []
        for td in tlist:
            try:
                tr = Track(
                    x1=float(td["x1"]), y1=float(td["y1"]), x2=float(td["x2"]), y2=float(td["y2"]),
                    seen_count=int(td.get("seen_count", 1)),
                    parked=bool(td.get("parked", False)),
                    last_seen_epoch=float(td.get("last_seen_epoch", now)),
                )
                if now - tr.last_seen_epoch <= TRACK_TTL_SECONDS:
                    tracks.append(tr)
            except Exception:
                continue
        state[cam_id] = tracks
    return state

def save_tracks_state(state):
    raw = {cam_id: [asdict(t) for t in tracks] for cam_id, tracks in state.items()}
    save_json_atomic(STATE_TRACKS_PATH, raw)

def update_tracks_and_get_new_events(cam_id: str, tracks_by_cam: dict, detections_abs: list, img_w: int, img_h: int):
    now = time.time()
    tracks = tracks_by_cam.get(cam_id, [])
    tracks = [t for t in tracks if (now - t.last_seen_epoch) <= TRACK_TTL_SECONDS]

    new_events = []

    for (cls, conf, x1, y1, x2, y2) in detections_abs:
        rel_bbox = to_rel_bbox(x1, y1, x2, y2, img_w, img_h)

        best_iou = 0.0
        best_idx = -1
        for i, tr in enumerate(tracks):
            iou = bbox_iou((tr.x1, tr.y1, tr.x2, tr.y2), rel_bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= IOU_MATCH_THRESHOLD and best_idx >= 0:
            tr = tracks[best_idx]
            nb = ema_bbox((tr.x1, tr.y1, tr.x2, tr.y2), rel_bbox, alpha=0.4)
            tr.x1, tr.y1, tr.x2, tr.y2 = nb
            tr.last_seen_epoch = now
            tr.seen_count += 1
            if tr.seen_count >= PARKED_SEEN_COUNT:
                tr.parked = True
            tracks[best_idx] = tr
            continue

        new_events.append((cls, conf, x1, y1, x2, y2))
        tracks.append(Track(
            x1=rel_bbox[0], y1=rel_bbox[1], x2=rel_bbox[2], y2=rel_bbox[3],
            seen_count=1,
            parked=False,
            last_seen_epoch=now
        ))

    if len(tracks) > MAX_TRACKS_PER_CAM:
        tracks.sort(key=lambda t: t.last_seen_epoch, reverse=True)
        tracks = tracks[:MAX_TRACKS_PER_CAM]

    tracks_by_cam[cam_id] = tracks
    return new_events


# -------------------
# YOLO
# -------------------
def run_yolo_vehicle_only(model: YOLO, img_path: Path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    results = model.predict(
        source=img,
        imgsz=640,
        conf=0.25,
        verbose=False,
        device="cpu",
        classes=COCO_VEHICLE_CLASS_IDS,
    )
    r = results[0]
    boxes = r.boxes

    detections = []
    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            _cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            cls_name = "vehicle"
            detections.append((cls_name, conf, x1, y1, x2, y2))

    return detections, (w, h)

def draw_boxes_only(img_path: Path, detections_abs: list, out_path: Path):
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"cv2.imread failed for {img_path}")
    for (cls, conf, x1, y1, x2, y2) in detections_abs:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        label = f"vehicle {conf:.2f}"
        cv2.putText(img, label, (p1[0], max(15, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(out_path), img)


# -------------------
# Main
# -------------------
def main():
    cams = load_webcams(CSV_PATH)
    if not cams:
        raise SystemExit("No webcam rows found in webcams.csv")

    lat_ref, lon_ref = mean_lat_lon(cams)

    model = YOLO("yolov8n.pt")

    hash_state = load_json(STATE_HASH_PATH, default={})
    tracks_state = load_tracks_state()

    con = db_connect()
    db_init(con)

    print(f"Loaded {len(cams)} webcams. Starting loop with CYCLE_SECONDS={CYCLE_SECONDS}s")
    print(f"DB: {DB_PATH}")
    print(f"Status: {STATUS_PATH}")
    print(f"Daylight ref (mean): lat={lat_ref:.5f} lon={lon_ref:.5f} tz={TZ_LOCAL}")

    cycle = 0
    while True:
        # ---- Night pause gate ----
        run_state = compute_run_state(lat_ref, lon_ref)
        if not run_state["can_run"]:
            next_wake = run_state["next_wake_local"]
            reason = run_state["reason"]
            win = run_state["window_local"]

            write_status(
                paused=True,
                reason=reason,
                lat=lat_ref,
                lon=lon_ref,
                window_local=win,
                next_wake_local=next_wake,
            )

            # sleep until next_wake (but be safe)
            now_l = now_local_dt()
            sleep_s = max(30.0, (next_wake - now_l).total_seconds()) if next_wake else 60.0
            print(
                f"[PAUSE NIGHT] reason={reason} now={now_l.strftime('%Y-%m-%d %H:%M:%S')} "
                f"next_wake={next_wake.strftime('%Y-%m-%d %H:%M:%S')} sleep={sleep_s:.0f}s "
                f"twilight={win.get('start')}→{win.get('end')} src={win.get('source')}"
            )
            time.sleep(sleep_s)
            continue

        # running
        write_status(
            paused=False,
            reason="daylight",
            lat=lat_ref,
            lon=lon_ref,
            window_local=run_state["window_local"],
            next_wake_local=None,
        )

        cycle += 1
        cycle_start = time.time()
        ts = now_ts_id()

        print(f"\n===== CYCLE {cycle} @ {ts} =====")
        print(f"Step A: download {len(cams)} images (parallel, workers={MAX_DOWNLOAD_WORKERS})")

        downloads = {}
        ok_count = 0
        with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as ex:
            futures = {ex.submit(download_one, cam): cam for cam in cams}
            for fut in as_completed(futures):
                res = fut.result()
                cam_id = res["cam_id"]
                downloads[cam_id] = res
                if res["ok"]:
                    ok_count += 1
                    print(f"  cam{cam_id} download OK {len(res['bytes'])} bytes ({res['dt']:.2f}s)")
                else:
                    print(f"  cam{cam_id} download ERROR: {res['err']}")

        print(f"Downloads done: ok={ok_count}/{len(cams)}")

        print("Step B: check freshness (sha256) + save raw images")
        candidates = []
        raw_paths = {}
        hashes = {}

        for cam in cams:
            cam_id = cam["id_webcam"].strip()
            res = downloads.get(cam_id)
            if not res or not res.get("ok"):
                continue

            b = res["bytes"]
            h = sha256_bytes(b)
            if hash_state.get(cam_id) == h:
                print(f"  cam{cam_id} unchanged -> skip")
                continue

            raw_path = OUT_DIR / f"cam{cam_id}_{ts}_raw.jpg"
            raw_path.write_bytes(b)

            candidates.append(cam_id)
            raw_paths[cam_id] = raw_path
            hashes[cam_id] = h
            hash_state[cam_id] = h
            print(f"  cam{cam_id} NEW -> saved {raw_path.name}")

        save_json_atomic(STATE_HASH_PATH, hash_state)
        print(f"Fresh images: {len(candidates)} (YOLO+parking filter will run only on these)")

        print("Step C: YOLO vehicles-only + parking filter + DB only if NEW vehicles exist")
        yolo_count = 0
        saved_events = 0
        vehicles_total = 0

        for cam_id in candidates:
            img_path = raw_paths[cam_id]

            t0 = time.time()
            try:
                detections_all, (w, h) = run_yolo_vehicle_only(model, img_path)
            except Exception as e:
                print(f"  cam{cam_id} YOLO ERROR: {e}")
                continue
            yolo_count += 1
            dt = time.time() - t0

            if not detections_all:
                try:
                    img_path.unlink(missing_ok=True)
                except Exception:
                    pass
                print(f"  cam{cam_id} YOLO OK total=0 in {dt:.2f}s -> discard")
                continue

            new_events = update_tracks_and_get_new_events(
                cam_id=cam_id,
                tracks_by_cam=tracks_state,
                detections_abs=detections_all,
                img_w=w,
                img_h=h
            )

            if not new_events:
                try:
                    img_path.unlink(missing_ok=True)
                except Exception:
                    pass
                print(
                    f"  cam{cam_id} YOLO OK total={len(detections_all)} new=0 in {dt:.2f}s "
                    "-> discard (parked/already seen)"
                )
                continue

            ann_path = OUT_DIR / f"cam{cam_id}_{ts}_annotated.jpg"
            try:
                draw_boxes_only(img_path, new_events, ann_path)
            except Exception as e:
                print(f"  cam{cam_id} annotate ERROR: {e}")
                try:
                    img_path.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

            try:
                frame_id = db_insert_frame_and_detections(
                    con=con,
                    cam_id=cam_id,
                    ts_utc=ts,
                    raw_path=img_path,
                    ann_path=ann_path,
                    sha256=hashes[cam_id],
                    vehicles=new_events
                )
            except Exception as e:
                print(f"  cam{cam_id} DB ERROR: {e}")
                try:
                    ann_path.unlink(missing_ok=True)
                    img_path.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

            # --- Privacy/minimization: degrade images for web display (IN-PLACE, naming unchanged) ---
            downscale_and_blur_inplace_jpeg(
                img_path,
                max_size=WEB_MAX_SIZE,
                blur_radius=RAW_BLUR_RADIUS,
                quality=RAW_JPEG_QUALITY
            )
            downscale_and_blur_inplace_jpeg(
                ann_path,
                max_size=WEB_MAX_SIZE,
                blur_radius=ANN_BLUR_RADIUS,
                quality=ANN_JPEG_QUALITY
            )

            saved_events += 1
            vehicles_total += len(new_events)
            print(
                f"  cam{cam_id} YOLO OK total={len(detections_all)} new={len(new_events)} in {dt:.2f}s "
                f"-> SAVED frame_id={frame_id} (downscaled+blurred)"
            )

        save_tracks_state(tracks_state)

        elapsed = time.time() - cycle_start
        print(
            f"Step D: summary: fresh={len(candidates)} yolo={yolo_count} "
            f"saved_events={saved_events} new_vehicles_total={vehicles_total} elapsed={elapsed:.1f}s"
        )

        sleep_s = CYCLE_SECONDS - elapsed
        if sleep_s < 0:
            print(f"WARNING: cycle took {elapsed:.1f}s > {CYCLE_SECONDS}s, skipping sleep")
            sleep_s = 0

        print(f"Sleeping {sleep_s:.1f}s (fixed cadence)\n")
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
