#!/usr/bin/env python
"""Prepare Florida Gulf 2025 AIS data using the historical version-5 rules.

Only FG is supported; DMA's upstream preprocessing is a separate procedure.
Provide extracted daily CSVs for 2025-01-01 through 2025-04-30. No data or
maps are downloaded. Outputs are private data files, not repository assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_NAME = "florida_gulf_2025"
SOURCE_COLUMNS = ("mmsi", "base_date_time", "longitude", "latitude", "sog", "cog", "status")
LAT, LON, SOG, COG, STATUS, TIMESTAMP = range(6)
EARTH_RADIUS_NM = 3440.065


@dataclass(frozen=True)
class Rules:
    lat_min: float = 26.0
    lat_max: float = 28.5
    lon_min: float = -85.0
    lon_max: float = -82.0
    speed_max_knots: float = 30.0
    max_gap_seconds: int = 7200
    raw_min_points: int = 20
    min_duration_seconds: int = 14400
    sample_interval_seconds: int = 600
    sampled_min_points: int = 25
    sampled_max_points: int = 144
    anchor_moored_ratio: float = 0.70
    status_min_coverage: float = 0.50
    stationary_max_sog_knots: float = 1.0
    low_speed_knots: float = 2.0
    low_speed_ratio: float = 0.80


RULES = Rules()


def split_for_day(day: date) -> str:
    if date(2025, 1, 1) <= day <= date(2025, 3, 31):
        return "train"
    if date(2025, 4, 1) <= day <= date(2025, 4, 15):
        return "valid"
    if date(2025, 4, 16) <= day <= date(2025, 4, 30):
        return "test"
    raise ValueError(f"Date outside the configured dataset period: {day}")


def discover_sources(input_dir: Path) -> list[tuple[date, Path]]:
    start, end = date(2025, 1, 1), date(2025, 4, 30)
    expected = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    found = {}
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        match = re.fullmatch(r"ais-(\d{4}-\d{2}-\d{2})(?:\.csv)?", path.name)
        conventional = re.fullmatch(r"AIS_(\d{4})_(\d{2})_(\d{2})\.csv", path.name)
        if match:
            day = date.fromisoformat(match.group(1))
        elif conventional:
            day = date(*(int(part) for part in conventional.groups()))
        else:
            continue
        if not start <= day <= end:
            continue
        if day in found:
            raise ValueError(f"Multiple daily files for {day}: {found[day]} and {path}")
        found[day] = path
    missing = [str(day) for day in expected if day not in found]
    if missing:
        raise ValueError(f"Missing daily files for FG: {', '.join(missing)}")
    return [(day, found[day]) for day in expected]


def read_chunks(path: Path, chunk_size: int):
    aliases = {"basedatetime": "base_date_time", "lon": "longitude", "lat": "latitude"}
    header = pd.read_csv(path, nrows=0).columns
    renamed = {}
    for column in header:
        key = column.lower().replace("_", "").strip()
        canonical = aliases.get(key, key)
        if canonical in SOURCE_COLUMNS:
            if canonical in renamed.values():
                raise ValueError(f"Ambiguous columns in {path}: {canonical}")
            renamed[column] = canonical
    missing = set(SOURCE_COLUMNS) - set(renamed.values())
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    for chunk in pd.read_csv(path, usecols=list(renamed), chunksize=chunk_size,
                             on_bad_lines="skip", low_memory=False):
        yield chunk.rename(columns=renamed)


def clean_chunk(chunk: pd.DataFrame, day: date, coastline_path=None) -> tuple[pd.DataFrame, Counter]:
    counts: Counter = Counter(raw_rows=len(chunk))
    data = pd.DataFrame(index=chunk.index)
    data["mmsi"] = pd.to_numeric(chunk["mmsi"], errors="coerce")
    data["timestamp_parsed"] = pd.to_datetime(
        chunk["base_date_time"], format="%Y-%m-%d %H:%M:%S", errors="coerce", utc=True)
    data["lon"] = pd.to_numeric(chunk["longitude"], errors="coerce")
    data["lat"] = pd.to_numeric(chunk["latitude"], errors="coerce")
    data["sog"] = pd.to_numeric(chunk["sog"], errors="coerce")
    data["cog"] = pd.to_numeric(chunk["cog"], errors="coerce")
    data["status"] = pd.to_numeric(chunk["status"], errors="coerce")
    required = data[["mmsi", "timestamp_parsed", "lon", "lat"]].notna().all(axis=1)
    required &= np.isfinite(data[["mmsi", "lon", "lat"]]).all(axis=1)
    required &= data["mmsi"].between(100_000_000, 999_999_999, inclusive="both")
    required &= np.floor(data["mmsi"]) == data["mmsi"]
    counts["invalid_required"] += int((~required).sum())
    data = data.loc[required].copy()
    if data.empty:
        counts["kept_rows"] += 0
        return data, counts
    data["timestamp"] = data["timestamp_parsed"].astype("int64") // 1_000_000_000
    start = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())
    same_day = data["timestamp"].between(start, start + 24 * 3600 - 1, inclusive="both")
    counts["timestamp_file_mismatch"] += int((~same_day).sum())
    data = data.loc[same_day]
    in_roi = (data["lat"].between(RULES.lat_min, RULES.lat_max, inclusive="both")
              & data["lon"].between(RULES.lon_min, RULES.lon_max, inclusive="both"))
    counts["outside_roi"] += int((~in_roi).sum())
    data = data.loc[in_roi]
    if data.empty:
        counts["kept_rows"] += 0
        return data, counts
    valid_sog = data["sog"].notna() & np.isfinite(data["sog"])
    valid_sog &= data["sog"].between(0.0, RULES.speed_max_knots, inclusive="both")
    counts["invalid_sog"] += int((~valid_sog).sum())
    data = data.loc[valid_sog]
    valid_cog = data["cog"].notna() & np.isfinite(data["cog"])
    valid_cog &= (data["cog"] >= 0.0) & (data["cog"] < 360.0)
    counts["invalid_cog"] += int((~valid_cog).sum())
    data = data.loc[valid_cog].copy()
    if data.empty:
        counts["kept_rows"] += 0
        return data, counts
    valid_status = data["status"].between(0, 15, inclusive="both")
    data.loc[~valid_status, "status"] = -1
    result = data[["mmsi", "timestamp", "lat", "lon", "sog", "cog", "status"]].copy()
    for column, dtype in {"mmsi": np.int64, "timestamp": np.int64, "lat": np.float64,
                          "lon": np.float64, "sog": np.float32, "cog": np.float32,
                          "status": np.int16}.items():
        result[column] = result[column].astype(dtype)
    before = len(result)
    result.drop_duplicates(inplace=True)
    counts["exact_duplicates_in_chunk"] += before - len(result)
    counts["kept_rows"] += len(result)
    return result, counts


def haversine_nm(lat1, lon1, lat2, lon2):
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_NM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def empirical_speeds_knots(track: np.ndarray, offset: int = 1) -> np.ndarray:
    if len(track) <= offset:
        return np.empty(0, dtype=np.float64)
    dt_hours = (track[offset:, TIMESTAMP] - track[:-offset, TIMESTAMP]) / 3600.0
    distance = haversine_nm(track[:-offset, LAT], track[:-offset, LON],
                            track[offset:, LAT], track[offset:, LON])
    with np.errstate(divide="ignore", invalid="ignore"):
        return distance / dt_hours


def split_on_gap(track: np.ndarray, gap_seconds: int) -> list[np.ndarray]:
    if len(track) == 0:
        return []
    cut_indices = np.flatnonzero(np.diff(track[:, TIMESTAMP]) >= gap_seconds) + 1
    return [part for part in np.split(track, cut_indices) if len(part)]


def remove_position_outliers(track: np.ndarray) -> tuple[np.ndarray, int]:
    current = track
    removed_total = 0
    for _ in range(32):
        n = len(current)
        if n < 3:
            break
        adjacency = [set() for _ in range(n)]
        edge_count = 0
        for offset in range(1, min(4, n - 1) + 1):
            speeds = empirical_speeds_knots(current, offset)
            bad = np.flatnonzero(speeds > RULES.speed_max_knots + 1e-9)
            for left in bad.tolist():
                right = left + offset
                adjacency[left].add(right)
                adjacency[right].add(left)
                edge_count += 1
        if edge_count == 0:
            break
        degree = np.fromiter((len(neighbors) for neighbors in adjacency), dtype=np.int32, count=n)
        removed = np.zeros(n, dtype=bool)
        while degree.max(initial=0) > 0:
            index = int(np.argmax(degree))
            removed[index] = True
            for neighbor in tuple(adjacency[index]):
                if not removed[neighbor]:
                    adjacency[neighbor].discard(index)
                    degree[neighbor] -= 1
            adjacency[index].clear()
            degree[index] = 0
        count = int(removed.sum())
        if count == 0:
            break
        current = current[~removed]
        removed_total += count
    return current, removed_total


def spherical_interpolate(lat0, lon0, lat1, lon1, fraction):
    lat0r, lon0r = np.radians(lat0), np.radians(lon0)
    lat1r, lon1r = np.radians(lat1), np.radians(lon1)
    v0 = np.column_stack((np.cos(lat0r) * np.cos(lon0r), np.cos(lat0r) * np.sin(lon0r), np.sin(lat0r)))
    v1 = np.column_stack((np.cos(lat1r) * np.cos(lon1r), np.cos(lat1r) * np.sin(lon1r), np.sin(lat1r)))
    omega = np.arccos(np.clip(np.sum(v0 * v1, axis=1), -1.0, 1.0))
    sin_omega = np.sin(omega)
    near = np.abs(sin_omega) < 1e-12
    a = np.empty_like(fraction, dtype=np.float64)
    b = np.empty_like(fraction, dtype=np.float64)
    a[~near] = np.sin((1.0 - fraction[~near]) * omega[~near]) / sin_omega[~near]
    b[~near] = np.sin(fraction[~near] * omega[~near]) / sin_omega[~near]
    a[near] = 1.0 - fraction[near]
    b[near] = fraction[near]
    vectors = a[:, None] * v0 + b[:, None] * v1
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    lat = np.degrees(np.arctan2(vectors[:, 2], np.hypot(vectors[:, 0], vectors[:, 1])))
    lon = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    return lat, lon


def resample_track(track: np.ndarray) -> np.ndarray:
    start, end = int(track[0, TIMESTAMP]), int(track[-1, TIMESTAMP])
    targets = np.arange(start, end + 1, RULES.sample_interval_seconds, dtype=np.int64)
    if len(targets) < RULES.sampled_min_points:
        return np.empty((0, 6), dtype=np.float64)
    source_times = track[:, TIMESTAMP].astype(np.int64)
    left = np.searchsorted(source_times, targets, side="right") - 1
    left = np.clip(left, 0, len(track) - 1)
    exact = source_times[left] == targets
    right = np.minimum(left + 1, len(track) - 1)
    right[exact] = left[exact]
    full_dt = source_times[right] - source_times[left]
    fraction = np.zeros(len(targets), dtype=np.float64)
    interpolated = full_dt > 0
    fraction[interpolated] = (targets[interpolated] - source_times[left[interpolated]]) / full_dt[interpolated]
    lat, lon = spherical_interpolate(track[left, LAT], track[left, LON],
                                      track[right, LAT], track[right, LON], fraction)
    sog = track[left, SOG] + fraction * (track[right, SOG] - track[left, SOG])
    cog_delta = (track[right, COG] - track[left, COG] + 180.0) % 360.0 - 180.0
    cog = (track[left, COG] + fraction * cog_delta) % 360.0
    nearest = np.where(fraction <= 0.5, left, right)
    status = track[nearest, STATUS]
    return np.column_stack((lat, lon, sog, cog, status, targets)).astype(np.float64, copy=False)


def trajectory_rejection_reason(track: np.ndarray) -> str | None:
    if len(track) < RULES.sampled_min_points:
        return "sampled_too_short"
    if float(np.max(track[:, SOG])) < RULES.stationary_max_sog_knots:
        return "stationary_max_sog"
    if float(np.mean(track[:, SOG] < RULES.low_speed_knots)) > RULES.low_speed_ratio:
        return "low_speed_ratio"
    known = track[:, STATUS] >= 0
    if float(np.mean(known)) >= RULES.status_min_coverage:
        stationary_status = np.isin(track[known, STATUS].astype(np.int16), (1, 5))
        if float(np.mean(stationary_status)) > RULES.anchor_moored_ratio:
            return "anchor_moored_status"
    speeds = empirical_speeds_knots(track)
    if len(speeds) and float(np.max(speeds)) > RULES.speed_max_knots + 1e-6:
        return "sampled_empirical_speed"
    return None


def normalize_entry(track: np.ndarray, mmsi: int) -> dict:
    lat_norm = (track[:, LAT] - RULES.lat_min) / (RULES.lat_max - RULES.lat_min)
    lon_norm = (track[:, LON] - RULES.lon_min) / (RULES.lon_max - RULES.lon_min)
    sog_norm = track[:, SOG] / RULES.speed_max_knots
    cog_norm = track[:, COG] / 360.0
    upper = np.nextafter(1.0, 0.0)
    features = np.column_stack((np.clip(lat_norm, 0.0, upper), np.clip(lon_norm, 0.0, upper),
                                np.clip(sog_norm, 0.0, upper), np.clip(cog_norm, 0.0, upper),
                                track[:, TIMESTAMP], np.full(len(track), mmsi, dtype=np.float64)))
    return {"mmsi": int(mmsi), "traj": features.astype(np.float64, copy=False)}


def process_mmsi_track(raw: np.ndarray, mmsi: int, stats: Counter) -> list[dict]:
    order = np.argsort(raw[:, TIMESTAMP], kind="stable")
    raw = raw[order]
    _, unique_index = np.unique(raw[:, TIMESTAMP], return_index=True)
    unique_index.sort()
    stats["duplicate_timestamps"] += len(raw) - len(unique_index)
    raw = raw[unique_index]
    entries: list[dict] = []
    for voyage in split_on_gap(raw, RULES.max_gap_seconds):
        stats["candidate_voyages"] += 1
        if len(voyage) < RULES.raw_min_points:
            stats["raw_too_few_points"] += 1
            continue
        if voyage[-1, TIMESTAMP] - voyage[0, TIMESTAMP] < RULES.min_duration_seconds:
            stats["raw_too_short_duration"] += 1
            continue
        cleaned, removed = remove_position_outliers(voyage)
        stats["position_outlier_points"] += removed
        for clean_part in split_on_gap(cleaned, RULES.max_gap_seconds):
            if len(clean_part) < RULES.raw_min_points:
                stats["post_outlier_too_few_points"] += 1
                continue
            if clean_part[-1, TIMESTAMP] - clean_part[0, TIMESTAMP] < RULES.min_duration_seconds:
                stats["post_outlier_too_short_duration"] += 1
                continue
            sampled = resample_track(clean_part)
            if sampled.size == 0:
                stats["sampled_too_short"] += 1
                continue
            in_roi = ((sampled[:, LAT] >= RULES.lat_min) & (sampled[:, LAT] <= RULES.lat_max)
                      & (sampled[:, LON] >= RULES.lon_min) & (sampled[:, LON] <= RULES.lon_max))
            if not np.all(in_roi):
                stats["sampled_outside_roi_points"] += int((~in_roi).sum())
                sampled = sampled[in_roi]
            for continuous in split_on_gap(sampled, RULES.sample_interval_seconds + 1):
                for start in range(0, len(continuous), RULES.sampled_max_points):
                    piece = continuous[start:start + RULES.sampled_max_points]
                    reason = trajectory_rejection_reason(piece)
                    if reason is not None:
                        stats[reason] += 1
                        continue
                    entries.append(normalize_entry(piece, mmsi))
                    stats["output_tracks"] += 1
                    stats["output_points"] += len(piece)
    return entries


def stage_sources(inventory, work_dir: Path, chunk_size: int, buckets: int) -> Counter:
    totals = Counter()
    for index, (day, path) in enumerate(inventory, start=1):
        parts = []
        for chunk in read_chunks(path, chunk_size):
            filtered, counts = clean_chunk(chunk, day)
            totals.update(counts)
            if not filtered.empty:
                parts.append(filtered)
        if parts:
            daily = pd.concat(parts, ignore_index=True)
            before = len(daily)
            daily.drop_duplicates(inplace=True)
            totals["exact_duplicates_across_chunks"] += before - len(daily)
            daily["bucket"] = (daily["mmsi"] % buckets).astype(np.int32)
            split = split_for_day(day)
            for bucket, part in daily.groupby("bucket", sort=False):
                destination = work_dir / split / f"{int(bucket):04d}"
                destination.mkdir(parents=True, exist_ok=True)
                values = part[["mmsi", "lat", "lon", "sog", "cog", "status", "timestamp"]]
                np.save(destination / f"{day.isoformat()}.npy", values.to_numpy(dtype=np.float64))
        print(f"Filtered day {index}/{len(inventory)}: {day}", flush=True)
    return totals


def prepare_split(split: str, work_dir: Path, buckets: int) -> tuple[list[dict], Counter]:
    output, stats = [], Counter()
    columns = ["mmsi", "lat", "lon", "sog", "cog", "status", "timestamp"]
    for bucket in range(buckets):
        paths = sorted((work_dir / split / f"{bucket:04d}").glob("*.npy"))
        if not paths:
            continue
        data = pd.DataFrame(np.concatenate([np.load(path, allow_pickle=False) for path in paths]),
                            columns=columns)
        data.sort_values(["mmsi", "timestamp"], kind="stable", inplace=True)
        for mmsi, group in data.groupby("mmsi", sort=True):
            raw = group[["lat", "lon", "sog", "cog", "status", "timestamp"]].to_numpy(dtype=np.float64)
            output.extend(process_mmsi_track(raw, int(mmsi), stats))
    return output, stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Extracted daily AIS CSVs, recursively searched; timestamps use UTC YYYY-MM-DD HH:MM:SS.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--buckets", type=int, default=64,
                        help="Keep 64 to preserve the historical trajectory order.")
    args = parser.parse_args(argv)
    if args.chunk_size <= 0 or args.buckets <= 0:
        parser.error("--chunk-size and --buckets must be positive")
    if not args.input_dir.is_dir():
        parser.error("--input-dir must be an existing directory")
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    splits = ("train", "valid", "test")
    destinations = [output_dir / f"{DATASET_NAME}_{split}.pkl" for split in splits]
    manifest_path = output_dir / "dataset_manifest.json"
    if any(path.exists() for path in destinations + [manifest_path]):
        parser.error("Output files already exist; choose a new --output-dir")
    inventory = discover_sources(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    with tempfile.TemporaryDirectory(prefix="prepare-fg-", dir=output_dir) as temporary:
        work_dir = Path(temporary)
        coarse_counts = stage_sources(inventory, work_dir, args.chunk_size, args.buckets)
        for split, destination in zip(splits, destinations):
            entries, counts = prepare_split(split, work_dir, args.buckets)
            staging = destination.with_suffix(".pkl.tmp")
            with staging.open("wb") as handle:
                pickle.dump(entries, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(staging, destination)
            digest = hashlib.sha256()
            with destination.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            reports[split] = {"file": destination.name, "sha256": digest.hexdigest(), **dict(counts)}
            print(f"{split}: {len(entries)} trajectories -> {destination.name}", flush=True)
    manifest = {"dataset_name": DATASET_NAME, "historical_rules_version": 5,
                "rules": asdict(RULES), "land_filter": False, "nearshore_filter": False,
                "buckets": args.buckets, "chunk_size": args.chunk_size,
                "traj_columns": ["lat_norm", "lon_norm", "sog_norm", "cog_norm", "timestamp", "mmsi"],
                "split_periods_utc": {"train": ["2025-01-01", "2025-03-31"],
                                      "valid": ["2025-04-01", "2025-04-15"],
                                      "test": ["2025-04-16", "2025-04-30"]},
                "source_files": [path.relative_to(input_dir).as_posix() for _, path in inventory],
                "coarse_counts": dict(coarse_counts), "splits": reports}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
