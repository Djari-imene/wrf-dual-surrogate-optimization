#!/usr/bin/env python3
"""
Download GPM IMERG Daily Precipitation for Batna (35.75°N, 6.17°E)
===================================================================
Saves one CSV file per period with date and precipitation (mm/day).

Requirements:
    pip install earthaccess h5py pandas numpy

NASA Earthdata account (free): https://urs.earthdata.nasa.gov
"""

import os
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from datetime import date, timedelta
import earthaccess

# ── SETTINGS ──────────────────────────────────────────────────────────────────
SITE_LAT  = 35.75   # degrees North
SITE_LON  =  6.17   # degrees East

EARTHDATA_USER = ""   # ← your NASA Earthdata username
EARTHDATA_PASS = ""   # ← your NASA Earthdata password

# Periods to download
PERIODS = {
    "training_May_2025"     : (date(2025, 5, 1),  date(2025, 5, 14)),
    "validation_March_2025" : (date(2025, 3, 2),  date(2025, 3, 11)),
    "validation_April_2025" : (date(2025, 4, 11),  date(2025, 4, 20)),
    "validation_May_2025"   : (date(2025, 5, 16),  date(2025, 5, 25)),
}

# Output folder
OUT_DIR   = Path("imerg_data")
CACHE_DIR = OUT_DIR / "raw_hdf5"

# NASA credentials (set as environment variables or fill in directly)
EARTHDATA_USER = os.environ.get("EARTHDATA_USER", "")
EARTHDATA_PASS = os.environ.get("EARTHDATA_PASS", "")

# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def authenticate():
    if EARTHDATA_USER and EARTHDATA_PASS:
        os.environ["EARTHDATA_USERNAME"] = EARTHDATA_USER
        os.environ["EARTHDATA_PASSWORD"] = EARTHDATA_PASS
        earthaccess.login(strategy="environment")
    else:
        try:
            earthaccess.login(strategy="netrc")
        except Exception:
            earthaccess.login(strategy="interactive")
    print("✓ NASA Earthdata authenticated\n")


def date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def download_one_day(target_date: date) -> Path | None:
    """Download IMERG Daily Final file for one date. Returns local path."""
    fname = CACHE_DIR / f"IMERG_{target_date.strftime('%Y%m%d')}.HDF5"
    if fname.exists():
        return fname   # already cached

    results = earthaccess.search_data(
        short_name   = "GPM_3IMERGDF",
        version      = "07",
        temporal     = (
            target_date.strftime("%Y-%m-%dT00:00:00"),
            target_date.strftime("%Y-%m-%dT23:59:59"),
        ),
    )

    if not results:
        print(f" No data for {target_date}")
        return None

    downloaded = earthaccess.download(results, str(CACHE_DIR))
    if downloaded:
        Path(downloaded[0]).rename(fname)
        return fname
    return None


def extract_precip_at_point(hdf5_path: Path, lat: float, lon: float) -> float:
    """
    Extract precipitation (mm/day) at the nearest IMERG grid point
    to (lat, lon) from one daily HDF5 file.
    """
    with h5py.File(hdf5_path, "r") as f:
        imerg_lons  = f["Grid/lon"][:]          # 1D, 3600 values, 0.1° step
        imerg_lats  = f["Grid/lat"][:]          # 1D, 1800 values, 0.1° step
        precip_3d   = f["Grid/precipitationCal"][:]  # shape: (1, 3600, 1800)

    # Find nearest grid indices
    i_lon = int(np.argmin(np.abs(imerg_lons - lon)))
    i_lat = int(np.argmin(np.abs(imerg_lats - lat)))

    precip_val = float(precip_3d[0, i_lon, i_lat])

    # Replace fill/missing with NaN
    if precip_val < 0:
        precip_val = float("nan")

    return precip_val


def process_period(label: str, start: date, end: date) -> pd.DataFrame:
    """Download and extract precipitation for all days in a period."""
    print(f"── {label}  ({start} → {end})")
    records = []

    for d in date_range(start, end):
        fpath = download_one_day(d)
        if fpath is None:
            records.append({"date": d, "precip_mm_day": float("nan")})
            continue

        precip = extract_precip_at_point(fpath, SITE_LAT, SITE_LON)
        records.append({"date": d, "precip_mm_day": round(precip, 4)})
        print(f"  {d}  →  {precip:.2f} mm/day")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    # Add cumulative and summary
    df["precip_cumul_mm"] = df["precip_mm_day"].cumsum()
    total = df["precip_mm_day"].sum()
    print(f"  Total accumulated: {total:.2f} mm over {len(df)} days\n")

    return df


def main():
    print("=" * 55)
    print("GPM IMERG — Precipitation Extraction")
    print(f"Site : {SITE_LAT}°N  {SITE_LON}°E")
    print("=" * 55 + "\n")

    authenticate()

    all_dfs = []

    for label, (start, end) in PERIODS.items():
        df = process_period(label, start, end)
        df.insert(0, "period", label)

        # Save individual CSV
        csv_path = OUT_DIR / f"precip_{label}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved: {csv_path}")

        all_dfs.append(df)

    # Save combined CSV
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = OUT_DIR / "precip_all_periods.csv"
    combined.to_csv(combined_path, index=False)

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    for label, (start, end) in PERIODS.items():
        sub   = combined[combined["period"] == label]
        total = sub["precip_mm_day"].sum()
        print(f"  {label:<35} {total:7.2f} mm")

    print(f"\n✓ All data saved in:  {OUT_DIR}/")
    print(f"✓ Combined file   :  {combined_path}")


if __name__ == "__main__":
    main()
