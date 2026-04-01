"""
Convert LaFan1 retargeted CSV files to GMR-format pickle files.

The LaFan1 CSV format (from LAFAN1_Retargeting_Dataset/g1/) has 36 columns per row:
  - Columns 0-2:  root_pos (x, y, z)
  - Columns 3-6:  root_rot quaternion (x, y, z, w) — same as GMR convention
  - Columns 7-35: dof_pos (29 DOFs in GMR/body-grouped order)

Output GMR pickle format:
  - 'fps': Frame rate (int)
  - 'root_pos': ndarray (num_frames, 3)
  - 'root_rot': ndarray (num_frames, 4) in (x, y, z, w) format
  - 'dof_pos': ndarray (num_frames, 29)

Usage:
    python csv_to_gmr_pkl.py \
        --input_dir /path/to/csv_dir/ \
        --output_dir /path/to/output/ \
        --files walk1_subject1.csv run1_subject2.csv fallAndGetUp1_subject1.csv \
        --fps 30
"""

import argparse
import csv
import os
import pickle
from pathlib import Path

import numpy as np


def csv_to_gmr(csv_path: str, fps: int) -> dict:
    """Convert a single LaFan1 CSV file to GMR-format dictionary."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append([float(v) for v in row])

    data = np.array(rows, dtype=np.float64)
    assert data.shape[1] == 36, f"Expected 36 columns, got {data.shape[1]}"

    root_pos = data[:, 0:3]
    root_rot = data[:, 3:7]  # (x, y, z, w) format — matches GMR convention
    dof_pos = data[:, 7:36]

    return {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert LaFan1 CSV to GMR pickle.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with CSV files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for PKL files")
    parser.add_argument("--files", nargs="+", required=True, help="CSV filenames to convert")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate (default: 30)")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for filename in args.files:
        csv_path = os.path.join(args.input_dir, filename)
        if not os.path.exists(csv_path):
            print(f"[WARN] File not found, skipping: {csv_path}")
            continue

        gmr_data = csv_to_gmr(csv_path, args.fps)
        out_name = Path(filename).stem + ".pkl"
        out_path = os.path.join(args.output_dir, out_name)

        with open(out_path, "wb") as f:
            pickle.dump(gmr_data, f)

        print(f"[OK] {filename} -> {out_name}  ({gmr_data['dof_pos'].shape[0]} frames)")

    print("Done.")


if __name__ == "__main__":
    main()
