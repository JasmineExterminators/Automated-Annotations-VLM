#!/usr/bin/env python3
from pathlib import Path
import sys

# ←– EDIT THIS to the path of the folder containing your .txt annotation files:
ANNOTATIONS_FOLDER = Path("C:/Users/wuad3/Downloads/30FPS_ANNOTATIONS")

# 30 fps → 20 fps scaling factor
FACTOR = 1.5

def sec_to_hms(sec: float) -> str:
    """Convert seconds (float) to HH:MM:SS.mmm"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - (h*3600 + m*60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def convert_line(line: str, factor: float = FACTOR) -> str:
    # split on tabs; format is:
    # [0]=label, [1]=empty, [2]=start_hms, [3]=start_sec,
    # [4]=end_hms,   [5]=end_sec,   [6]=dur_hms,   [7]=dur_sec, [8]=annotation
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 6:
        return line  # not enough columns
    try:
        orig_start = float(parts[3])
        orig_end   = float(parts[5])
    except ValueError:
        return line  # non-data line (e.g. header or text)
    new_start = orig_start * factor
    new_end   = orig_end   * factor

    new_dur = new_end - new_start

    # overwrite the four time fields
    parts[2] = sec_to_hms(new_start)
    parts[3] = f"{new_start:.3f}"
    parts[4] = sec_to_hms(new_end)
    parts[5] = f"{new_end:.3f}"
    # overwrite the two duration fields
    parts[6] = sec_to_hms(new_dur)
    parts[7] = f"{new_dur:.3f}"

    return "\t".join(parts) + "\n"

def main():
    src = ANNOTATIONS_FOLDER
    if not src.is_dir():
        print(f"Error: {src!r} is not a directory.", file=sys.stderr)
        sys.exit(1)

    dst = src / "converted_annotations"
    dst.mkdir(exist_ok=True)

    txts = list(src.glob("*.txt"))
    if not txts:
        print(f"No .txt files found in {src!r}.", file=sys.stderr)
        sys.exit(1)

    for f in txts:
        out = dst / f.name
        with f.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
            for line in fin:
                fout.write(convert_line(line))
        print(f"Converted {f.name} → {out}")

if __name__ == "__main__":
    main()
