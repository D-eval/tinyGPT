from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .read import DATA_DIR, META_PATH, TARGET_BYTES, corpus_stats
except ImportError:
    from read import DATA_DIR, META_PATH, TARGET_BYTES, corpus_stats


REPORT_DIR = Path(__file__).resolve().parent / "report"


def format_gb(size_bytes: int) -> str:
    return f"{size_bytes / 1024**3:.4f}G"


def next_report_path(report_dir: Path = REPORT_DIR) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    numbers: list[int] = []
    for path in report_dir.glob("*.txt"):
        try:
            numbers.append(int(path.stem))
        except ValueError:
            continue
    return report_dir / f"{max(numbers, default=0) + 1}.txt"


def write_failure_report(stats: dict[str, int | float | bool], target_bytes: int) -> Path:
    report = next_report_path()
    report.write_text(
        "\n".join(
            [
                "dataset2 vertify.py 校验未通过",
                f"当前文件数: {stats['files']}",
                f"当前文本文件数: {stats['text_files']}",
                f"当前数据大小: {stats['bytes']} bytes ({format_gb(int(stats['bytes']))})",
                f"目标数据大小: {target_bytes} bytes ({format_gb(target_bytes)})",
                f"meta.csv 行数: {stats['meta_rows']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def verify(
    dataset_dir: str | Path = DATA_DIR,
    target_bytes: int = TARGET_BYTES,
    meta_path: str | Path = META_PATH,
) -> dict[str, int | float | bool]:
    stats = corpus_stats(dataset_dir, meta_path)
    stats["target_bytes"] = target_bytes
    stats["target_gb"] = round(target_bytes / 1024**3, 4)
    stats["target_reached"] = int(stats["bytes"]) >= target_bytes
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify dataset2/data size.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="dataset2 data directory")
    parser.add_argument("--meta", type=Path, default=META_PATH, help="metadata CSV path")
    parser.add_argument(
        "--target-mb",
        type=float,
        default=TARGET_BYTES / 1024**2,
        help="required corpus size",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--write-report", action="store_true", help="write dataset2/report/N.txt on failure")
    args = parser.parse_args()

    target_bytes = int(args.target_mb * 1024**2)
    stats = verify(args.data_dir, target_bytes, args.meta)
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        print(
            "dataset2/data: "
            f"files={stats['files']} "
            f"size={format_gb(int(stats['bytes']))} "
            f"target={format_gb(target_bytes)} "
            f"ok={stats['target_reached']}"
        )

    if not stats["target_reached"]:
        if args.write_report:
            report = write_failure_report(stats, target_bytes)
            print(f"failure_report={report}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
