import argparse
import csv
import json
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_CSV = SCRIPT_DIR.parent / "data" / "merinorainbow.csv"
DEFAULT_ASIN_CSV = SCRIPT_DIR / "merinorainbow_asin_generated.csv"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "merinorainbow_for_data_replace.csv"


def normalize_digits(text: str) -> str:
    return text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))


def normalize_color_number(text: str) -> str:
    raw = normalize_digits((text or "").strip())
    if not raw:
        return ""
    digits = re.findall(r"\d+", raw)
    if not digits:
        return ""
    value = digits[0]
    return value.zfill(3) if len(value) <= 3 else value


def load_asin_map(asin_csv_path: Path) -> dict[str, str]:
    if not asin_csv_path.exists():
        raise FileNotFoundError(f"ASIN生成CSVが見つかりません: {asin_csv_path}")

    with open(asin_csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("ASIN生成CSVのヘッダーがありません")

        required = {"色番", "ASIN"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"ASIN生成CSVに必要な列がありません: {required}")

        ret: dict[str, str] = {}
        for row in reader:
            color_number = normalize_color_number(row.get("色番", ""))
            if not color_number:
                continue
            ret[color_number] = (row.get("ASIN", "") or "").strip().upper()
        return ret


def build_rows_for_data(base_csv_path: Path, asin_map: dict[str, str]) -> tuple[list[list[str]], dict[str, int]]:
    if not base_csv_path.exists():
        raise FileNotFoundError(f"ベースCSVが見つかりません: {base_csv_path}")

    out_rows: list[list[str]] = [["系統", "色番", "R", "G", "B", "ASIN"]]
    total = 0
    assigned = 0

    with open(base_csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("ベースCSVのヘッダーがありません")

        required = {"系統", "色番", "R", "G", "B"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"ベースCSVに必要な列がありません: {required}")

        for row in reader:
            total += 1
            color_number = normalize_color_number(row.get("色番", ""))
            asin = asin_map.get(color_number, "") if color_number else ""
            if asin:
                assigned += 1

            out_rows.append([
                (row.get("系統", "") or "").strip(),
                color_number,
                (row.get("R", "") or "").strip(),
                (row.get("G", "") or "").strip(),
                (row.get("B", "") or "").strip(),
                asin,
            ])

    stats = {
        "total": total,
        "assigned": assigned,
        "missing": total - assigned,
    }
    return out_rows, stats


def write_csv(rows: list[list[str]], output_csv_path: Path) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "merinorainbow専用: 生成済みASIN CSV（色番,ASIN）を"
            "data置換向けフォーマット（系統,色番,R,G,B,ASIN）へ変換します。"
        )
    )
    parser.add_argument("--base-csv", default=str(DEFAULT_BASE_CSV), help="ベースとなるmerinorainbow.csv")
    parser.add_argument("--asin-csv", default=str(DEFAULT_ASIN_CSV), help="ASIN生成CSV")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="出力CSV")
    parser.add_argument("--dry-run", action="store_true", help="ファイル書き込みなし")
    args = parser.parse_args()

    asin_map = load_asin_map(Path(args.asin_csv))
    rows, stats = build_rows_for_data(Path(args.base_csv), asin_map)

    if not args.dry_run:
        write_csv(rows, Path(args.output_csv))

    result = {
        "base_csv": str(Path(args.base_csv)),
        "asin_csv": str(Path(args.asin_csv)),
        "output_csv": str(Path(args.output_csv)),
        "dry_run": args.dry_run,
        **stats,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
