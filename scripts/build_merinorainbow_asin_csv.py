import argparse
import csv
import json
import re
import html as html_lib
import importlib
from pathlib import Path
from urllib.parse import unquote
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRODUCT_URL = "https://www.amazon.co.jp/dp/B0B7QVT3BS?th=1"
DEFAULT_COLOR_LIST_PATH = SCRIPT_DIR / "merinorainbow_color_numbers.txt"
DEFAULT_OUTPUT_CSV_PATH = SCRIPT_DIR / "merinorainbow_asin_generated.csv"
DEFAULT_OUTPUT_MAP_JSON_PATH = SCRIPT_DIR / "merinorainbow_color_asin_map.json"

ASIN_PATTERN = re.compile(r"\b(B0[A-Z0-9]{8})\b", re.IGNORECASE)
SCRIPT_TAG_PATTERN = re.compile(r"<script[^>]*>(.*?)</script>", re.IGNORECASE | re.DOTALL)
TAG_PATTERN = re.compile(r"<([a-zA-Z0-9]+)([^>]*)>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
ATTR_PATTERN = re.compile(r'([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*(["\'])(.*?)\2', re.DOTALL)


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


def extract_asin(text: str) -> str:
    match = ASIN_PATTERN.search(text or "")
    return match.group(1).upper() if match else ""


def extract_asin_from_text_candidates(candidates: list[str]) -> str:
    for value in candidates:
        asin = extract_asin(value)
        if asin:
            return asin
        decoded = unquote(value or "")
        asin = extract_asin(decoded)
        if asin:
            return asin
    return ""


def extract_json_object_after_key(text: str, key: str) -> dict:
    key_idx = text.find(key)
    if key_idx < 0:
        return {}
    start = text.find("{", key_idx)
    if start < 0:
        return {}

    depth = 0
    in_string = False
    escaped = False
    end = -1

    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break

    if end < 0:
        return {}

    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}


def extract_color_number_from_label(label: str) -> str:
    return normalize_color_number(html_lib.unescape(label or ""))


def parse_attrs(attrs_text: str) -> dict[str, str]:
    attrs = {}
    for key, _, value in ATTR_PATTERN.findall(attrs_text or ""):
        attrs[key.lower()] = html_lib.unescape(value)
    return attrs


def merge_pairs_to_map(pairs: list[tuple[str, str]]) -> dict[str, str]:
    ret: dict[str, str] = {}
    for color_number, asin in pairs:
        if color_number and asin and color_number not in ret:
            ret[color_number] = asin
    return ret


def extract_pairs_from_tags(html: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for _, attrs_text, inner_text in TAG_PATTERN.findall(html):
        attrs = parse_attrs(attrs_text)
        asin = extract_asin_from_text_candidates(list(attrs.values()) + [inner_text])
        if not asin:
            continue

        label_candidates = [
            attrs.get("title", ""),
            attrs.get("aria-label", ""),
            attrs.get("alt", ""),
            attrs.get("data-value", ""),
            attrs.get("value", ""),
            inner_text,
        ]
        color_number = ""
        for label in label_candidates:
            color_number = extract_color_number_from_label(label)
            if color_number:
                break

        if color_number:
            pairs.append((color_number, asin))
    return pairs


def extract_pairs_from_scripts(html: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for script_body in SCRIPT_TAG_PATTERN.findall(html):
        obj = extract_json_object_after_key(script_body, '"dimensionValuesDisplayData"')
        if not obj:
            continue

        dim_map = obj.get("dimensionValuesDisplayData")
        if not isinstance(dim_map, dict):
            continue

        color_dim = None
        for key in dim_map.keys():
            if "color" in key.lower() or "色" in key:
                color_dim = key
                break
        if color_dim is None:
            continue

        value_map = dim_map.get(color_dim)
        if not isinstance(value_map, dict):
            continue

        for asin, label in value_map.items():
            normalized_asin = extract_asin(str(asin))
            color_number = extract_color_number_from_label(str(label))
            if normalized_asin and color_number:
                pairs.append((color_number, normalized_asin))
    return pairs


def extract_pairs_from_twister_json(html: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    keys = [
        '"twister-js-init-dpx-data"',
        '"asinVariationValues"',
    ]

    for script_body in SCRIPT_TAG_PATTERN.findall(html):
        for key in keys:
            obj = extract_json_object_after_key(script_body, key)
            if not obj:
                continue

            dim_data = obj.get("variationDisplayLabels") or obj.get("dimensionValuesDisplayData")
            if not isinstance(dim_data, dict):
                continue

            color_dim_key = ""
            for dim_key in dim_data.keys():
                if "color" in str(dim_key).lower() or "色" in str(dim_key):
                    color_dim_key = dim_key
                    break
            if not color_dim_key:
                continue

            color_map = dim_data.get(color_dim_key)
            if not isinstance(color_map, dict):
                continue

            for asin, label in color_map.items():
                normalized_asin = extract_asin(str(asin))
                color_number = extract_color_number_from_label(str(label))
                if normalized_asin and color_number:
                    pairs.append((color_number, normalized_asin))

    return pairs


def fetch_html(url: str) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
        },
    )
    with urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8", errors="ignore")


def fetch_html_playwright(url: str, timeout_ms: int, headless: bool) -> tuple[str, list[dict[str, str]]]:
    try:
        sync_api = importlib.import_module("playwright.sync_api")
        sync_playwright = sync_api.sync_playwright
    except Exception as exc:
        raise RuntimeError(
            "Playwright が見つかりません。開発依存をインストールしてください: "
            "pip install -r requirements-dev.txt && python -m playwright install chromium"
        ) from exc

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(locale="ja-JP")
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(2000)

        dom_items = page.evaluate(
            """
            () => {
              const selectors = [
                '#variation_color_name li',
                '#variation_color_name option',
                '[id*="variation_color"] li',
                '[id*="variation_color"] option',
                '[data-defaultasin]',
                '[data-asin]'
              ];
              const nodes = [];
              for (const sel of selectors) {
                for (const node of document.querySelectorAll(sel)) {
                  nodes.push(node);
                }
              }

              const results = [];
              for (const node of nodes) {
                const attrs = node.getAttributeNames ? node.getAttributeNames() : [];
                const bag = {
                  text: (node.textContent || '').trim(),
                  title: node.getAttribute('title') || '',
                  ariaLabel: node.getAttribute('aria-label') || '',
                  value: node.getAttribute('value') || '',
                  href: node.getAttribute('href') || '',
                  dataDefaultAsin: node.getAttribute('data-defaultasin') || '',
                  dataAsin: node.getAttribute('data-asin') || '',
                  dataDpUrl: node.getAttribute('data-dp-url') || '',
                  dataValue: node.getAttribute('data-value') || '',
                  attrsJson: ''
                };
                const obj = {};
                for (const attrName of attrs) {
                  obj[attrName] = node.getAttribute(attrName) || '';
                }
                bag.attrsJson = JSON.stringify(obj);
                results.push(bag);
              }
              return results;
            }
            """
        )

        html = page.content()
        browser.close()
        return html, dom_items


def extract_pairs_from_dom_items(dom_items: list[dict[str, str]]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in dom_items:
        label_candidates = [
            item.get("title", ""),
            item.get("ariaLabel", ""),
            item.get("text", ""),
            item.get("dataValue", ""),
            item.get("value", ""),
        ]
        color_number = ""
        for label in label_candidates:
            color_number = extract_color_number_from_label(label)
            if color_number:
                break
        if not color_number:
            continue

        asin_candidates = [
            item.get("dataDefaultAsin", ""),
            item.get("dataAsin", ""),
            item.get("dataDpUrl", ""),
            item.get("href", ""),
            item.get("attrsJson", ""),
        ]
        asin = extract_asin_from_text_candidates(asin_candidates)
        if asin:
            pairs.append((color_number, asin))
    return pairs


def extract_color_to_asin_map_http(url: str) -> dict[str, str]:
    html = fetch_html(url)
    ordered_pairs: list[tuple[str, str]] = []
    ordered_pairs.extend(extract_pairs_from_scripts(html))
    ordered_pairs.extend(extract_pairs_from_twister_json(html))
    ordered_pairs.extend(extract_pairs_from_tags(html))
    return merge_pairs_to_map(ordered_pairs)


def extract_color_to_asin_map_playwright(url: str, timeout_ms: int, headless: bool) -> dict[str, str]:
    html, dom_items = fetch_html_playwright(url, timeout_ms=timeout_ms, headless=headless)
    ordered_pairs: list[tuple[str, str]] = []
    ordered_pairs.extend(extract_pairs_from_dom_items(dom_items))
    ordered_pairs.extend(extract_pairs_from_scripts(html))
    ordered_pairs.extend(extract_pairs_from_twister_json(html))
    ordered_pairs.extend(extract_pairs_from_tags(html))
    return merge_pairs_to_map(ordered_pairs)


def read_color_number_list(file_path: Path) -> list[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"色番リストが見つかりません: {file_path}")

    numbers: list[str] = []
    seen: set[str] = set()

    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        color_number = normalize_color_number(stripped)
        if not color_number:
            continue
        if color_number in seen:
            continue
        seen.add(color_number)
        numbers.append(color_number)

    return numbers


def build_generated_rows(color_numbers: list[str], color_to_asin: dict[str, str]) -> list[list[str]]:
    rows = [["色番", "ASIN", "Amazonリンク"]]
    for color_number in color_numbers:
        asin = color_to_asin.get(color_number, "")
        link = f"https://www.amazon.co.jp/dp/{asin}?th=1" if asin else ""
        rows.append([color_number, asin, link])
    return rows


def write_csv(rows: list[list[str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def resolve_engine(engine: str, url: str, timeout_ms: int, headless: bool) -> tuple[dict[str, str], str]:
    if engine == "playwright":
        return extract_color_to_asin_map_playwright(url, timeout_ms=timeout_ms, headless=headless), "playwright"
    if engine == "http":
        return extract_color_to_asin_map_http(url), "http"

    try:
        return (
            extract_color_to_asin_map_playwright(url, timeout_ms=timeout_ms, headless=headless),
            "playwright",
        )
    except Exception:
        return extract_color_to_asin_map_http(url), "http"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "merinorainbow専用: Amazon色バリエーションから色番→ASINを取得し、"
            "色番リストtxtから再生成CSVを作成します。"
        )
    )
    parser.add_argument("--url", default=DEFAULT_PRODUCT_URL, help="merinorainbowの商品URL")
    parser.add_argument(
        "--color-list",
        default=str(DEFAULT_COLOR_LIST_PATH),
        help="色番リストtxt（一行一色番）",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV_PATH),
        help="再生成CSVの出力先",
    )
    parser.add_argument(
        "--dump-json",
        default=str(DEFAULT_OUTPUT_MAP_JSON_PATH),
        help="抽出した色番→ASINマップの出力先",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "playwright", "http"],
        default="auto",
        help="取得エンジン（既定:auto）",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=60000,
        help="playwright利用時のタイムアウト(ms)",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="playwright利用時にブラウザを可視化",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ファイルを書き込まない",
    )
    args = parser.parse_args()

    color_numbers = read_color_number_list(Path(args.color_list))
    color_to_asin, used_engine = resolve_engine(
        engine=args.engine,
        url=args.url,
        timeout_ms=args.timeout_ms,
        headless=not args.show_browser,
    )

    rows = build_generated_rows(color_numbers, color_to_asin)
    assigned = sum(1 for row in rows[1:] if row[1])
    missing = len(rows) - 1 - assigned

    if not args.dry_run:
        write_csv(rows, Path(args.output_csv))
        if args.dump_json:
            out_json = Path(args.dump_json)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(color_to_asin, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "engine": used_engine,
        "list_count": len(color_numbers),
        "assigned": assigned,
        "missing": missing,
        "output_csv": str(Path(args.output_csv)),
        "output_json": str(Path(args.dump_json)) if args.dump_json else "",
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
