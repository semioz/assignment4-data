import argparse
import concurrent.futures
import gzip
import os
from collections import Counter
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm

from cs336_data.language_id import identify_language
from cs336_data.pii import mask_emails, mask_ips, mask_phone_numbers
from cs336_data.quality import classify_quality, passes_gopher_quality_filters


def output_path(input_path: Path, output_directory: Path) -> Path:
    name = input_path.name
    suffix = ".warc.wet.gz"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return output_directory / f"{name}.txt.gz"


def process_wet_file(input_path: str, output_directory: str) -> Counter:
    input_file = Path(input_path)
    output_file = output_path(input_file, Path(output_directory))
    counts = Counter()

    with gzip.open(input_file, "rb") as source, gzip.open(output_file, "wt") as output:
        for record in ArchiveIterator(source, record_types=WarcRecordType.conversion):
            counts["records"] += 1
            text = record.reader.read().decode("utf-8", errors="replace")
            language, confidence = identify_language(text)
            if language != "en" or confidence < 0.7:
                counts["language_rejected"] += 1
                continue
            if not passes_gopher_quality_filters(text):
                counts["gopher_rejected"] += 1
                continue

            label, _ = classify_quality(text)
            if label != "wiki":
                counts["quality_rejected"] += 1
                continue

            text, masked = mask_emails(text)
            text, count = mask_phone_numbers(text)
            masked += count
            text, count = mask_ips(text)
            counts["pii_masks"] += masked + count
            output.write(" ".join(text.split()))
            output.write("\n")
            counts["kept"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_paths", nargs="+", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    args.output_directory.mkdir(parents=True, exist_ok=True)
    workers = args.workers or os.cpu_count() or 1
    counts = Counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_wet_file, str(path), str(args.output_directory))
            for path in args.input_paths
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            counts.update(future.result())

    total = counts["records"]
    for key in (
        "records",
        "language_rejected",
        "gopher_rejected",
        "quality_rejected",
        "kept",
        "pii_masks",
    ):
        count = counts[key]
        suffix = f" ({count / total:.2%})" if key != "pii_masks" and total else ""
        print(f"{key}: {count}{suffix}")


if __name__ == "__main__":
    main()
