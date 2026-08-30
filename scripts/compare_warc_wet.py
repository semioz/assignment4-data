import argparse
import gzip

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract import extract_text_from_html_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("warc_path")
    parser.add_argument("wet_path")
    args = parser.parse_args()

    with gzip.open(args.wet_path, "rb") as wet_file:
        wet_record = next(
            ArchiveIterator(wet_file, record_types=WarcRecordType.conversion)
        )
        url = wet_record.headers.get("WARC-Target-URI")
        wet_text = wet_record.reader.read().decode("utf-8", errors="replace")

    with gzip.open(args.warc_path, "rb") as warc_file:
        for warc_record in ArchiveIterator(
            warc_file, record_types=WarcRecordType.response, parse_http=True
        ):
            if warc_record.headers.get("WARC-Target-URI") == url:
                our_text = extract_text_from_html_bytes(warc_record.reader.read())
                print(f"URL: {url}\n\nOURS:\n{our_text[:2000]}\n\nWET:\n{wet_text[:2000]}")
                return

    raise SystemExit(f"No WARC response found for {url}")


if __name__ == "__main__":
    main()
