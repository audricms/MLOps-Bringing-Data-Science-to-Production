import re
import xml.etree.ElementTree as ET
from io import StringIO
from urllib.parse import quote, urlsplit, urlunsplit

import pandas as pd
import requests


def natural_key(s: str):
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def encode_url_path(url: str) -> str:
    """Return URL with an encoded path while preserving scheme/host/query."""
    parts = urlsplit(url)
    encoded_path = quote(parts.path, safe="/")
    return urlunsplit(
        (parts.scheme, parts.netloc, encoded_path, parts.query, parts.fragment)
    )


def build_public_toy_csv_url(base_url: str, filename: str) -> str:
    """Build a public URL for a CSV file stored in MinIO/S3."""
    normalized_base = encode_url_path(base_url.rstrip("/"))
    return f"{normalized_base}/{quote(filename)}"


def list_s3_csv_files(base_url: str) -> list[str]:
    parts = urlsplit(base_url)
    path_parts = parts.path.lstrip("/").split("/", 1)
    bucket = path_parts[0]
    prefix = (path_parts[1].rstrip("/") + "/") if len(path_parts) > 1 else ""
    list_url = (
        f"{parts.scheme}://{parts.netloc}/{bucket}?list-type=2&prefix={quote(prefix)}"
    )
    response = requests.get(list_url, timeout=10)
    response.raise_for_status()
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    root = ET.fromstring(response.text)
    return sorted(
        [
            key.text.split("/")[-1]
            for key in root.findall(".//s3:Key", ns)
            if key.text and key.text.endswith(".csv")
        ],
        key=natural_key,
    )


def read_csv_from_public_url(url: str) -> pd.DataFrame:
    """Read CSV content from an HTTP(S) URL using requests."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))
