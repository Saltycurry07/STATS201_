#!/usr/bin/env python3
"""Scrape Fox News for Donald Trump articles and download frontal-face photos.

This script searches Fox News for Donald Trump articles, extracts article dates
and image URLs, downloads images, and filters for frontal faces using OpenCV's
Haar cascades. Results are recorded in a CSV with dates.
"""

import argparse
import csv
import hashlib
import os
import time
from datetime import datetime
from io import BytesIO
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup

SEARCH_URL = "https://www.foxnews.com/search-results/search"
USER_AGENT = "Mozilla/5.0 (compatible; STATS201-Project/1.0; +https://example.com)"
HEADERS = {"User-Agent": USER_AGENT}


def fetch_html(url, params=None, timeout=15):
    response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_search_results(html):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for card in soup.select(".search-list .info .title a"):
        href = card.get("href")
        if href:
            links.append(urljoin("https://www.foxnews.com", href))
    return links


def parse_article(html):
    soup = BeautifulSoup(html, "html.parser")

    title = soup.find("h1")
    title_text = title.get_text(strip=True) if title else ""

    date_text = None
    date_meta = soup.find("meta", {"name": "dcterms.created"})
    if date_meta and date_meta.get("content"):
        date_text = date_meta["content"]
    else:
        time_tag = soup.find("time")
        if time_tag and time_tag.get("datetime"):
            date_text = time_tag["datetime"]

    article_date = normalize_date(date_text)

    image_urls = []
    for img in soup.select("article img"):
        src = img.get("data-src") or img.get("src")
        if src and src.startswith("http"):
            image_urls.append(src)

    return title_text, article_date, image_urls


def normalize_date(date_text):
    if not date_text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%Z", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(date_text, fmt)
            return parsed.date().isoformat()
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(date_text.replace("Z", "+00:00"))
        return parsed.date().isoformat()
    except ValueError:
        return None


def download_image(url, timeout=20):
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.content, response.headers.get("Content-Type", "")


def is_frontal_face(image_bytes):
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return len(faces) > 0


def safe_filename(url, date_text, ext):
    hash_part = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    date_part = date_text or "unknown-date"
    return f"trump_{date_part}_{hash_part}{ext}"


def guess_extension(content_type, url):
    if content_type:
        if "jpeg" in content_type or "jpg" in content_type:
            return ".jpg"
        if "png" in content_type:
            return ".png"
        if "webp" in content_type:
            return ".webp"
    path = urlparse(url).path
    _, ext = os.path.splitext(path)
    return ext if ext else ".jpg"


def scrape(args):
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    results_path = os.path.join(args.output_dir, "results.csv")
    seen_articles = set()
    saved_rows = []

    for page in range(1, args.max_pages + 1):
        params = {"q": "Donald Trump", "page": page}
        html = fetch_html(SEARCH_URL, params=params)
        article_links = parse_search_results(html)
        if not article_links:
            break

        for link in article_links:
            if link in seen_articles:
                continue
            seen_articles.add(link)

            article_html = fetch_html(link)
            title, article_date, image_urls = parse_article(article_html)

            for img_url in image_urls:
                try:
                    image_bytes, content_type = download_image(img_url)
                except requests.RequestException:
                    continue
                if not is_frontal_face(image_bytes):
                    continue

                ext = guess_extension(content_type, img_url)
                filename = safe_filename(img_url, article_date, ext)
                file_path = os.path.join(images_dir, filename)
                if not os.path.exists(file_path):
                    with open(file_path, "wb") as file_handle:
                        file_handle.write(image_bytes)

                saved_rows.append(
                    {
                        "article_url": link,
                        "article_title": title,
                        "article_date": article_date or "",
                        "image_url": img_url,
                        "local_path": os.path.relpath(file_path, args.output_dir),
                    }
                )

            time.sleep(args.delay)

    with open(results_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["article_url", "article_title", "article_date", "image_url", "local_path"],
        )
        writer.writeheader()
        writer.writerows(saved_rows)

    print(f"Saved {len(saved_rows)} images. Results written to {results_path}.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Download frontal-face photos of Donald Trump from Fox News with dates."
    )
    parser.add_argument("--output-dir", default="output", help="Directory to store images/CSV")
    parser.add_argument("--max-pages", type=int, default=3, help="Search pages to scan")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests")
    return parser


def main():
    args = build_parser().parse_args()
    scrape(args)


if __name__ == "__main__":
    main()
