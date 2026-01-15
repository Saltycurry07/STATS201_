# STATS 201 Course Project

This repository includes a Fox News scraper that downloads Donald Trump images
with a detected frontal face and records the associated article date.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python scripts/fox_trump_face_scraper.py --output-dir output --max-pages 3 --delay 1.0
```

Outputs:
- `output/images/` contains downloaded images with the article date in the filename.
- `output/results.csv` lists article URL, title, date, image URL, and local path.

> Note: Please ensure you comply with Fox News terms of service and robots.txt
> before running large-scale scraping jobs.
