"""
Generate qualitative_comparison_grid.png from individual qualitative result tiles.
Organizes tiles by category into a publication-quality grid.
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FIGURES_DIR = Path(r"c:\Users\Zachy\OneDrive\Desktop\Water Body Segmentation\manuscript\figures")
TILES_DIR = FIGURES_DIR / "qualitative_results_best"
OUTPUT_PATH = FIGURES_DIR / "qualitative_comparison_grid.png"

# Define categories and their display labels
CATEGORIES = [
    ("model_wins_big",   "Model Wins\n(Large Margin)"),
    ("model_wins_small", "Model Wins\n(Small Margin)"),
    ("both_excel",       "Both Methods\nExcel"),
    ("ndwi_wins_small",  "NDWI Wins\n(Small Margin)"),
    ("ndwi_wins_big",    "NDWI Wins\n(Large Margin)"),
    ("both_struggle",    "Both Methods\nStruggle"),
    ("tie",              "Tie"),
]

def load_tiles_by_category():
    """Load up to 2 tiles per category, sorted."""
    tiles = {}
    for prefix, label in CATEGORIES:
        matches = sorted(TILES_DIR.glob(f"{prefix}_*.png"))
        tiles[prefix] = (label, matches[:2])
    return tiles

def make_grid():
    tiles = load_tiles_by_category()

    # Collect all rows: (category_label, [img, img])
    rows = []
    for prefix, (label, paths) in tiles.items():
        if not paths:
            continue
        imgs = [Image.open(p) for p in paths]
        rows.append((label, imgs))

    if not rows:
        print("No tiles found!")
        return

    # Determine tile dimensions (resize all to same width)
    target_w = 1100
    resized_rows = []
    for label, imgs in rows:
        resized = []
        for img in imgs:
            ratio = target_w / img.width
            new_h = int(img.height * ratio)
            resized.append(img.resize((target_w, new_h), Image.LANCZOS))
        resized_rows.append((label, resized))

    tile_h = resized_rows[0][1][0].height
    tile_w = target_w

    # Layout: label column (200px) + up to 2 tiles per row
    label_col_w = 200
    n_cols = 2
    padding = 10
    header_h = 60
    row_label_font_size = 22

    total_w = label_col_w + n_cols * (tile_w + padding) + padding
    total_h = header_h + len(resized_rows) * (tile_h + padding) + padding

    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Try to load a font
    try:
        font_title = ImageFont.truetype("arial.ttf", 28)
        font_label = ImageFont.truetype("arial.ttf", row_label_font_size)
        font_col   = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font_title = ImageFont.load_default()
        font_label = font_title
        font_col   = font_title

    # Title
    title = "Qualitative Comparison: Deep Learning vs. Classical NDWI"
    draw.text((total_w // 2, padding + 10), title, fill=(0, 0, 0),
              font=font_title, anchor="mm")

    # Column headers
    col_headers = ["Representative Example 1", "Representative Example 2"]
    for ci, ch in enumerate(col_headers):
        cx = label_col_w + padding + ci * (tile_w + padding) + tile_w // 2
        draw.text((cx, header_h - 18), ch, fill=(60, 60, 60),
                  font=font_col, anchor="mm")

    # Draw rows
    for ri, (label, imgs) in enumerate(resized_rows):
        y = header_h + ri * (tile_h + padding) + padding

        # Category label (centered vertically in the row)
        label_x = label_col_w // 2
        label_y = y + tile_h // 2
        # Draw background for label
        draw.rectangle([0, y, label_col_w - 5, y + tile_h], fill=(240, 240, 248))
        # Draw text (multi-line)
        lines = label.split("\n")
        line_h = row_label_font_size + 4
        start_y = label_y - (len(lines) * line_h) // 2
        for li, line in enumerate(lines):
            draw.text((label_x, start_y + li * line_h), line,
                      fill=(30, 30, 100), font=font_label, anchor="mm")

        # Paste tiles
        for ci, img in enumerate(imgs[:n_cols]):
            x = label_col_w + padding + ci * (tile_w + padding)
            canvas.paste(img, (x, y))

        # Separator line
        if ri < len(resized_rows) - 1:
            sep_y = y + tile_h + padding // 2
            draw.line([(0, sep_y), (total_w, sep_y)], fill=(200, 200, 200), width=1)

    canvas.save(OUTPUT_PATH, dpi=(300, 300))
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Size: {canvas.size[0]}x{canvas.size[1]} px")

if __name__ == "__main__":
    make_grid()
