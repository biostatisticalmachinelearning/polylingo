#!/usr/bin/env python3
"""
Unicode Character Dataset Generator

Generates 64x64 PNG images for Unicode characters across world writing systems
using Noto fonts, organized by script with JSON metadata.
"""

import json
import os
import unicodedata
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

# Configuration
IMAGE_SIZE = 64
BACKGROUND_COLOR = "white"
TEXT_COLOR = "black"
OUTPUT_DIR = Path("data/unicode_chars")
FONTS_DIR = Path("fonts")

# Script definitions: (name, unicode_ranges, font_filename, font_url)
# Font URLs point to Google's Noto fonts GitHub releases
SCRIPTS = [
    # Latin scripts (using NotoSans-Regular)
    {
        "name": "latin",
        "ranges": [(0x0020, 0x007F), (0x00A0, 0x00FF), (0x0100, 0x017F), (0x0180, 0x024F)],
        "font": "NotoSans-Regular.ttf",
        "url": "https://github.com/notofonts/latin-greek-cyrillic/releases/download/NotoSans-v2.015/NotoSans-v2.015.zip",
        "zip_path": "NotoSans/unhinted/ttf/NotoSans-Regular.ttf",
    },
    {
        "name": "cyrillic",
        "ranges": [(0x0400, 0x04FF), (0x0500, 0x052F)],
        "font": "NotoSans-Regular.ttf",
        "url": None,  # Same font as Latin
    },
    {
        "name": "greek",
        "ranges": [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
        "font": "NotoSans-Regular.ttf",
        "url": None,  # Same font as Latin
    },
    # Arabic
    {
        "name": "arabic",
        "ranges": [(0x0600, 0x06FF), (0x0750, 0x077F)],
        "font": "NotoSansArabic-Regular.ttf",
        "url": "https://github.com/notofonts/arabic/releases/download/NotoSansArabic-v2.013/NotoSansArabic-v2.013.zip",
        "zip_path": "NotoSansArabic/unhinted/ttf/NotoSansArabic-Regular.ttf",
    },
    # Hebrew
    {
        "name": "hebrew",
        "ranges": [(0x0590, 0x05FF)],
        "font": "NotoSansHebrew-Regular.ttf",
        "url": "https://github.com/notofonts/hebrew/releases/download/NotoSansHebrew-v2.003/NotoSansHebrew-v2.003.zip",
        "zip_path": "NotoSansHebrew/unhinted/ttf/NotoSansHebrew-Regular.ttf",
    },
    # Indic scripts
    {
        "name": "devanagari",
        "ranges": [(0x0900, 0x097F)],
        "font": "NotoSansDevanagari-Regular.ttf",
        "url": "https://github.com/notofonts/devanagari/releases/download/NotoSansDevanagari-v2.005/NotoSansDevanagari-v2.005.zip",
        "zip_path": "NotoSansDevanagari/unhinted/ttf/NotoSansDevanagari-Regular.ttf",
    },
    {
        "name": "bengali",
        "ranges": [(0x0980, 0x09FF)],
        "font": "NotoSansBengali-Regular.ttf",
        "url": "https://github.com/notofonts/bengali/releases/download/NotoSansBengali-v2.003/NotoSansBengali-v2.003.zip",
        "zip_path": "NotoSansBengali/unhinted/ttf/NotoSansBengali-Regular.ttf",
    },
    {
        "name": "tamil",
        "ranges": [(0x0B80, 0x0BFF)],
        "font": "NotoSansTamil-Regular.ttf",
        "url": "https://github.com/notofonts/tamil/releases/download/NotoSansTamil-v2.004/NotoSansTamil-v2.004.zip",
        "zip_path": "NotoSansTamil/unhinted/ttf/NotoSansTamil-Regular.ttf",
    },
    {
        "name": "telugu",
        "ranges": [(0x0C00, 0x0C7F)],
        "font": "NotoSansTelugu-Regular.ttf",
        "url": "https://github.com/notofonts/telugu/releases/download/NotoSansTelugu-v2.005/NotoSansTelugu-v2.005.zip",
        "zip_path": "NotoSansTelugu/unhinted/ttf/NotoSansTelugu-Regular.ttf",
    },
    {
        "name": "kannada",
        "ranges": [(0x0C80, 0x0CFF)],
        "font": "NotoSansKannada-Regular.ttf",
        "url": "https://github.com/notofonts/kannada/releases/download/NotoSansKannada-v2.006/NotoSansKannada-v2.006.zip",
        "zip_path": "NotoSansKannada/unhinted/ttf/NotoSansKannada-Regular.ttf",
    },
    {
        "name": "malayalam",
        "ranges": [(0x0D00, 0x0D7F)],
        "font": "NotoSansMalayalam-Regular.ttf",
        "url": "https://github.com/notofonts/malayalam/releases/download/NotoSansMalayalam-v2.104/NotoSansMalayalam-v2.104.zip",
        "zip_path": "NotoSansMalayalam/unhinted/ttf/NotoSansMalayalam-Regular.ttf",
    },
    {
        "name": "gujarati",
        "ranges": [(0x0A80, 0x0AFF)],
        "font": "NotoSansGujarati-Regular.ttf",
        "url": "https://github.com/notofonts/gujarati/releases/download/NotoSerifGujarati-v2.106/NotoSerifGujarati-v2.106.zip",
        "zip_path": "NotoSerifGujarati/unhinted/ttf/NotoSerifGujarati-Regular.ttf",
    },
    {
        "name": "gurmukhi",
        "ranges": [(0x0A00, 0x0A7F)],
        "font": "NotoSansGurmukhi-Regular.ttf",
        "url": "https://github.com/notofonts/gurmukhi/releases/download/NotoSerifGurmukhi-v2.004/NotoSerifGurmukhi-v2.004.zip",
        "zip_path": "NotoSerifGurmukhi/unhinted/ttf/NotoSerifGurmukhi-Regular.ttf",
    },
    {
        "name": "oriya",
        "ranges": [(0x0B00, 0x0B7F)],
        "font": "NotoSansOriya-Regular.ttf",
        "url": "https://github.com/notofonts/oriya/releases/download/NotoSansOriya-v2.005/NotoSansOriya-v2.005.zip",
        "zip_path": "NotoSansOriya/unhinted/ttf/NotoSansOriya-Regular.ttf",
    },
    {
        "name": "sinhala",
        "ranges": [(0x0D80, 0x0DFF)],
        "font": "NotoSansSinhala-Regular.ttf",
        "url": "https://github.com/notofonts/sinhala/releases/download/NotoSansSinhala-v3.000/NotoSansSinhala-v3.000.zip",
        "zip_path": "NotoSansSinhala/unhinted/ttf/NotoSansSinhala-Regular.ttf",
    },
    # Southeast Asian scripts
    {
        "name": "thai",
        "ranges": [(0x0E00, 0x0E7F)],
        "font": "NotoSansThai-Regular.ttf",
        "url": "https://github.com/notofonts/thai/releases/download/NotoSansThai-v2.002/NotoSansThai-v2.002.zip",
        "zip_path": "NotoSansThai/unhinted/ttf/NotoSansThai-Regular.ttf",
    },
    {
        "name": "lao",
        "ranges": [(0x0E80, 0x0EFF)],
        "font": "NotoSansLao-Regular.ttf",
        "url": "https://github.com/notofonts/lao/releases/download/NotoSansLao-v2.003/NotoSansLao-v2.003.zip",
        "zip_path": "NotoSansLao/unhinted/ttf/NotoSansLao-Regular.ttf",
    },
    {
        "name": "myanmar",
        "ranges": [(0x1000, 0x109F)],
        "font": "NotoSansMyanmar-Regular.ttf",
        "url": "https://github.com/notofonts/myanmar/releases/download/NotoSansMyanmar-v2.107/NotoSansMyanmar-v2.107.zip",
        "zip_path": "NotoSansMyanmar/unhinted/ttf/NotoSansMyanmar-Regular.ttf",
    },
    {
        "name": "khmer",
        "ranges": [(0x1780, 0x17FF)],
        "font": "NotoSansKhmer-Regular.ttf",
        "url": "https://github.com/notofonts/khmer/releases/download/NotoSansKhmer-v2.004/NotoSansKhmer-v2.004.zip",
        "zip_path": "NotoSansKhmer/unhinted/ttf/NotoSansKhmer-Regular.ttf",
    },
    {
        "name": "tibetan",
        "ranges": [(0x0F00, 0x0FFF)],
        "font": "NotoSerifTibetan-Regular.ttf",
        "url": "https://github.com/notofonts/tibetan/releases/download/NotoSerifTibetan-v2.103/NotoSerifTibetan-v2.103.zip",
        "zip_path": "NotoSerifTibetan/unhinted/ttf/NotoSerifTibetan-Regular.ttf",
    },
    # Caucasian scripts
    {
        "name": "georgian",
        "ranges": [(0x10A0, 0x10FF), (0x2D00, 0x2D2F)],
        "font": "NotoSansGeorgian-Regular.ttf",
        "url": "https://github.com/notofonts/georgian/releases/download/NotoSansGeorgian-v2.005/NotoSansGeorgian-v2.005.zip",
        "zip_path": "NotoSansGeorgian/unhinted/ttf/NotoSansGeorgian-Regular.ttf",
    },
    {
        "name": "armenian",
        "ranges": [(0x0530, 0x058F)],
        "font": "NotoSansArmenian-Regular.ttf",
        "url": "https://github.com/notofonts/armenian/releases/download/NotoSansArmenian-v2.008/NotoSansArmenian-v2.008.zip",
        "zip_path": "NotoSansArmenian/unhinted/ttf/NotoSansArmenian-Regular.ttf",
    },
    # African scripts
    {
        "name": "ethiopic",
        "ranges": [(0x1200, 0x137F)],
        "font": "NotoSerifEthiopic-Regular.ttf",
        "url": "https://github.com/notofonts/ethiopic/releases/download/NotoSerifEthiopic-v2.102/NotoSerifEthiopic-v2.102.zip",
        "zip_path": "NotoSerifEthiopic/unhinted/ttf/NotoSerifEthiopic-Regular.ttf",
    },
    {
        "name": "thaana",
        "ranges": [(0x0780, 0x07BF)],
        "font": "NotoSansThaana-Regular.ttf",
        "url": "https://github.com/notofonts/thaana/releases/download/NotoSansThaana-v3.001/NotoSansThaana-v3.001.zip",
        "zip_path": "NotoSansThaana/unhinted/ttf/NotoSansThaana-Regular.ttf",
    },
    {
        "name": "nko",
        "ranges": [(0x07C0, 0x07FF)],
        "font": "NotoSansNKo-Regular.ttf",
        "url": "https://github.com/notofonts/nko/releases/download/NotoSansNKo-v2.003/NotoSansNKo-v2.003.zip",
        "zip_path": "NotoSansNKo/unhinted/ttf/NotoSansNKo-Regular.ttf",
    },
    {
        "name": "tifinagh",
        "ranges": [(0x2D30, 0x2D7F)],
        "font": "NotoSansTifinagh-Regular.ttf",
        "url": "https://github.com/notofonts/tifinagh/releases/download/NotoSansTifinagh-v2.006/NotoSansTifinagh-v2.006.zip",
        "zip_path": "NotoSansTifinagh/unhinted/ttf/NotoSansTifinagh-Regular.ttf",
    },
    {
        "name": "vai",
        "ranges": [(0xA500, 0xA63F)],
        "font": "NotoSansVai-Regular.ttf",
        "url": "https://github.com/notofonts/vai/releases/download/NotoSansVai-v2.001/NotoSansVai-v2.001.zip",
        "zip_path": "NotoSansVai/unhinted/ttf/NotoSansVai-Regular.ttf",
    },
    # CJK Scripts (using Google Fonts static releases)
    {
        "name": "han_cjk",
        "ranges": [(0x4E00, 0x9FFF)],  # CJK Unified Ideographs
        "font": "NotoSansSC-Regular.otf",
        "url": "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/08_NotoSansCJKsc.zip",
        "zip_path": "NotoSansCJKsc-Regular.otf",
        "common_chars": 8000,  # Limit to common characters
    },
    {
        "name": "hiragana",
        "ranges": [(0x3040, 0x309F)],
        "font": "NotoSansJP-Regular.otf",
        "url": "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/06_NotoSansCJKjp.zip",
        "zip_path": "NotoSansCJKjp-Regular.otf",
    },
    {
        "name": "katakana",
        "ranges": [(0x30A0, 0x30FF)],
        "font": "NotoSansJP-Regular.otf",
        "url": None,  # Same font as hiragana
    },
    {
        "name": "hangul",
        "ranges": [(0xAC00, 0xD7AF)],  # Hangul Syllables
        "font": "NotoSansKR-Regular.otf",
        "url": "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/07_NotoSansCJKkr.zip",
        "zip_path": "NotoSansCJKkr-Regular.otf",
        "common_chars": 2350,  # Common syllables
    },
    # Additional scripts
    {
        "name": "mongolian",
        "ranges": [(0x1800, 0x18AF)],
        "font": "NotoSansMongolian-Regular.ttf",
        "url": "https://github.com/notofonts/mongolian/releases/download/NotoSansMongolian-v3.002/NotoSansMongolian-v3.002.zip",
        "zip_path": "NotoSansMongolian/unhinted/ttf/NotoSansMongolian-Regular.ttf",
    },
    {
        "name": "syriac",
        "ranges": [(0x0700, 0x074F)],
        "font": "NotoSansSyriacWestern-Regular.ttf",
        "url": "https://github.com/notofonts/syriac/releases/download/NotoSansSyriacWestern-v3.001/NotoSansSyriacWestern-v3.001.zip",
        "zip_path": "NotoSansSyriacWestern/unhinted/ttf/NotoSansSyriacWestern-Regular.ttf",
    },
    {
        "name": "canadian_aboriginal",
        "ranges": [(0x1400, 0x167F)],
        "font": "NotoSansCanadianAboriginal-Regular.ttf",
        "url": "https://github.com/notofonts/canadian-aboriginal/releases/download/NotoSansCanadianAboriginal-v2.003/NotoSansCanadianAboriginal-v2.003.zip",
        "zip_path": "NotoSansCanadianAboriginal/unhinted/ttf/NotoSansCanadianAboriginal-Regular.ttf",
    },
    {
        "name": "cherokee",
        "ranges": [(0x13A0, 0x13FF)],
        "font": "NotoSansCherokee-Regular.ttf",
        "url": "https://github.com/notofonts/cherokee/releases/download/NotoSansCherokee-v2.001/NotoSansCherokee-v2.001.zip",
        "zip_path": "NotoSansCherokee/unhinted/ttf/NotoSansCherokee-Regular.ttf",
    },
    {
        "name": "javanese",
        "ranges": [(0xA980, 0xA9DF)],
        "font": "NotoSansJavanese-Regular.ttf",
        "url": "https://github.com/notofonts/javanese/releases/download/NotoSansJavanese-v2.005/NotoSansJavanese-v2.005.zip",
        "zip_path": "NotoSansJavanese/unhinted/ttf/NotoSansJavanese-Regular.ttf",
    },
    {
        "name": "balinese",
        "ranges": [(0x1B00, 0x1B7F)],
        "font": "NotoSansBalinese-Regular.ttf",
        "url": "https://github.com/notofonts/balinese/releases/download/NotoSansBalinese-v2.005/NotoSansBalinese-v2.005.zip",
        "zip_path": "NotoSansBalinese/unhinted/ttf/NotoSansBalinese-Regular.ttf",
    },
    {
        "name": "sundanese",
        "ranges": [(0x1B80, 0x1BBF)],
        "font": "NotoSansSundanese-Regular.ttf",
        "url": "https://github.com/notofonts/sundanese/releases/download/NotoSansSundanese-v2.004/NotoSansSundanese-v2.004.zip",
        "zip_path": "NotoSansSundanese/unhinted/ttf/NotoSansSundanese-Regular.ttf",
    },
    {
        "name": "tagalog",
        "ranges": [(0x1700, 0x171F)],
        "font": "NotoSansTagalog-Regular.ttf",
        "url": "https://github.com/notofonts/tagalog/releases/download/NotoSansTagalog-v2.001/NotoSansTagalog-v2.001.zip",
        "zip_path": "NotoSansTagalog/unhinted/ttf/NotoSansTagalog-Regular.ttf",
    },
    {
        "name": "buginese",
        "ranges": [(0x1A00, 0x1A1F)],
        "font": "NotoSansBuginese-Regular.ttf",
        "url": "https://github.com/notofonts/buginese/releases/download/NotoSansBuginese-v2.002/NotoSansBuginese-v2.002.zip",
        "zip_path": "NotoSansBuginese/unhinted/ttf/NotoSansBuginese-Regular.ttf",
    },
    {
        "name": "cham",
        "ranges": [(0xAA00, 0xAA5F)],
        "font": "NotoSansCham-Regular.ttf",
        "url": "https://github.com/notofonts/cham/releases/download/NotoSansCham-v2.003/NotoSansCham-v2.003.zip",
        "zip_path": "NotoSansCham/unhinted/ttf/NotoSansCham-Regular.ttf",
    },
    {
        "name": "tai_le",
        "ranges": [(0x1950, 0x197F)],
        "font": "NotoSansTaiLe-Regular.ttf",
        "url": "https://github.com/notofonts/tai-le/releases/download/NotoSansTaiLe-v2.002/NotoSansTaiLe-v2.002.zip",
        "zip_path": "NotoSansTaiLe/unhinted/ttf/NotoSansTaiLe-Regular.ttf",
    },
    {
        "name": "new_tai_lue",
        "ranges": [(0x1980, 0x19DF)],
        "font": "NotoSansNewTaiLue-Regular.ttf",
        "url": "https://github.com/notofonts/new-tai-lue/releases/download/NotoSansNewTaiLue-v2.003/NotoSansNewTaiLue-v2.003.zip",
        "zip_path": "NotoSansNewTaiLue/unhinted/ttf/NotoSansNewTaiLue-Regular.ttf",
    },
    {
        "name": "tai_tham",
        "ranges": [(0x1A20, 0x1AAF)],
        "font": "NotoSansTaiTham-Regular.ttf",
        "url": "https://github.com/notofonts/tai-tham/releases/download/NotoSansTaiTham-v2.002/NotoSansTaiTham-v2.002.zip",
        "zip_path": "NotoSansTaiTham/unhinted/ttf/NotoSansTaiTham-Regular.ttf",
    },
    {
        "name": "tai_viet",
        "ranges": [(0xAA80, 0xAADF)],
        "font": "NotoSansTaiViet-Regular.ttf",
        "url": "https://github.com/notofonts/tai-viet/releases/download/NotoSansTaiViet-v2.002/NotoSansTaiViet-v2.002.zip",
        "zip_path": "NotoSansTaiViet/unhinted/ttf/NotoSansTaiViet-Regular.ttf",
    },
    {
        "name": "lepcha",
        "ranges": [(0x1C00, 0x1C4F)],
        "font": "NotoSansLepcha-Regular.ttf",
        "url": "https://github.com/notofonts/lepcha/releases/download/NotoSansLepcha-v2.006/NotoSansLepcha-v2.006.zip",
        "zip_path": "NotoSansLepcha/unhinted/ttf/NotoSansLepcha-Regular.ttf",
    },
    {
        "name": "ol_chiki",
        "ranges": [(0x1C50, 0x1C7F)],
        "font": "NotoSansOlChiki-Regular.ttf",
        "url": "https://github.com/notofonts/ol-chiki/releases/download/NotoSansOlChiki-v2.003/NotoSansOlChiki-v2.003.zip",
        "zip_path": "NotoSansOlChiki/unhinted/ttf/NotoSansOlChiki-Regular.ttf",
    },
    {
        "name": "limbu",
        "ranges": [(0x1900, 0x194F)],
        "font": "NotoSansLimbu-Regular.ttf",
        "url": "https://github.com/notofonts/limbu/releases/download/NotoSansLimbu-v2.003/NotoSansLimbu-v2.003.zip",
        "zip_path": "NotoSansLimbu/unhinted/ttf/NotoSansLimbu-Regular.ttf",
    },
    {
        "name": "meetei_mayek",
        "ranges": [(0xABC0, 0xABFF)],
        "font": "NotoSansMeeteiMayek-Regular.ttf",
        "url": "https://github.com/notofonts/meetei-mayek/releases/download/NotoSansMeeteiMayek-v2.002/NotoSansMeeteiMayek-v2.002.zip",
        "zip_path": "NotoSansMeeteiMayek/unhinted/ttf/NotoSansMeeteiMayek-Regular.ttf",
    },
    {
        "name": "saurashtra",
        "ranges": [(0xA880, 0xA8DF)],
        "font": "NotoSansSaurashtra-Regular.ttf",
        "url": "https://github.com/notofonts/saurashtra/releases/download/NotoSansSaurashtra-v2.002/NotoSansSaurashtra-v2.002.zip",
        "zip_path": "NotoSansSaurashtra/unhinted/ttf/NotoSansSaurashtra-Regular.ttf",
    },
    {
        "name": "kayah_li",
        "ranges": [(0xA900, 0xA92F)],
        "font": "NotoSansKayahLi-Regular.ttf",
        "url": "https://github.com/notofonts/kayah-li/releases/download/NotoSansKayahLi-v2.002/NotoSansKayahLi-v2.002.zip",
        "zip_path": "NotoSansKayahLi/unhinted/ttf/NotoSansKayahLi-Regular.ttf",
    },
    # Symbols
    {
        "name": "symbols",
        "ranges": [(0x2000, 0x206F), (0x2190, 0x21FF), (0x2200, 0x22FF), (0x2300, 0x23FF),
                   (0x25A0, 0x25FF), (0x2600, 0x26FF), (0x2700, 0x27BF)],
        "font": "NotoSansSymbols2-Regular.ttf",
        "url": "https://github.com/notofonts/symbols/releases/download/NotoSansSymbols2-v2.008/NotoSansSymbols2-v2.008.zip",
        "zip_path": "NotoSansSymbols2/unhinted/ttf/NotoSansSymbols2-Regular.ttf",
    },
]

# Cache for loaded fonts
font_cache = {}


def download_font(script_info: dict) -> bool:
    """Download a Noto font if not already present."""
    font_path = FONTS_DIR / script_info["font"]

    if font_path.exists():
        print(f"  Font already exists: {script_info['font']}")
        return True

    if script_info.get("url") is None:
        # Font should already exist from another script
        print(f"  Font should exist from another script: {script_info['font']}")
        return font_path.exists()

    url = script_info["url"]
    zip_path = script_info.get("zip_path", "")

    print(f"  Downloading: {url}")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        import io
        import zipfile

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            # Find the font file in the zip
            for name in zf.namelist():
                if name.endswith(zip_path) or name == zip_path:
                    # Extract the font
                    font_data = zf.read(name)
                    font_path.write_bytes(font_data)
                    print(f"  Extracted: {script_info['font']}")
                    return True

            # If zip_path not found, try to find any matching font
            for name in zf.namelist():
                if name.endswith('.ttf') or name.endswith('.otf'):
                    if 'Regular' in name:
                        font_data = zf.read(name)
                        font_path.write_bytes(font_data)
                        print(f"  Extracted: {script_info['font']} (from {name})")
                        return True

        print(f"  Warning: Could not find font in zip: {zip_path}")
        return False

    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False


def get_font(font_name: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a font from cache or disk."""
    cache_key = (font_name, size)
    if cache_key in font_cache:
        return font_cache[cache_key]

    font_path = FONTS_DIR / font_name
    if not font_path.exists():
        raise FileNotFoundError(f"Font not found: {font_path}")

    font = ImageFont.truetype(str(font_path), size)
    font_cache[cache_key] = font
    return font


def get_character_name(codepoint: int) -> str:
    """Get the Unicode name of a character."""
    try:
        return unicodedata.name(chr(codepoint))
    except ValueError:
        return f"U+{codepoint:04X}"


def is_printable_char(codepoint: int) -> bool:
    """Check if a codepoint represents a printable character."""
    char = chr(codepoint)
    cat = unicodedata.category(char)
    # Skip control characters, unassigned, surrogates, private use
    if cat.startswith('C') or cat == 'Cn':
        return False
    # Skip combining marks that need a base character
    if cat.startswith('M'):
        return False
    return True


def calculate_font_size(char: str, font_name: str, max_size: int = 48) -> int:
    """Calculate optimal font size to fit character in image."""
    for size in range(max_size, 8, -2):
        try:
            font = get_font(font_name, size)
            bbox = font.getbbox(char)
            if bbox:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                # Leave 8 pixels margin on each side
                if width <= IMAGE_SIZE - 16 and height <= IMAGE_SIZE - 16:
                    return size
        except Exception:
            continue
    return 12  # Minimum size


def render_character(char: str, font_name: str, output_path: Path) -> bool:
    """Render a single character to a PNG file."""
    try:
        # Create image
        img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)

        # Calculate optimal font size
        font_size = calculate_font_size(char, font_name)
        font = get_font(font_name, font_size)

        # Get bounding box
        bbox = font.getbbox(char)
        if not bbox:
            return False

        # Calculate position to center the character
        char_width = bbox[2] - bbox[0]
        char_height = bbox[3] - bbox[1]
        x = (IMAGE_SIZE - char_width) // 2 - bbox[0]
        y = (IMAGE_SIZE - char_height) // 2 - bbox[1]

        # Draw character
        draw.text((x, y), char, font=font, fill=TEXT_COLOR)

        # Save image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, 'PNG')
        return True

    except Exception as e:
        print(f"    Error rendering {char!r}: {e}")
        return False


def generate_script(script_info: dict, metadata: dict) -> int:
    """Generate images for all characters in a script."""
    script_name = script_info["name"]
    font_name = script_info["font"]
    ranges = script_info["ranges"]
    common_limit = script_info.get("common_chars")

    script_dir = OUTPUT_DIR / script_name
    script_metadata = {}
    count = 0
    total_chars = 0

    # Collect all codepoints
    codepoints = []
    for start, end in ranges:
        for cp in range(start, end + 1):
            if is_printable_char(cp):
                codepoints.append(cp)

    # Apply common character limit if specified
    if common_limit and len(codepoints) > common_limit:
        codepoints = codepoints[:common_limit]

    print(f"  Processing {len(codepoints)} characters...")

    for cp in codepoints:
        char = chr(cp)
        hex_code = f"{cp:04X}"
        output_path = script_dir / f"{hex_code}.png"

        if render_character(char, font_name, output_path):
            script_metadata[hex_code] = {
                "char": char,
                "name": get_character_name(cp)
            }
            count += 1

        total_chars += 1
        if total_chars % 500 == 0:
            print(f"    Progress: {total_chars}/{len(codepoints)}")

    metadata[script_name] = script_metadata
    return count


def main():
    """Main entry point."""
    print("Unicode Character Dataset Generator")
    print("=" * 40)

    # Create directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    FONTS_DIR.mkdir(exist_ok=True)

    # Download all fonts first
    print("\nPhase 1: Downloading fonts...")
    fonts_to_download = {s["font"]: s for s in SCRIPTS if s.get("url")}
    for font_name, script_info in fonts_to_download.items():
        print(f"\nFont: {font_name}")
        download_font(script_info)

    # Generate images
    print("\n" + "=" * 40)
    print("Phase 2: Generating character images...")

    metadata = {}
    total_count = 0

    for script_info in SCRIPTS:
        script_name = script_info["name"]
        font_path = FONTS_DIR / script_info["font"]

        print(f"\nScript: {script_name}")

        if not font_path.exists():
            print(f"  Skipping - font not available: {script_info['font']}")
            continue

        count = generate_script(script_info, metadata)
        total_count += count
        print(f"  Generated: {count} images")

    # Save metadata
    print("\n" + "=" * 40)
    print("Phase 3: Saving metadata...")

    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nMetadata saved to: {metadata_path}")

    # Summary
    print("\n" + "=" * 40)
    print("Summary")
    print("=" * 40)
    print(f"Total scripts processed: {len(metadata)}")
    print(f"Total images generated: {total_count}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")

    # List scripts with counts
    print("\nImages per script:")
    for script_name, chars in sorted(metadata.items()):
        print(f"  {script_name}: {len(chars)}")


if __name__ == "__main__":
    main()
