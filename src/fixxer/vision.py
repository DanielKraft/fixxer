#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIXXER Vision
AI-powered image analysis using local Ollama models.

Includes RAW file support, image encoding, and creative critique functionality.
"""

from __future__ import annotations

import base64
import json
import re
import time
import threading
import requests
from pathlib import Path
from io import BytesIO
from typing import Optional, Tuple, List, Dict, Any, Callable

# Import config module (for mutable RAW_SUPPORT)
from . import config
from .config import (
    INGEST_TIMEOUT,
    CRITIQUE_TIMEOUT
)
from .vision_providers import get_provider


# ==============================================================================
# AI PROMPTS
# ==============================================================================

AI_CRITIC_PROMPT = """
You are a professional Creative Director and magazine photo editor. Your job is to provide ambitious, artistic, and creative feedback to elevate a photo from "good" to "great."

**CREATIVE TOOLBOX (Use these for your suggestion):**
* **Mood & Atmosphere:** (e.g., 'cinematic,' 'moody,' 'ethereal,' 'nostalgic,' 'dramatic')
* **Color Grading:** (e.g., 'filmic teal-orange,' 'warm vintage,' 'cool desaturation,' 'split-toning')
* **Light & Shadow:** (e.g., 'crushed blacks,' 'soft, lifted shadows,' 'localized dodging/burning,' 'a subtle vignette')
* **Texture:** (e.g., 'add fine-grain film,' 'soften the focus,' 'increase clarity')

**YOUR TASK:**
Analyze the provided image by following these steps *internally*:
1.  **Composition:** Analyze balance, guiding principles (thirds, lines), and subject placement. Rate it 1-10.
2.  **Lighting & Exposure:** Analyze quality, direction, temperature, and any blown highlights or crushed shadows.
3.  **Color & Style:** Analyze the color palette, white balance, and current post-processing style.

After your analysis, you MUST return **ONLY a single, valid JSON object**. Do not provide *any* other text, preamble, or conversation. Your response must be 100% valid JSON, formatted *exactly* like this template:

```json
{
  "composition_score": <an integer from 1 to 10>,
  "composition_critique": "<A brief, one-sentence critique of the composition.>",
  "lighting_critique": "<A brief, one-sentence critique of the lighting and exposure.>",
  "color_critique": "<A brief, one-sentence critique of the color and current style.>",
  "final_verdict": "<A one-sentence summary of what works and what doesn't.>",
  "creative_mood": "<The single, most ambitious 'Creative Mood' this photo could have, chosen from the toolbox.>",
  "creative_suggestion": "<Your single, ambitious, artistic post-processing suggestion to achieve that mood. This must be a detailed, actionable paragraph.>"
}
```
"""


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def no_op_logger(message: str) -> None:
    """A dummy logger that does nothing, for when no callback is provided."""
    pass


# ==============================================================================
# PROVIDER HELPER
# ==============================================================================

def _get_active_provider():
    """Helper to get the configured vision provider."""
    app_config = config.load_app_config()
    return get_provider(app_config)


# ==============================================================================
# RAW FILE CONVERSION
# ==============================================================================

def convert_raw_to_jpeg(raw_path: Path, log_callback: Callable[[str], None] = no_op_logger) -> Optional[bytes]:
    """
    Convert RAW file to JPEG bytes using rawpy (Python-native, cross-platform).

    Method 1: Embedded Thumbnail Extraction (Fast)
    - Extracts embedded JPEG thumbnail using rawpy
    - Typical size: 500-1000KB
    - Quality: Perfect for CLIP embeddings and BRISQUE scoring

    Method 2: Quick Demosaic Fallback (Slower)
    - Uses half-size demosaic for speed
    - Only if thumbnail extraction fails

    Benefits vs dcraw:
    - 100% Python-native (no subprocess, no system dependency)
    - Faster (C++ library called directly)
    - Supports 100+ RAW formats via libraw
    - Cross-platform (Linux, macOS, Windows)
    - No temp files (pure memory operation)

    Args:
        raw_path: Path to RAW file
        log_callback: Optional logging function

    Returns:
        JPEG file as bytes, or None on failure
    """
    if not config.RAW_SUPPORT:
        return None

    try:
        import rawpy
    except ImportError:
        log_callback(f"   [red]rawpy not available for RAW conversion[/red]")
        return None

    # Method 1: Extract embedded thumbnail (fast, works for most modern cameras)
    try:
        with rawpy.imread(str(raw_path)) as raw:
            # Try to extract embedded thumbnail
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    # Perfect! Camera provided JPEG thumbnail
                    return thumb.data
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                    # Got a bitmap thumbnail, convert to JPEG via Pillow
                    from PIL import Image
                    img = Image.frombytes('RGB', (thumb.width, thumb.height), thumb.data)
                    jpeg_buffer = BytesIO()
                    img.save(jpeg_buffer, format='JPEG', quality=95)
                    jpeg_buffer.seek(0)
                    return jpeg_buffer.read()
            except rawpy.LibRawNoThumbnailError:
                # No embedded thumbnail, fall through to Method 2
                pass
            except Exception as thumb_error:
                # Thumbnail extraction failed for some reason, fall through
                log_callback(f"   [dim]Thumbnail extraction failed for {raw_path.name}, using demosaic[/dim]")

            # Method 2: Quick demosaic fallback (half-size for speed)
            try:
                rgb = raw.postprocess(
                    use_camera_wb=True,    # Use camera white balance
                    half_size=True,        # Half resolution for speed (still plenty for AI)
                    no_auto_bright=True,   # Don't auto-brighten
                    output_bps=8           # 8-bit output
                )

                # Convert numpy array to JPEG via Pillow in-memory
                from PIL import Image
                img = Image.fromarray(rgb)
                jpeg_buffer = BytesIO()
                img.save(jpeg_buffer, format='JPEG', quality=95)
                jpeg_buffer.seek(0)
                return jpeg_buffer.read()

            except Exception as demosaic_error:
                log_callback(f"   [red]Demosaic failed for {raw_path.name}:[/red] {demosaic_error}")
                return None

    except Exception as e:
        log_callback(f"   [red]Error converting RAW file {raw_path.name}:[/red] {e}")
        return None


def encode_image(image_path: Path, log_callback: Callable[[str], None] = no_op_logger) -> Optional[str]:
    """
    Convert image to base64 string, handling RAW files.

    Args:
        image_path: Path to image file (JPEG, PNG, or RAW)
        log_callback: Optional logging function

    Returns:
        Base64-encoded string, or None on failure
    """
    try:
        # All RAW formats supported by rawpy
        raw_formats = {'.rw2', '.cr2', '.cr3', '.nef', '.arw', '.dng', '.raf', '.orf', '.pef', '.srw'}
        if image_path.suffix.lower() in raw_formats:
            jpeg_bytes = convert_raw_to_jpeg(image_path, log_callback)
            if jpeg_bytes:
                return base64.b64encode(jpeg_bytes).decode('utf-8')
            else:
                return None

        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    except Exception as e:
        log_callback(f"   [red]Error encoding {image_path.name}:[/red] {e}")
        return None


def get_image_bytes_for_analysis(image_path: Path, log_callback: Callable[[str], None] = no_op_logger) -> Optional[bytes]:
    """
    Helper to get bytes from any supported file.

    Args:
        image_path: Path to image file
        log_callback: Optional logging function

    Returns:
        Image bytes, or None on failure
    """
    ext = image_path.suffix.lower()
    # All RAW formats supported by rawpy
    raw_formats = {'.rw2', '.cr2', '.cr3', '.nef', '.arw', '.dng', '.raf', '.orf', '.pef', '.srw'}
    if ext in raw_formats:
        return convert_raw_to_jpeg(image_path, log_callback)
    elif ext in ('.jpg', '.jpeg', '.png'):
        try:
            with open(image_path, 'rb') as f:
                return f.read()
        except Exception as e:
            log_callback(f"   [red]Failed to read {image_path.name}:[/red] {e}")
            return None
    return None


# ==============================================================================
# OLLAMA CONNECTION CHECK
# ==============================================================================

def check_ollama_connection(
    log_callback: Callable[[str], None] = no_op_logger,
    all_systems_go: bool = True
) -> bool:
    """
    Check if the vision provider is running and accessible.
    Maintains legacy naming but delegates to the active provider.
    """
    app_config = config.load_app_config()
    provider_name = app_config.get("api_provider", "ollama").lower()
    provider = get_provider(app_config)

    try:
        if provider_name == "ollama":
            # Line 1: The Search
            log_callback("   [grey]Looking for llamas 🔎🦙[/grey]")
        else:
            log_callback(f"   [grey]Connecting to {provider_name} vision provider...[/grey]")

        if provider.check_connection():
            # Line 2: The Discovery
            if provider_name == "ollama":
                log_callback(f"   [green]✓ Ollama connected[/green]")
            else:
                log_callback(f"   [green]✓ {provider_name.upper()} provider connected[/green]")

            # Line 3: The Punchline + System Status
            if all_systems_go:
                status = "[green]✅ (FULL VISION)[/green]"
            else:
                status = "[yellow]⚠️  (limited features)[/yellow]"

            if provider_name == "ollama":
                log_callback(f"   [grey]¿Cómo se Llama? Se llama 'Speed' 🦙🦙💨 {status}[/grey]")
            else:
                log_callback(f"   [grey]Vision system ready {status}[/grey]")

            return True
        else:
            log_callback(f"   [red]✗ {provider_name.capitalize()} connection failed[/red]")
            if provider_name == "ollama":
                 log_callback("   Start with: ollama serve")
            return False

    except Exception as e:
        log_callback(f"   [red]✗ Connection check failed:[/red] {e}")
        return False


# ==============================================================================
# AI IMAGE ANALYSIS
# ==============================================================================

def get_ai_description(
    image_path: Path,
    model_name: str,
    log_callback: Callable[[str], None] = no_op_logger
) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Get structured filename and tags from AI.

    Args:
        image_path: Path to image file
        model_name: Ollama model to use
        log_callback: Logging function

    Returns:
        Tuple of (filename: str, tags: List[str]) or (None, None) on failure
    """
    base64_image = encode_image(image_path, log_callback)
    if not base64_image:
        return None, None

    AI_NAMING_PROMPT = """You are an expert file-naming AI.
Analyze this image and generate a concise, descriptive filename and three relevant tags.
You MUST return ONLY a single, valid JSON object, formatted *exactly* like this:
{
  "filename": "<a-concise-and-descriptive-filename>",
  "tags": ["<tag1>", "<tag2>", "<tag3>"]
}
"""

    payload = {
        "model": model_name,
        "messages": [
            { "role": "user", "content": AI_NAMING_PROMPT, "images": [base64_image] }
        ],
        "stream": False,
        "format": "json"
    }

    try:
        provider = _get_active_provider()
        json_string = provider.send_request(payload, timeout=INGEST_TIMEOUT)
        data = json.loads(json_string.strip())
        filename = data.get("filename")
        tags = data.get("tags")
        if not filename or not isinstance(tags, list):
            log_callback(f"   [yellow]Warning: Model returned valid JSON but missing keys for {image_path.name}[/yellow]")
            return None, None
        return str(filename), list(tags)

    except requests.exceptions.Timeout:
        log_callback(f"   [red]Timeout processing {image_path.name}[/red]")
        return None, None
    except json.JSONDecodeError:
        log_callback(f"   [red]Error: Model returned invalid JSON for {image_path.name}[/red]")
        return None, None
    except Exception as e:
        log_callback(f"   [red]Error processing {image_path.name}: {e}[/red]")
        return None, None


def get_ai_name_with_cache(
    img_path: Path,
    model: str,
    cache: Optional[Dict[str, Dict]],
    cache_lock: Optional[threading.Lock],
    log_callback: Callable[[str], None] = no_op_logger
) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Get AI name/tags, using cache if valid (dry-run preview feature).

    Thread-safe caching with validation:
    - Checks file modification time (mtime) to detect changes
    - Checks cache age (10 min expiry)
    - Model-aware cache keys: f"{model}:{path}"
    - Protected by threading.Lock for concurrent access

    Args:
        img_path: Image file path
        model: Ollama model name
        cache: Optional cache dict (None = always run AI)
        cache_lock: Optional threading.Lock for thread-safe cache access
        log_callback: Logging function

    Returns:
        Tuple of (filename: str, tags: List[str]) or (None, None) on failure
    """
    if cache is None:
        # No cache provided, always run AI
        return get_ai_description(img_path, model, log_callback)

    # Model-aware cache key (critical: different models = different results)
    cache_key = f"{model}:{str(img_path.absolute())}"
    current_mtime = img_path.stat().st_mtime

    # Thread-safe cache read
    cached_entry = None
    if cache_lock:
        with cache_lock:
            cached_entry = cache.get(cache_key)
    else:
        cached_entry = cache.get(cache_key)

    # Check cache validity
    if cached_entry:
        # Validate: file unchanged + cache fresh (<10 min)
        age = time.time() - cached_entry['cached_at']
        if cached_entry['mtime'] == current_mtime and age < 600:
            log_callback(f"   [dim]⚡ Using cached AI result[/dim]")
            return cached_entry['filename'], cached_entry['tags']
        else:
            # Cache invalid (file changed or expired)
            if cached_entry['mtime'] != current_mtime:
                log_callback(f"   [yellow]File changed, re-running AI[/yellow]")
            else:
                log_callback(f"   [dim]Cache expired ({age/60:.1f}m old), re-running AI[/dim]")

    # Cache miss or invalid - run AI
    log_callback(f"   [grey]🤖 Generating AI name...[/grey]")
    filename, tags = get_ai_description(img_path, model, log_callback)

    if filename and tags:
        # Thread-safe cache write
        entry = {
            'filename': filename,
            'tags': tags,
            'mtime': current_mtime,
            'cached_at': time.time()
        }
        if cache_lock:
            with cache_lock:
                cache[cache_key] = entry
        else:
            cache[cache_key] = entry

    return filename, tags


def critique_single_image(
    image_path: Path,
    model_name: str,
    log_callback: Callable[[str], None] = no_op_logger
) -> Optional[Dict[str, Any]]:
    """
    Get AI creative critique for a single image.

    Args:
        image_path: Path to image file
        model_name: Ollama model to use
        log_callback: Logging function

    Returns:
        Dictionary with critique data, or None on failure
    """
    base64_image = encode_image(image_path, log_callback)
    if not base64_image:
        log_callback(f"[red]Failed to encode image for critique[/red]")
        return None

    payload = {
        "model": model_name,
        "messages": [
            { "role": "user", "content": AI_CRITIC_PROMPT, "images": [base64_image] }
        ],
        "stream": False,
        "format": "json"
    }

    try:
        log_callback(f"   [grey]Sending to {model_name} for analysis...[/grey]")
        provider = _get_active_provider()
        json_string = provider.send_request(payload, timeout=CRITIQUE_TIMEOUT)
        json_string = json_string.strip()

        # Clean up potential markdown formatting
        if json_string.startswith("```"):
            json_string = json_string.split("```")[1]
            if json_string.startswith("json"):
                json_string = json_string[4:]
            json_string = json_string.strip()

        data = json.loads(json_string)

        # Validate expected fields
        expected_fields = [
            "composition_score", "composition_critique", "lighting_critique",
            "color_critique", "final_verdict", "creative_mood", "creative_suggestion"
        ]

        for field in expected_fields:
            if field not in data:
                log_callback(f"[yellow]Warning: Missing field '{field}' in critique response[/yellow]")

        return data

    except requests.exceptions.Timeout:
        log_callback(f"[red]Timeout waiting for critique response[/red]")
        return None
    except json.JSONDecodeError as e:
        log_callback(f"[red]Error: Model returned invalid JSON: {e}[/red]")
        return None
    except Exception as e:
        log_callback(f"[red]Error during critique: {e}[/red]")
        return None
