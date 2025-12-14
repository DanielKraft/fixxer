#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIXXER Security
Cryptographic integrity verification and audit trail management.

Professional-grade file operations with SHA256 hash verification.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any, Callable


# ==============================================================================
# HASH CALCULATION & VERIFICATION
# ==============================================================================

def calculate_sha256(file_path: Path, log_callback: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """
    Calculate SHA256 hash of a file.

    Args:
        file_path: Path to file to hash
        log_callback: Optional logging function

    Returns:
        SHA256 hash as hex string, or None on error
    """
    try:
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    except Exception as e:
        if log_callback:
            log_callback(f"   [red]✗[/red] Hash calculation failed: {e}")
        return None


# ==============================================================================
# SIDECAR FILE MANAGEMENT
# ==============================================================================

def read_existing_sidecar(file_path: Path, log_callback: Optional[Callable[[str], None]] = None) -> Optional[Dict[str, Any]]:
    """
    Read existing sidecar file if it exists at the source location.

    Args:
        file_path: Path to the image file (sidecar will be <file_path>.fixxer.json)
        log_callback: Optional logging function

    Returns:
        Dictionary with sidecar data if found, None otherwise
    """
    try:
        sidecar_path = file_path.parent / f"{file_path.name}.fixxer.json"

        if not sidecar_path.exists():
            return None

        with open(sidecar_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data

    except Exception as e:
        if log_callback:
            log_callback(f"   [yellow]⚠️[/yellow] Could not read existing sidecar: {e}")
        return None


def write_sidecar_file(
    destination_path: Path,
    original_path: Path,
    source_hash: str,
    verified: bool,
    dest_hash: Optional[str] = None,
    existing_history: Optional[List[Dict[str, str]]] = None,
    log_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Write a JSON sidecar file with hash verification metadata and move history.

    Sidecar filename: <original_filename>.fixxer.json
    Example: photo.jpg -> photo.jpg.fixxer.json

    Args:
        destination_path: Where the file ended up
        original_path: Where the file came from
        source_hash: SHA256 of source
        verified: True if hashes matched
        dest_hash: Optional SHA256 of destination (for mismatch debugging)
        existing_history: List of previous moves (from old sidecar)
        log_callback: Optional logging function

    Returns:
        True on success, False on error
    """
    try:
        sidecar_path = destination_path.parent / f"{destination_path.name}.fixxer.json"

        # Build move history by appending current move to existing history
        move_history = existing_history if existing_history else []

        current_move = {
            "timestamp": datetime.now().isoformat(),
            "from": str(original_path),
            "to": str(destination_path),
            "operation": "file_move"
        }
        move_history.append(current_move)

        metadata = {
            "fixxer_version": "1.0",
            "filename": destination_path.name,
            "sha256_source": source_hash,
            "verified": verified,
            "move_history": move_history
        }

        # Add destination hash if provided (for mismatch cases)
        if dest_hash and dest_hash != source_hash:
            metadata["sha256_destination"] = dest_hash
            metadata["corruption_detected"] = True

        with open(sidecar_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        return True

    except Exception as e:
        if log_callback:
            log_callback(f"   [yellow]⚠️[/yellow] Sidecar write error: {e}")
        return False


# ==============================================================================
# VERIFIED FILE OPERATIONS
# ==============================================================================

def verify_file_move_with_hash(
    source_path: Path,
    destination_path: Path,
    log_callback: Optional[Callable[[str], None]] = None,
    generate_sidecar: bool = True
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Move file with SHA256 integrity verification.

    Workflow:
    1. Calculate hash of source file
    2. Move file to destination
    3. Calculate hash of destination file
    4. Compare hashes
    5. Optionally generate JSON sidecar
    6. Return success status

    Args:
        source_path: Source file path
        destination_path: Destination file path
        log_callback: Optional logging function
        generate_sidecar: If True, create .fixxer.json sidecar file

    Returns:
        Tuple of (success: bool, source_hash: str, dest_hash: str)
        On failure, raises RuntimeError to halt workflow
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)

    try:
        # Step 0: Read existing sidecar (if any) before moving
        existing_sidecar = read_existing_sidecar(source_path, log_callback)
        existing_history = []
        if existing_sidecar and 'move_history' in existing_sidecar:
            existing_history = existing_sidecar['move_history']

        # Step 1: Calculate source hash
        log(f"   → Computing integrity hash...")
        source_hash = calculate_sha256(source_path, log_callback)

        if not source_hash:
            error_msg = f"Failed to calculate source hash for {source_path.name}"
            log(f"   [red]✗[/red] {error_msg}")
            raise RuntimeError(error_msg)

        # Show shortened hash in logs
        short_hash = f"{source_hash[:16]}..."
        log(f"   → SHA256: [cyan]{short_hash}[/cyan]")

        # Step 2: Move the file
        log(f"   → Moving to {destination_path.parent.name}/")
        shutil.move(str(source_path), str(destination_path))

        # Step 3: Calculate destination hash
        log(f"   → Verifying integrity...")
        dest_hash = calculate_sha256(destination_path, log_callback)

        if not dest_hash:
            error_msg = f"Failed to calculate destination hash for {destination_path.name}"
            log(f"   [red]✗[/red] {error_msg}")
            raise RuntimeError(error_msg)

        # Step 4: Compare hashes
        if source_hash == dest_hash:
            log(f"   [green]✓[/green] Hash verified: MATCH")

            # Step 5: Generate sidecar if requested
            if generate_sidecar:
                sidecar_success = write_sidecar_file(
                    destination_path,
                    source_path,
                    source_hash,
                    verified=True,
                    existing_history=existing_history,
                    log_callback=log_callback
                )
                if not sidecar_success:
                    log(f"   [yellow]⚠️[/yellow] Sidecar write failed (non-critical)")

                # Step 6: Clean up old sidecar at source location (if it existed)
                if existing_sidecar:
                    old_sidecar_path = source_path.parent / f"{source_path.name}.fixxer.json"
                    try:
                        if old_sidecar_path.exists():
                            old_sidecar_path.unlink()
                    except Exception as e:
                        # Non-critical - just log the warning
                        log(f"   [yellow]⚠️[/yellow] Could not remove old sidecar (non-critical): {e}")

            return True, source_hash, dest_hash

        else:
            # CRITICAL: Hash mismatch detected - HALT WORKFLOW
            log(f"   [red]✗ CRITICAL: Hash verified: MISMATCH[/red]")
            log(f"   [red]✗ Source:      {source_hash[:32]}...[/red]")
            log(f"   [red]✗ Destination: {dest_hash[:32]}...[/red]")
            log(f"   [red]✗ FILE CORRUPTION DETECTED[/red]")

            # Write sidecar with corruption flag
            if generate_sidecar:
                write_sidecar_file(
                    destination_path,
                    source_path,
                    source_hash,
                    verified=False,
                    dest_hash=dest_hash,
                    existing_history=existing_history,
                    log_callback=log_callback
                )

            # HALT: Raise exception to stop workflow
            error_msg = f"Hash mismatch detected for {destination_path.name} - workflow halted for safety"
            raise RuntimeError(error_msg)

    except RuntimeError:
        # Re-raise RuntimeError (our controlled halt)
        raise
    except Exception as e:
        error_msg = f"File move failed: {e}"
        log(f"   [red]✗[/red] {error_msg}")
        raise RuntimeError(error_msg)
