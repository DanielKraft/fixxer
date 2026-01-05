# CHANGELOG

All notable changes to FIXXER PRO will be documented in this file.

## [v1.2.0] - Multi-Provider Vision API Support

### Added
- **OpenAI-compatible API provider adapter** - Support for llama.cpp, vLLM, LocalAI, Jan, and any OpenAI-compatible vision API (#8)
- **Provider abstraction layer** - Clean architecture with `VisionProvider` base class
- **OllamaProvider** - Default provider (backward compatible, zero config changes required)
- **OpenAICompatProvider** - Handles OpenAI `/v1/chat/completions` format with automatic payload translation
- **Environment variable configuration** - `FIXXER_API_PROVIDER` and `FIXXER_API_ENDPOINT` for runtime provider selection
- **Config file API section** - `[api]` section in `~/.fixxer.conf` for provider settings
- **Configuration priority system** - Environment Variables > Config File > Defaults

### Changed
- Vision API calls now use provider abstraction instead of direct Ollama integration
- Connection checking is now provider-agnostic
- Default behavior unchanged - Ollama remains the zero-config default

### Fixed
- Improved error handling for vision API failures across different providers

### Technical Details
- New `vision_providers/` module with factory pattern for provider instantiation
- Payload translation from Ollama format to OpenAI multi-modal content structure
- Graceful fallback for connection checks (tries `/v1/models` then base URL)
- Maintains full backward compatibility with existing Ollama workflows

---

## [v1.1.0] - Modular Architecture Refactor

### Added
- **Modular architecture** - Split monolithic engine.py into focused modules:
  - `config.py` - Centralized configuration and settings management
  - `security.py` - SHA256 hash verification and sidecar file operations
  - `vision.py` - AI/Ollama integration and RAW file processing
  - `engine.py` - Workflow orchestration (now ~800 lines shorter)
- **Case-insensitive file extension matching** - Finds both `.jpg` and `.JPG` files
- **Improved maintainability** - Clean separation of concerns and dependencies

### Fixed
- Removed deprecated `check_dcraw()` function (replaced by rawpy)
- Removed dead code (MockSessionTracker class)
- Fixed config file logging to show `.fixxer.conf` instead of `.photosort.conf`
- Fixed BytesIO import for burst stacking with RAW files
- Fixed Easy Button (Simple Sort) workflow execution

### Changed
- Version bumped to v1.1.0 across all components
- Cleaned up version history from source files (moved to CHANGELOG.md)

---

## [v1.0] - Professional Grade Release

### Added
- **SHA256 hash verification** for ALL file move operations
- **JSON sidecar files** for audit trail (`.fixxer.json`)
- **Halt-on-mismatch integrity protection**
- **"CHAOS PATCHED // LOGIC INJECTED"** - Professional-grade reliability

### Security
- Cryptographic verification prevents silent data corruption
- Immutable audit trail for every file operation
- Integrity checks halt workflow on any hash mismatch

---

## [v10.8] - Cross-Platform Migration

### Changed
- **REMOVED macOS-only `sips` dependency completely**
- `convert_raw_to_jpeg()` now uses Pillow for PPM→JPEG conversion
- **100% cross-platform** (Linux, macOS, Windows with rawpy)
- **Zero temp files created** (pure in-memory operation via BytesIO)
- **5x smaller output files** (689KB vs 3.7MB from sips)

### Improved
- RAW conversion performance
- Memory efficiency with in-memory processing
- Cross-platform compatibility

---

## [v10.7] - RAW Processing Fixes

### Fixed
- `convert_raw_to_jpeg()` tries embedded thumbnail first (`-e` flag)
- Falls back to full demosaic only if embedded thumbnail fails
- Added timeouts to prevent hanging on problematic RAW files

### Improved
- RAW file processing speed (embedded thumbnails are instant)
- Reliability with corrupted or unusual RAW formats

---

## [v10.0] - HUD Support & Stats Tracking

### Added
- `StatsTracker` class for real-time workflow progress
- Thread-safe callback communication between engine and TUI
- Live statistics updates (bursts, tiers, heroes, archived counts)

---

## [v7.1] - AI Vision & Phrase Library

### Added
- Phrase library with 200+ motivational/humorous progress messages
- Warez/demoscene aesthetic messaging
- Anti-repetition logic for phrase rotation

---

## Earlier Versions

Pre-v7.1 history available in git commit logs.
