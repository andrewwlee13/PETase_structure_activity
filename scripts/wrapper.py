#!/usr/bin/env python
"""
Run embedding extraction scripts from a YAML config.

This is the orchestrator: it does NOT compute embeddings itself.
It reads a per-task config file, then calls the individual embedder
scripts in start_here/embedding_scripts/ as subprocesses.

Usage:
    python scripts/wrapper.py --config configs/seo.yaml
    python scripts/wrapper.py --config configs/seo.yaml --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------
# REPO_ROOT: project root (parent of start_here/). Config paths like
#   Data/Seo_data/... are resolved relative to here.
# START_HERE: this folder (start_here/), where wrapper.py lives.
# EMBEDDING_SCRIPTS: Nathan's per-model extraction scripts.

REPO_ROOT = Path(__file__).resolve().parent.parent
START_HERE = Path(__file__).resolve().parent
EMBEDDING_SCRIPTS = START_HERE / "embedding_scripts"

# Embedders we know about. Order here is the order they run in.
EMBEDDER_NAMES = ("esm2", "proteinmpnn", "gearnet", "esm_gearnet")

# Default script paths when config omits embedders.<name>.script
DEFAULT_SCRIPTS = {
    "esm2": EMBEDDING_SCRIPTS / "extract_esm2_embeddings.py",
    "proteinmpnn": EMBEDDING_SCRIPTS / "extract_proteinmpnn_embeddings.py",
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """Read and parse the YAML config file for one task/dataset."""
    with config_path.open() as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def resolve_path(path: str | Path | None, *, base: Path) -> Path | None:
    """
    Turn a config path into an absolute Path.

    Relative paths (e.g. Data/Seo_data/seo_final_pdbs) are joined to base
    (usually REPO_ROOT). Absolute paths are left unchanged.
    """
    if path is None or path == "":
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def as_bool(value: Any) -> bool:
    """Parse YAML enabled flags (true/false, yes/no, or missing → false)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() in {"true", "yes", "1"}
    return bool(value)


# ---------------------------------------------------------------------------
# PDB discovery
# ---------------------------------------------------------------------------

def discover_pdbs(pdb_dir: Path, pattern: str = "*.pdb") -> list[Path]:
    """
    List PDB files to embed. Used for a pre-flight count before running
    anything expensive. Actual embedding still happens inside each script.
    """
    if not pdb_dir.is_dir():
        raise FileNotFoundError(f"PDB directory not found: {pdb_dir}")
    return sorted(pdb_dir.glob(pattern))


# ---------------------------------------------------------------------------
# Build and run embedder subprocesses
# ---------------------------------------------------------------------------

def build_embedder_command(
    name: str,
    settings: dict[str, Any],
    *,
    pdb_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> list[str]:
    """
    Build the shell command for one embedder.

    Maps config fields (model, device, batch_size, etc.) to the CLI flags
    that each extraction script expects. Skip/resume is handled inside each
    embedder script: a PDB is skipped only when
    output_dir/<pdb_stem>_<embedder_tag>.<format> already exists.
    """
    # Use config script path if set, otherwise fall back to DEFAULT_SCRIPTS
    script = settings.get("script")
    if script:
        script_path = resolve_path(script, base=REPO_ROOT)
    else:
        script_path = DEFAULT_SCRIPTS.get(name)

    if script_path is None or not script_path.is_file():
        raise FileNotFoundError(
            f"No script configured for embedder '{name}'. "
            f"Set embedders.{name}.script in the config."
        )

    # Base args shared by all embedders
    embedder_tag = settings.get("embedder_tag") or name
    cmd = [
        sys.executable,
        str(script_path),
        "--pdb_dir",
        str(pdb_dir),
        "--output_dir",
        str(output_dir),
        "--embedder_tag",
        str(embedder_tag),
    ]

    model = settings.get("model")
    if model:
        cmd.extend(["--model", str(model)])

    device = settings.get("device")
    if device:
        cmd.extend(["--device", str(device)])

    # Embedder-specific optional flags
    if name == "esm2":
        batch_size = settings.get("batch_size")
        if batch_size is not None:
            cmd.extend(["--batch_size", str(batch_size)])
        max_length = settings.get("max_length")
        if max_length is not None:
            cmd.extend(["--max_length", str(max_length)])

    if name == "proteinmpnn" and as_bool(settings.get("simple")):
        cmd.append("--simple")  # geometric features only, no MPNN model

    pattern = settings.get("pattern")
    if pattern:
        cmd.extend(["--pattern", str(pattern)])

    save_format = settings.get("save_format")
    if save_format:
        cmd.extend(["--save_format", str(save_format)])

    if force or as_bool(settings.get("force")) or as_bool(settings.get("force_rerun")):
        cmd.append("--force")

    return cmd


def run_embedder(name: str, cmd: list[str], *, dry_run: bool) -> int:
    """
    Print and optionally execute one embedder command.

    Returns the subprocess exit code (0 = success). With --dry-run,
    only prints the command and returns 0.
    """
    print(f"\n{'=' * 60}")
    print(f"Embedder: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    if dry_run:
        return 0

    # Run from REPO_ROOT so relative imports/paths inside scripts work
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"Embedder '{name}' failed with exit code {result.returncode}")
    return result.returncode


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    # --- CLI arguments ---
    parser = argparse.ArgumentParser(description="Run embedding extraction from YAML config")
    parser.add_argument(
        "--config",
        type=Path,
        default=START_HERE / "config.yaml",
        help="Path to YAML config file (use a per-task config, e.g. configs/seo.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print embedder commands without running them",
    )
    parser.add_argument(
        "--embedders",
        nargs="+",
        choices=EMBEDDER_NAMES,
        help="Run only these embedders (must also be enabled in config)",
    )
    args = parser.parse_args()

    # --- Load config and resolve input/output paths ---
    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    config = load_config(config_path)
    paths = config.get("paths") or {}

    pdb_dir = resolve_path(paths.get("pdb_dir"), base=REPO_ROOT)
    output_root = resolve_path(paths.get("output_dir"), base=REPO_ROOT)

    if pdb_dir is None:
        print("Config error: paths.pdb_dir is required", file=sys.stderr)
        return 1
    if output_root is None:
        print("Config error: paths.output_dir is required", file=sys.stderr)
        return 1

    # --- Pre-flight: confirm PDBs exist before loading any models ---
    pattern = paths.get("pattern", "*.pdb")
    force_rerun = as_bool(paths.get("force_rerun"))
    pdb_files = discover_pdbs(pdb_dir, pattern=pattern)
    print(f"Config: {config_path}")
    print(f"PDB dir: {pdb_dir} ({len(pdb_files)} files matching {pattern})")
    print(f"Output root: {output_root}")

    if not pdb_files:
        print("No PDB files found. Check paths.pdb_dir and paths.pattern.", file=sys.stderr)
        return 1

    # --- Loop over embedders and run enabled ones ---
    # Each embedder writes to output_root/<output_subdir>/
    # (e.g. seo_esm2_embeddings/ — override via embedders.<name>.output_subdir)
    embedders = config.get("embedders") or {}
    selected = set(args.embedders) if args.embedders else None
    failures = 0
    ran_any = False

    for name in EMBEDDER_NAMES:
        settings = embedders.get(name)
        if not settings:
            continue
        if selected is not None and name not in selected:
            continue
        if not as_bool(settings.get("enabled")):
            continue

        output_subdir = settings.get("output_subdir") or name
        embedder_output = output_root / output_subdir
        embedder_output.mkdir(parents=True, exist_ok=True)

        try:
            cmd = build_embedder_command(
                name,
                settings,
                pdb_dir=pdb_dir,
                output_dir=embedder_output,
                force=force_rerun,
            )
        except FileNotFoundError as exc:
            # e.g. gearnet enabled but script not added yet
            print(f"Skipping '{name}': {exc}", file=sys.stderr)
            failures += 1
            continue

        ran_any = True
        rc = run_embedder(name, cmd, dry_run=args.dry_run)
        if rc != 0:
            failures += 1

    # --- Exit status ---
    if not ran_any:
        print(
            "No embedders ran. Enable at least one embedder in the config "
            "(embedders.<name>.enabled: true).",
            file=sys.stderr,
        )
        return 1

    if failures:
        print(f"\nFinished with {failures} embedder failure(s).")
        return 1

    print("\nAll requested embedders finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
