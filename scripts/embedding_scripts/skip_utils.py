"""Shared logic for deciding whether a PDB was already embedded."""

from __future__ import annotations

from pathlib import Path


def embedding_output_stem(pdb_stem: str, embedder_tag: str) -> str:
    """Build the output filename stem, e.g. seo_11_rep1_model_0_esm2."""
    return f"{pdb_stem}_{embedder_tag}"


def embedding_output_path(
    pdb_path: Path,
    output_dir: Path,
    save_format: str,
    embedder_tag: str,
) -> Path:
    """Return the tagged output file path for one PDB."""
    return output_dir / f"{embedding_output_stem(pdb_path.stem, embedder_tag)}.{save_format}"


def legacy_embedding_output_path(
    pdb_path: Path,
    output_dir: Path,
    save_format: str,
) -> Path:
    """Older runs used untagged names, e.g. seo_11_rep1_model_0.npz."""
    return output_dir / f"{pdb_path.stem}.{save_format}"


def should_skip_embedding(
    pdb_path: Path,
    output_dir: Path,
    save_format: str,
    embedder_tag: str,
    *,
    force: bool = False,
) -> bool:
    """
    Return True if embedding for this PDB can be skipped.

    Checks, in order:
      1. output_dir / <pdb_stem>_<embedder_tag>.<save_format>  (current naming)
      2. output_dir / <pdb_stem>.<save_format>                  (legacy naming)
    """
    if force:
        return False
    if embedding_output_path(pdb_path, output_dir, save_format, embedder_tag).is_file():
        return True
    return legacy_embedding_output_path(pdb_path, output_dir, save_format).is_file()
