#!/usr/bin/env python
"""
Extract ESM-2 embeddings for PINDER structures.

This script extracts per-residue embeddings from ESM-2 model
for use as sequence-based features in downstream tasks.

Usage:
    python extract_esm2_embeddings.py --pdb_dir ../Data/[name]_data/[name]_final_pdbs \
                                      --output_dir /embeddings_npz/[name]_embeddings/ \
                                      --model esm2_t33_650M_UR50D
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from tqdm import tqdm

from skip_utils import embedding_output_path, should_skip_embedding


def parse_pdb_sequence(pdb_path: str) -> Tuple[str, List[int], str]:
    """
    Extract sequence from PDB file.
    
    Returns:
        sequence: amino acid sequence string
        res_ids: list of residue IDs
        chain_id: chain identifier
    """
    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
    
    sequences = {}  # chain -> [(res_id, aa)]
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                res_name = line[17:20].strip()
                res_id = int(line[22:26].strip())
                chain = line[21]
                
                if chain not in sequences:
                    sequences[chain] = []
                
                # Skip if we've already seen this residue
                if sequences[chain] and sequences[chain][-1][0] == res_id:
                    continue
                
                aa = aa_map.get(res_name, 'X')
                sequences[chain].append((res_id, aa))
    
    # Return the first/longest chain
    if not sequences:
        return '', [], ''
    
    chain_id = max(sequences.keys(), key=lambda c: len(sequences[c]))
    res_data = sequences[chain_id]
    
    res_ids = [r[0] for r in res_data]
    sequence = ''.join([r[1] for r in res_data])
    
    return sequence, res_ids, chain_id


def extract_all_chains(pdb_path: str) -> Dict[str, Tuple[str, List[int]]]:
    """
    Extract sequences from all chains in PDB file.
    
    Returns:
        dict mapping chain_id -> (sequence, res_ids)
    """
    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
    
    sequences = {}  # chain -> [(res_id, aa)]
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                res_name = line[17:20].strip()
                res_id = int(line[22:26].strip())
                chain = line[21]
                
                if chain not in sequences:
                    sequences[chain] = []
                
                # Skip if we've already seen this residue
                if sequences[chain] and sequences[chain][-1][0] == res_id:
                    continue
                
                aa = aa_map.get(res_name, 'X')
                sequences[chain].append((res_id, aa))
    
    result = {}
    for chain_id, res_data in sequences.items():
        res_ids = [r[0] for r in res_data]
        sequence = ''.join([r[1] for r in res_data])
        result[chain_id] = (sequence, res_ids)
    
    return result


def load_esm_model(model_name: str = "esm2_t33_650M_UR50D", device: str = "cpu"):
    """Load ESM-2 model and alphabet."""
    import esm
    
    print(f"Loading ESM model: {model_name}")
    
    # Available models (smaller to larger):
    # esm2_t6_8M_UR50D - 8M params
    # esm2_t12_35M_UR50D - 35M params  
    # esm2_t30_150M_UR50D - 150M params
    # esm2_t33_650M_UR50D - 650M params (good balance)
    # esm2_t36_3B_UR50D - 3B params (needs more memory)
    
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    model.eval()
    
    batch_converter = alphabet.get_batch_converter()
    
    return model, alphabet, batch_converter


def extract_embeddings(
    model, 
    batch_converter, 
    sequences: List[Tuple[str, str]],  # List of (name, sequence)
    device: str = "cpu",
    repr_layer: int = 33,  # Last layer for esm2_t33
) -> Dict[str, np.ndarray]:
    """
    Extract ESM-2 embeddings for a batch of sequences.
    
    Returns:
        dict mapping name -> embeddings array of shape (L, embedding_dim)
    """
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
    
    token_representations = results["representations"][repr_layer]
    
    embeddings = {}
    for i, (name, seq) in enumerate(sequences):
        # Remove BOS and EOS tokens
        emb = token_representations[i, 1:len(seq)+1].cpu().numpy()
        embeddings[name] = emb
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Extract ESM-2 embeddings")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory with PDB files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for embeddings")
    parser.add_argument("--model", type=str, default="esm2_t33_650M_UR50D", 
                        help="ESM model name (default: esm2_t33_650M_UR50D)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--pattern", type=str, default="*.pdb", help="File pattern to match")
    parser.add_argument("--save_format", type=str, default="npz", choices=["npz", "pt"],
                        help="Output format (npz or pt)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed even if output file already exists",
    )
    parser.add_argument(
        "--embedder_tag",
        type=str,
        default="esm2",
        help="Tag included in output filenames (e.g. seo_11_esm2.npz)",
    )
    args = parser.parse_args()
    
    pdb_dir = Path(args.pdb_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Determine representation layer based on model
    repr_layers = {
        "esm2_t6_8M_UR50D": 6,
        "esm2_t12_35M_UR50D": 12,
        "esm2_t30_150M_UR50D": 30,
        "esm2_t33_650M_UR50D": 33,
        "esm2_t36_3B_UR50D": 36,
    }
    repr_layer = repr_layers.get(args.model, 33)
    
    # Find PDB files
    pdb_files = sorted(pdb_dir.glob(args.pattern))
    print(f"Found {len(pdb_files)} PDB files")
    
    # Load model
    model, alphabet, batch_converter = load_esm_model(args.model, args.device)
    print(f"Model loaded. Embedding dim: {model.embed_dim}")
    
    # Process files
    failed = []
    processed = 0
    skipped = 0
    
    # Collect sequences in batches
    batch_data = []  # [(pdb_name, chain_id, sequence, res_ids), ...]
    
    print("Collecting sequences from PDB files...")
    for pdb_path in tqdm(pdb_files, desc="Reading PDBs"):
        pdb_name = pdb_path.stem
        
        # Skip only if this PDB's exact output file already exists
        if should_skip_embedding(
            pdb_path,
            output_dir,
            args.save_format,
            args.embedder_tag,
            force=args.force,
        ):
            skipped += 1
            continue
        
        try:
            chains = extract_all_chains(str(pdb_path))
            for chain_id, (sequence, res_ids) in chains.items():
                if len(sequence) == 0:
                    continue
                if len(sequence) > args.max_length:
                    print(f"Skipping {pdb_name} chain {chain_id}: sequence too long ({len(sequence)} > {args.max_length})")
                    continue
                if 'X' in sequence:
                    # Replace unknown residues with mask token
                    sequence = sequence.replace('X', '<mask>')
                
                batch_data.append((pdb_name, chain_id, sequence, res_ids))
        except Exception as e:
            failed.append((str(pdb_path), str(e)))
    
    print(f"Collected {len(batch_data)} sequences from {len(pdb_files) - skipped} PDBs")
    print(f"Skipped {skipped} already processed files")
    
    # Process in batches
    print(f"\nExtracting embeddings (batch_size={args.batch_size})...")
    
    # Group by PDB file for saving
    pdb_embeddings = {}  # pdb_name -> {chain_id: {'embeddings': ..., 'res_ids': ...}}
    
    for i in tqdm(range(0, len(batch_data), args.batch_size), desc="Extracting"):
        batch = batch_data[i:i + args.batch_size]
        
        # Prepare batch for ESM
        sequences = [(f"{d[0]}_{d[1]}", d[2]) for d in batch]
        
        try:
            embeddings = extract_embeddings(
                model, batch_converter, sequences, 
                device=args.device, repr_layer=repr_layer
            )
            
            # Store results
            for (pdb_name, chain_id, seq, res_ids), (name, _) in zip(batch, sequences):
                if pdb_name not in pdb_embeddings:
                    pdb_embeddings[pdb_name] = {}
                
                pdb_embeddings[pdb_name][chain_id] = {
                    'embeddings': embeddings[name],
                    'sequence': seq,
                    'res_ids': res_ids,
                }
        except Exception as e:
            for d in batch:
                failed.append((d[0], str(e)))
    
    # Save embeddings per PDB
    print(f"\nSaving embeddings...")
    for pdb_name, chains_data in tqdm(pdb_embeddings.items(), desc="Saving"):
        out_file = embedding_output_path(
            Path(f"{pdb_name}.pdb"),
            output_dir,
            args.save_format,
            args.embedder_tag,
        )

        if args.save_format == "npz":
            # Save as numpy archive
            save_dict = {}
            for chain_id, data in chains_data.items():
                save_dict[f"{chain_id}_embeddings"] = data['embeddings']
                save_dict[f"{chain_id}_res_ids"] = np.array(data['res_ids'])
                save_dict[f"{chain_id}_sequence"] = np.array(list(data['sequence']))
            np.savez_compressed(out_file, **save_dict)
        else:
            # Save as PyTorch
            save_dict = {}
            for chain_id, data in chains_data.items():
                save_dict[chain_id] = {
                    'embeddings': torch.from_numpy(data['embeddings']),
                    'res_ids': data['res_ids'],
                    'sequence': data['sequence'],
                }
            torch.save(save_dict, out_file)
        
        processed += 1
    
    print(f"\n{'='*50}")
    print(f"ESM-2 Embedding Extraction Complete")
    print(f"{'='*50}")
    print(f"Processed: {processed} PDB files")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed: {len(failed)}")
    print(f"Output directory: {output_dir}")
    print(f"Embedding dimension: {model.embed_dim}")
    
    if failed:
        failed_file = output_dir / "failed.json"
        with open(failed_file, 'w') as f:
            json.dump(failed, f, indent=2)
        print(f"Failed files logged to: {failed_file}")


if __name__ == "__main__":
    main()
