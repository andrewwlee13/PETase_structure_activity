#!/usr/bin/env python
"""
Extract ProteinMPNN encoder embeddings for PINDER structures.

This script extracts per-residue embeddings from ProteinMPNN's encoder
for use as structural features in downstream tasks.

Usage:
    python extract_proteinmpnn_embeddings.py --pdb_dir /mnt/d/pinder_data/pinder/2024-02/pdbs \
                                             --output_dir /mnt/d/pinder_data/pinder/2024-02/embeddings \
                                             --batch_size 1
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add ProteinMPNN to path
PROTEINMPNN_PATH = "/mnt/d/ProteinMPNN"
sys.path.insert(0, PROTEINMPNN_PATH)

import torch
from tqdm import tqdm


def parse_pdb(pdb_path: str) -> Tuple[np.ndarray, str, List[int]]:
    """
    Parse PDB file to extract CA coordinates, sequence, and residue IDs.
    
    Returns:
        coords: (L, 3) array of CA coordinates
        sequence: amino acid sequence string
        res_ids: list of residue IDs
    """
    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
    
    coords = []
    sequence = []
    res_ids = []
    seen_residues = set()
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                res_name = line[17:20].strip()
                res_id = int(line[22:26].strip())
                chain = line[21]
                
                # Skip if we've already seen this residue
                key = (chain, res_id)
                if key in seen_residues:
                    continue
                seen_residues.add(key)
                
                # Get coordinates
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
                
                # Get amino acid
                aa = aa_map.get(res_name, 'X')
                sequence.append(aa)
                res_ids.append(res_id)
    
    return np.array(coords), ''.join(sequence), res_ids


def get_backbone_coords(pdb_path: str) -> Dict:
    """
    Extract N, CA, C, O coordinates for ProteinMPNN input.
    
    Returns dict with:
        - 'coords': (L, 4, 3) array for N, CA, C, O
        - 'sequence': amino acid sequence
        - 'res_ids': residue IDs
    """
    coords_dict = {'N': [], 'CA': [], 'C': [], 'O': []}
    sequence = []
    res_ids = []
    seen_residues = {}
    
    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
                
            atom_name = line[12:16].strip()
            if atom_name not in ['N', 'CA', 'C', 'O']:
                continue
                
            res_name = line[17:20].strip()
            res_id = int(line[22:26].strip())
            chain = line[21]
            key = (chain, res_id)
            
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            
            if key not in seen_residues:
                seen_residues[key] = {
                    'res_name': res_name,
                    'coords': {}
                }
            
            seen_residues[key]['coords'][atom_name] = [x, y, z]
    
    # Build arrays in residue order
    for key in sorted(seen_residues.keys()):
        res_data = seen_residues[key]
        
        # Skip if missing backbone atoms
        if not all(atom in res_data['coords'] for atom in ['N', 'CA', 'C', 'O']):
            continue
        
        for atom in ['N', 'CA', 'C', 'O']:
            coords_dict[atom].append(res_data['coords'][atom])
        
        aa = aa_map.get(res_data['res_name'], 'X')
        sequence.append(aa)
        res_ids.append(key[1])  # residue ID
    
    # Stack into (L, 4, 3) array
    coords = np.stack([
        np.array(coords_dict['N']),
        np.array(coords_dict['CA']),
        np.array(coords_dict['C']),
        np.array(coords_dict['O']),
    ], axis=1)
    
    return {
        'coords': coords,
        'sequence': ''.join(sequence),
        'res_ids': res_ids,
    }


def load_proteinmpnn_model(model_name: str = "v_48_020", device: str = "cpu"):
    """Load ProteinMPNN model."""
    from protein_mpnn_utils import ProteinMPNN
    
    # Model paths
    model_weights = {
        "v_48_002": f"{PROTEINMPNN_PATH}/vanilla_model_weights/v_48_002.pt",
        "v_48_010": f"{PROTEINMPNN_PATH}/vanilla_model_weights/v_48_010.pt",
        "v_48_020": f"{PROTEINMPNN_PATH}/vanilla_model_weights/v_48_020.pt",
        "v_48_030": f"{PROTEINMPNN_PATH}/vanilla_model_weights/v_48_030.pt",
    }
    
    checkpoint_path = model_weights.get(model_name)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise ValueError(f"Model {model_name} not found. Available: {list(model_weights.keys())}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Model config from checkpoint
    hidden_dim = 128
    num_layers = 3
    
    model = ProteinMPNN(
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        vocab=21,
        k_neighbors=checkpoint['num_edges'],
        augment_eps=0.0,
        dropout=0.0,
        ca_only=False,  # Full backbone model
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, hidden_dim


def extract_embeddings_simple(pdb_path: str, device: str = "cpu") -> Dict:
    """
    Extract simple structural features (no ProteinMPNN required).
    
    Returns per-residue features:
    - CA coordinates (3D)
    - Local geometry features
    """
    data = get_backbone_coords(pdb_path)
    coords = data['coords']  # (L, 4, 3) - N, CA, C, O
    
    L = len(coords)
    if L == 0:
        return None
    
    ca_coords = coords[:, 1, :]  # CA coordinates
    
    # Compute local geometry features
    features = []
    
    for i in range(L):
        feat = list(ca_coords[i])  # x, y, z
        
        # Distance to neighbors
        if i > 0:
            feat.append(np.linalg.norm(ca_coords[i] - ca_coords[i-1]))
        else:
            feat.append(0.0)
            
        if i < L - 1:
            feat.append(np.linalg.norm(ca_coords[i] - ca_coords[i+1]))
        else:
            feat.append(0.0)
        
        # Local density (number of CA within 10Å)
        dists = np.linalg.norm(ca_coords - ca_coords[i], axis=1)
        feat.append(np.sum(dists < 10.0))
        
        features.append(feat)
    
    return {
        'pdb_file': os.path.basename(pdb_path),
        'sequence': data['sequence'],
        'res_ids': data['res_ids'],
        'features': np.array(features),  # (L, 6)
        'feature_names': ['x', 'y', 'z', 'dist_prev', 'dist_next', 'local_density'],
    }


def featurize_structure(pdb_path: str, device: str = "cpu") -> Dict:
    """
    Featurize a protein structure for ProteinMPNN.
    
    Returns dict with tensors ready for model input.
    """
    data = get_backbone_coords(pdb_path)
    if len(data['coords']) == 0:
        return None
    
    coords = data['coords']  # (L, 4, 3)
    L = len(coords)
    
    # Convert to tensor format expected by ProteinMPNN
    # X: (1, L, 4, 3) - backbone coordinates
    X = torch.from_numpy(coords).float().unsqueeze(0).to(device)
    
    # Mask: (1, L) - all residues valid
    mask = torch.ones(1, L).to(device)
    
    # Chain encoding: (1, L) - single chain
    chain_encoding = torch.ones(1, L).long().to(device)
    
    # Residue index: (1, L)
    residue_idx = torch.arange(L).unsqueeze(0).to(device)
    
    return {
        'X': X,
        'mask': mask,
        'chain_encoding': chain_encoding,
        'residue_idx': residue_idx,
        'sequence': data['sequence'],
        'res_ids': data['res_ids'],
    }


def gather_nodes(nodes, neighbor_idx):
    """Helper function to gather node features."""
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def extract_proteinmpnn_embeddings(model, pdb_path: str, device: str = "cpu") -> Dict:
    """
    Extract ProteinMPNN encoder embeddings for a structure.
    
    Returns per-residue embeddings from the encoder.
    """
    feats = featurize_structure(pdb_path, device)
    if feats is None:
        return None
    
    X = feats['X']
    mask = feats['mask']
    chain_encoding = feats['chain_encoding']
    residue_idx = feats['residue_idx']
    
    with torch.no_grad():
        # Get edge features from the structure
        E, E_idx = model.features(X, mask, residue_idx, chain_encoding)
        
        # Initialize node embeddings to zeros (same as in forward pass)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = model.W_e(E)
        
        # Build attention mask
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        
        # Run through encoder layers
        for layer in model.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        
        # h_V is now (1, L, hidden_dim)
        embeddings = h_V.squeeze(0).cpu().numpy()
    
    return {
        'embeddings': embeddings,
        'sequence': feats['sequence'],
        'res_ids': feats['res_ids'],
    }


def main():
    parser = argparse.ArgumentParser(description="Extract ProteinMPNN embeddings")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory with PDB files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for embeddings")
    parser.add_argument("--model", type=str, default="v_48_020", help="ProteinMPNN model version")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--simple", action="store_true", help="Use simple geometric features (no ProteinMPNN)")
    parser.add_argument("--pattern", type=str, default="*.pdb", help="File pattern to match")
    parser.add_argument("--save_format", type=str, default="npz", choices=["npz", "pt"],
                        help="Output format (npz or pt)")
    args = parser.parse_args()
    
    pdb_dir = Path(args.pdb_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find PDB files
    pdb_files = sorted(pdb_dir.glob(args.pattern))
    print(f"Found {len(pdb_files)} PDB files")
    
    if args.simple:
        print("Using simple geometric features (no ProteinMPNN)")
        
        all_embeddings = {}
        failed = []
        
        for pdb_path in tqdm(pdb_files, desc="Extracting features"):
            try:
                result = extract_embeddings_simple(str(pdb_path), args.device)
                if result is not None:
                    pdb_name = pdb_path.stem
                    all_embeddings[pdb_name] = {
                        'sequence': result['sequence'],
                        'res_ids': result['res_ids'],
                        'features': result['features'].tolist(),
                        'feature_names': result['feature_names'],
                    }
            except Exception as e:
                failed.append((str(pdb_path), str(e)))
        
        # Save embeddings
        output_file = output_dir / "structural_features.json"
        with open(output_file, 'w') as f:
            json.dump(all_embeddings, f)
        print(f"Saved {len(all_embeddings)} embeddings to {output_file}")
        
        if failed:
            print(f"Failed: {len(failed)} files")
            with open(output_dir / "failed.json", 'w') as f:
                json.dump(failed, f, indent=2)
    
    else:
        print(f"Loading ProteinMPNN model: {args.model}")
        try:
            model, hidden_dim = load_proteinmpnn_model(args.model, args.device)
            print(f"Model loaded successfully (hidden_dim={hidden_dim})")
        except Exception as e:
            print(f"Failed to load ProteinMPNN: {e}")
            print("Falling back to simple geometric features...")
            args.simple = True
            main()  # Recursive call with simple=True
            return
        
        # Extract embeddings for all files
        failed = []
        processed = 0
        skipped = 0
        
        for pdb_path in tqdm(pdb_files, desc="Extracting ProteinMPNN embeddings"):
            pdb_name = pdb_path.stem
            out_file = output_dir / f"{pdb_name}.{args.save_format}"
            
            # Skip if already processed
            if out_file.exists():
                skipped += 1
                continue
            
            try:
                result = extract_proteinmpnn_embeddings(model, str(pdb_path), args.device)
                if result is not None:
                    if args.save_format == "npz":
                        np.savez_compressed(
                            out_file,
                            embeddings=result['embeddings'],
                            res_ids=np.array(result['res_ids']),
                            sequence=np.array(list(result['sequence']))
                        )
                    else:
                        torch.save({
                            'embeddings': torch.from_numpy(result['embeddings']),
                            'res_ids': result['res_ids'],
                            'sequence': result['sequence'],
                        }, out_file)
                    processed += 1
            except Exception as e:
                failed.append((str(pdb_path), str(e)))
        
        print(f"\n{'='*50}")
        print(f"ProteinMPNN Embedding Extraction Complete")
        print(f"{'='*50}")
        print(f"Processed: {processed} PDB files")
        print(f"Skipped (already done): {skipped}")
        print(f"Failed: {len(failed)}")
        print(f"Output directory: {output_dir}")
        
        if failed:
            failed_file = output_dir / "failed.json"
            with open(failed_file, 'w') as f:
                json.dump(failed, f, indent=2)
            print(f"Failed files logged to: {failed_file}")


if __name__ == "__main__":
    main()
