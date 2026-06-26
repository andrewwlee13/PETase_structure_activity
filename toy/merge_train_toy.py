import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

def protein_id_from_npz(npz_path):
    return npz_path.stem.replace("_toy", "")

def numeric_id(protein_id):
    return int(protein_id.split("_")[1])

def load_xy(labels_csv, emb_dir, npz_glob="seo_*_toy.npz"):
    labels = pd.read_csv(labels_csv)
    labels["Library ID"] = labels["Library ID"].astype(str)
    labels["_sort_key"] = labels["Library ID"].map(numeric_id)
    labels = labels.sort_values("_sort_key").drop(columns="_sort_key").reset_index(drop=True)

    ids, X_list = [], []
    for npz_path in sorted(Path(emb_dir).glob(npz_glob), key=lambda p: numeric_id(protein_id_from_npz(p))):
        protein_id = protein_id_from_npz(npz_path)
        vec = np.load(npz_path)["A_embeddings"].mean(axis=0)
        ids.append(protein_id)
        X_list.append(vec)

    X = np.stack(X_list)
    ids = np.array(ids)
    meta = labels.set_index("Library ID").loc[ids].reset_index()
    y = meta["Activity"].values
    return X, y, meta

def main():
    X, y, meta = load_xy("seo_tested_seq_sorted_toy.csv", "embeddings")
    print("X:", X.shape)
    print("y:", y.shape)
    print(meta[["Library ID", "Activity"]].to_string(index=False))

    model = Ridge()
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print("CV R2 scores:", scores)
    print("Mean:", scores.mean())

if __name__ == "__main__":
    main()
