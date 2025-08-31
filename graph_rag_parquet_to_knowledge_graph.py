import os
import json
import ast
import numpy as np
import pandas as pd
from pandas.api.types import is_scalar

# ---- Config ----
BASE = "graph_proj/input/output"
OUT_JSON = "knowledge_graph.json"

# ---- Helpers ----
def safe_eval(x, default=None):
    """Safely evaluate string representations of Python literals (lists/dicts/tuples)."""
    if x is None:
        return default
    if isinstance(x, (list, tuple, dict)):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        if x.strip() == "" or x.strip().lower() == "nan":
            return default
        try:
            return ast.literal_eval(x)
        except Exception:
            return default
    return x

def normalize_tuids(val):
    """Normalize text_unit_ids to a Python list (or [] if missing)."""
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple, set)):
        return list(val)
    if isinstance(val, str):
        parsed = safe_eval(val, default=None)
        if parsed is None:
            # Single comma-separated string? Try a light split.
            parts = [p.strip() for p in val.split(",") if p.strip()]
            return parts if parts else []
        return normalize_tuids(parsed)
    # Fallback: wrap scalar in a list
    return [val]

def safe_get(row, column, default=None):
    """Safely get a value from a pandas row, treating only scalars as NA-checkable."""
    if column not in row.index:
        return default
    val = row[column]
    if is_scalar(val):
        return default if pd.isna(val) else val
    return val

# ---- Core ----
def load_full_graph(base_dir=BASE):
    entities_path = os.path.join(base_dir, "entities.parquet")
    rels_path = os.path.join(base_dir, "relationships.parquet")

    if not os.path.exists(entities_path):
        raise FileNotFoundError(f"Missing {entities_path}")
    if not os.path.exists(rels_path):
        raise FileNotFoundError(f"Missing {rels_path}")

    entities = pd.read_parquet(entities_path)
    relationships = pd.read_parquet(rels_path)

    # Load optional tables
    available_files = {}
    optional_files = [
        "text_units.parquet",
        "documents.parquet",
        "communities.parquet",
        "community_reports.parquet",
    ]
    for fname in optional_files:
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                available_files[fname] = df
                print(f"Loaded {fname}: {len(df)} rows")
            except Exception as e:
                print(f"Could not load {fname}: {e}")

    print("\nEntities columns:", list(entities.columns))
    print("Relationships columns:", list(relationships.columns))

    # Build node list
    nodes = []
    ent_has_tuids = "text_unit_ids" in entities.columns
    for _, row in entities.iterrows():
        node = {
            "id": str(safe_get(row, "id", "")),
            # prefer 'title' -> fallback to 'human_readable_id' -> empty
            "name": str(
                safe_get(row, "title", safe_get(row, "human_readable_id", ""))
            ),
            "type": str(safe_get(row, "type", "")),
        }

        # Optional descriptive fields
        desc = safe_get(row, "description", None)
        if desc is not None:
            node["description"] = str(desc)

        hri = safe_get(row, "human_readable_id", None)
        if hri is not None:
            node["human_readable_id"] = str(hri)

        # text_unit_ids on entities
        if ent_has_tuids:
            tuids = safe_get(row, "text_unit_ids", None)
            if tuids is not None:
                node["text_unit_ids"] = normalize_tuids(tuids)

        nodes.append(node)

    # Build edge list
    edges = []
    rel_has_weight = "weight" in relationships.columns
    rel_has_tuids = "text_unit_ids" in relationships.columns

    for _, row in relationships.iterrows():
        source = (
            safe_get(row, "source")
            or safe_get(row, "source_id")
            or safe_get(row, "src")
            or ""
        )
        target = (
            safe_get(row, "target")
            or safe_get(row, "target_id")
            or safe_get(row, "dst")
            or ""
        )
        edge = {"source": str(source), "target": str(target)}

        # relation/description
        desc = (
            safe_get(row, "description")
            or safe_get(row, "human_readable_id")
            or safe_get(row, "relation")
            or ""
        )
        if desc:
            edge["relation"] = str(desc)

        # weight
        if rel_has_weight:
            w = safe_get(row, "weight", None)
            if is_scalar(w) and w is not None and not pd.isna(w):
                try:
                    edge["weight"] = float(w)
                except Exception:
                    pass

        # text_unit_ids on relationships
        if rel_has_tuids:
            tuids = safe_get(row, "text_unit_ids", None)
            if tuids is not None:
                edge["text_unit_ids"] = normalize_tuids(tuids)

        edges.append(edge)

    result = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "total_entities": len(nodes),
            "total_relationships": len(edges),
            "available_files": list(available_files.keys()),
        },
    }

    # (Optional) Inspect community/cluster columns if present
    if "communities.parquet" in available_files:
        communities = available_files["communities.parquet"]
        print("\nCommunities columns:", list(communities.columns))

    if "community_reports.parquet" in available_files:
        reports = available_files["community_reports.parquet"]
        print("\nCommunity reports columns:", list(reports.columns))

    return result

def main():
    kg = load_full_graph(BASE)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {OUT_JSON}")
    print(f"Total nodes: {len(kg['nodes'])}")
    print(f"Total edges: {len(kg['edges'])}")

    # Show small sample
    if kg["nodes"]:
        print("\nSample node:", json.dumps(kg["nodes"][0], indent=2)[:2000])
    if kg["edges"]:
        print("\nSample edge:", json.dumps(kg["edges"][0], indent=2)[:2000])

if __name__ == "__main__":
    main()
