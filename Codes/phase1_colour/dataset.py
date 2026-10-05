"""Dataset indexing for the AID multi-label aerial image dataset.

Lays out:
    images/images_tr/<ClassName>/<name>.jpg   -> gallery
    images/images_test/<ClassName>/<name>.jpg -> query
    multilabel.csv                            -> 17 binary semantic labels per image name
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GALLERY_DIR = ROOT / "images" / "images_tr"
QUERY_DIR = ROOT / "images" / "images_test"
LABELS_CSV = ROOT / "multilabel.csv"


def list_images(split_dir: Path) -> pd.DataFrame:
    """Return a DataFrame with one row per image: id, path, scene_class."""
    rows = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for img_path in sorted(class_dir.glob("*.jpg")):
            rows.append({
                "id": img_path.stem,
                "path": str(img_path),
                "scene_class": class_dir.name,
            })
    return pd.DataFrame(rows)


def load_labels() -> pd.DataFrame:
    """Return the 17-column binary semantic label table, indexed by image id."""
    df = pd.read_csv(LABELS_CSV)
    df = df.rename(columns={df.columns[0]: "id"})
    df = df.set_index("id")
    return df


def label_columns() -> list:
    return list(load_labels().columns)


def gallery_index() -> pd.DataFrame:
    df = list_images(GALLERY_DIR)
    labels = load_labels()
    return df.join(labels, on="id")


def query_index() -> pd.DataFrame:
    df = list_images(QUERY_DIR)
    labels = load_labels()
    return df.join(labels, on="id")


if __name__ == "__main__":
    g = gallery_index()
    q = query_index()
    print(f"Gallery: {len(g)} images across {g['scene_class'].nunique()} classes")
    print(f"Query:   {len(q)} images across {q['scene_class'].nunique()} classes")
    print(f"Missing labels in gallery: {g[label_columns()].isna().any(axis=1).sum()}")
    print(f"Missing labels in query:   {q[label_columns()].isna().any(axis=1).sum()}")
    print(g.head())
