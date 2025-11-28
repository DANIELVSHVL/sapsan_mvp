from pathlib import Path

def main():
    root = Path("mvp")

    dirs = [
        root / "data" / "seatmaps",
        root / "data" / "events",
        root / "data" / "reference",
        root / "core",
        root / "demo" / "assets" / "v001",
        root / "tests",
        root / "legal",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    files = {
        root/"data/reference/classes_dict.csv": "service_subclass,service_class,description\n",
        root/"data/reference/coach_types.csv": "train_type,coach_no,coach_type_code,coach_type_name,coach_layout_id\n",

        root/"core/etl_seatmap.py": '"""CSV -> seatmap JSON + mirror + adjacency."""\n',
        root/"core/validation.py": '"""Seatmap validation."""\n',

        root/"tests/test_seatmap_integrity.py": '"""Seatmap integrity tests."""\n',
        root/"legal/VERSION.md": "seatmap: v001\n",
        root/"legal/changelog.md": "- v001: initial seatmap structure\n",

        Path("README.md"): "# Sapsan MVP seatmaps\n",
        Path("requirements.txt"): "pandas\npyarrow\nnumpy\n",
        Path(".gitignore"): "__pycache__/\n.ipynb_checkpoints/\n*.parquet\n",
    }

    for p, content in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():          # чтобы не затирать вручную правленное
            p.write_text(content, encoding="utf-8")

    print("Структура mvp проверена/создана.")

if __name__ == "__main__":
    main()
