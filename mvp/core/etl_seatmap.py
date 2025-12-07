import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


def _load_seatmaps(seatmaps_dir: Path) -> pd.DataFrame:
    """
    Читает все CSV из каталога seatmaps и склеивает в один DataFrame.
    Ожидает, что в каждом есть train_type, coach_no, seat_no, service_class и флаги.
    """
    csv_paths = sorted(seatmaps_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"В каталоге {seatmaps_dir} не найдено ни одного CSV с seatmap.")

    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if "train_type" not in df.columns:
            # fallback: попытка вытащить train_type из имени файла, если колонка отсутствует
            # но по ТЗ train_type в колонках есть, так что это просто страховка
            name = path.stem
            for t in ("type1", "type2", "type3", "type4"):
                if t in name:
                    df["train_type"] = t
                    break
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def _normalize_flags(value) -> int:
    """
    Приводит флажки к 0/1.
    Разрешённый ввод: 0/1, 0.0/1.0, '0'/'1', True/False, ''/NaN.
    Пустое/NaN трактуем как 0.
    """
    if pd.isna(value) or value == "":
        return 0
    if value in (True, False):
        return int(bool(value))
    if value in (0, 1):
        return int(value)
    if value in (0.0, 1.0):
        return int(value)
    if isinstance(value, str) and value.strip() in ("0", "1"):
        return int(value.strip())
    raise ValueError(f"Неожиданное значение флага: {value!r}")


def _normalize_forward(value):
    """
    Для is_forward_msk_spb:
    - 0/1 → 0/1
    - пустое/NaN → None
    """
    if pd.isna(value) or value == "":
        return None
    return _normalize_flags(value)


def build_seatmaps(
    seatmaps_dir: Path,
    classes_path: Path,
    coach_types_path: Path,
    output_dir: Path,
) -> None:
    """
    Основная функция ETL:
    - читает seatmaps CSV + справочники,
    - джойнит,
    - группирует по train_type и coach_layout_id,
    - пишет seatmap_<train_type>_v001.json в output_dir.
    """
    if not seatmaps_dir.exists():
        raise FileNotFoundError(f"Каталог seatmaps не найден: {seatmaps_dir}")
    if not classes_path.exists():
        raise FileNotFoundError(f"classes_dict.csv не найден: {classes_path}")
    if not coach_types_path.exists():
        raise FileNotFoundError(f"coach_types.csv не найден: {coach_types_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Seatmaps
    df_seats = _load_seatmaps(seatmaps_dir)

    required_cols = {
        "train_type",
        "coach_no",
        "seat_no",
        "service_class",  # по смыслу service_subclass
        "is_window",
        "is_aisle",
        "is_table",
        "is_forward_msk_spb",
        "is_service",
        "is_free_seating",
        "is_small_pet",
        "is_animal",
        "is_mgn",
        "is_unaccompanied_minor",
        "is_mother_child",
        "is_rotatable",
    }
    missing = required_cols - set(df_seats.columns)
    if missing:
        raise ValueError(f"В seatmaps нет обязательных колонок: {sorted(missing)}")

    # Приводим coach_no, seat_no к int (если там строки)
    df_seats["coach_no"] = df_seats["coach_no"].astype(int)
    df_seats["seat_no"] = df_seats["seat_no"].astype(int)

    # Переименовываем service_class → service_subclass
    if "service_subclass" in df_seats.columns:
        # если уже кто-то заранее переименовал – не трогаем
        service_sub_col = "service_subclass"
    else:
        df_seats = df_seats.rename(columns={"service_class": "service_subclass"})
        service_sub_col = "service_subclass"

    # 2. classes_dict
    df_classes = pd.read_csv(classes_path)
    class_required = {"service_subclass", "service_class", "description"}
    miss_cls = class_required - set(df_classes.columns)
    if miss_cls:
        raise ValueError(f"В classes_dict.csv нет колонок: {sorted(miss_cls)}")

    # join по service_subclass → добавляем service_class и description
    df_seats = df_seats.merge(
        df_classes[["service_subclass", "service_class", "description"]],
        how="left",
        left_on=service_sub_col,
        right_on="service_subclass",
        validate="many_to_one",
        suffixes=("", "_ref"),
    )
    if df_seats["service_class"].isna().any():
        missing_codes = df_seats.loc[df_seats["service_class"].isna(), service_sub_col].unique()
        raise ValueError(
            f"Не найдены коды service_subclass в classes_dict.csv: {sorted(missing_codes)}"
        )

    # 3. coach_types
    df_coach_types = pd.read_csv(coach_types_path)
    coach_required = {"train_type", "coach_no", "coach_type_code", "coach_type_name", "coach_layout_id"}
    miss_ct = coach_required - set(df_coach_types.columns)
    if miss_ct:
        raise ValueError(f"В coach_types.csv нет колонок: {sorted(miss_ct)}")

    df_coach_types["coach_no"] = df_coach_types["coach_no"].astype(int)

    df_seats = df_seats.merge(
        df_coach_types[["train_type", "coach_no", "coach_type_code", "coach_type_name", "coach_layout_id"]],
        how="left",
        on=["train_type", "coach_no"],
        validate="many_to_one",
        suffixes=("", "_coach"),
    )
    if df_seats["coach_layout_id"].isna().any():
        bad_pairs = (
            df_seats.loc[df_seats["coach_layout_id"].isna(), ["train_type", "coach_no"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(
            f"Не найдены записи в coach_types.csv для пар train_type/coach_no: {bad_pairs}"
        )

    # 4. Группируем по train_type
    for train_type, df_type in df_seats.groupby("train_type"):
        # coaches[]
        coaches_df = (
            df_type[["coach_no", "coach_type_code", "coach_layout_id"]]
            .drop_duplicates()
            .sort_values("coach_no")
        )

        coaches: List[Dict[str, Any]] = []
        for _, row in coaches_df.iterrows():
            coaches.append(
                {
                    "coach_no": int(row["coach_no"]),
                    "coach_type_code": str(row["coach_type_code"]),
                    "coach_layout_id": str(row["coach_layout_id"]),
                }
            )

        # coach_layouts{}
        coach_layouts: Dict[str, Dict[str, Any]] = {}
        for layout_id, df_layout_all in df_type.groupby("coach_layout_id"):
            # На первом шаге считаем, что layout определяется (train_type, coach_no).
            # Если один layout_id встречается на нескольких coach_no, берём минимальный coach_no
            # как "эталонный" пример.
            example_coach_no = df_layout_all["coach_no"].min()
            df_layout = df_layout_all[df_layout_all["coach_no"] == example_coach_no].copy()

            # Сортируем по номеру места, чтобы не было хаоса
            df_layout = df_layout.sort_values("seat_no")

            seats: List[Dict[str, Any]] = []
            for _, s in df_layout.iterrows():
                # Канонический ID места в рамках layout'а.
                # Для v001 берём простую схему: <layout_id>_SN<seat_no:03d>
                seat_id_canon = f"{layout_id}_SN{int(s['seat_no']):03d}"

                seat_record = {
                    "seat_id_canon": seat_id_canon,
                    "seat_no": int(s["seat_no"]),
                    "service_subclass": str(s[service_sub_col]),
                    "service_class": str(s["service_class"]),
                    "service_description": str(s["description"]),
                    "is_window": _normalize_flags(s["is_window"]),
                    "is_aisle": _normalize_flags(s["is_aisle"]),
                    "is_table": _normalize_flags(s["is_table"]),
                    "is_forward_msk_spb": _normalize_forward(s["is_forward_msk_spb"]),
                    "is_service": _normalize_flags(s["is_service"]),
                    "is_free_seating": _normalize_flags(s["is_free_seating"]),
                    "is_small_pet": _normalize_flags(s["is_small_pet"]),
                    "is_animal": _normalize_flags(s["is_animal"]),
                    "is_mgn": _normalize_flags(s["is_mgn"]),
                    "is_unaccompanied_minor": _normalize_flags(s["is_unaccompanied_minor"]),
                    "is_mother_child": _normalize_flags(s["is_mother_child"]),
                    "is_rotatable": _normalize_flags(s["is_rotatable"]),
                }
                seats.append(seat_record)

            coach_layouts[str(layout_id)] = {"seats": seats}

        # Собираем итоговую структуру для данного train_type
        result: Dict[str, Any] = {
            "train_type_id": str(train_type),
            "coaches": coaches,
            "coach_layouts": coach_layouts,
        }

        slug = str(train_type)  # если train_type = "type1" → seatmap_type1_v001.json
        out_path = output_dir / f"seatmap_{slug}_v001.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ETL: CSV seatmaps → seatmap_<train_type>_v001.json"
    )
    parser.add_argument(
        "--in",
        "--seatmaps-dir",
        dest="seatmaps_dir",
        type=Path,
        default=Path("mvp/data/seatmaps"),
        help="Каталог с CSV seatmaps (по умолчанию mvp/data/seatmaps)",
    )
    parser.add_argument(
        "--ref",
        "--reference-dir",
        dest="reference_dir",
        type=Path,
        default=Path("mvp/data/reference"),
        help="Каталог со справочниками (classes_dict.csv, coach_types.csv)",
    )
    parser.add_argument(
        "--out",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("mvp/demo/assets/v001"),
        help="Каталог для seatmap_*.json (по умолчанию mvp/demo/assets/v001)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    classes_path = args.reference_dir / "classes_dict.csv"
    coach_types_path = args.reference_dir / "coach_types.csv"
    build_seatmaps(
        seatmaps_dir=args.seatmaps_dir,
        classes_path=classes_path,
        coach_types_path=coach_types_path,
        output_dir=args.output_dir,
    )
