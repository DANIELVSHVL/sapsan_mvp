import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


FLAG_COLUMNS = [
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
]


def _load_seatmaps(seatmaps_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(seatmaps_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"В каталоге {seatmaps_dir} нет seatmap CSV.")

    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    return df


def _check_service_classes(df_seats: pd.DataFrame, classes_path: Path) -> None:
    df_classes = pd.read_csv(classes_path)
    if "service_subclass" not in df_classes.columns:
        raise ValueError("В classes_dict.csv нет колонки 'service_subclass'.")

    subclasses_in_dict = set(df_classes["service_subclass"].dropna().unique())

    if "service_class" not in df_seats.columns:
        raise ValueError("В seatmaps нет колонки 'service_class' (ожидалось как service_subclass).")

    subclasses_in_seats = set(df_seats["service_class"].dropna().unique())
    missing = subclasses_in_seats - subclasses_in_dict
    if missing:
        raise ValueError(
            f"Не все коды service_class (по смыслу service_subclass) есть в classes_dict.csv: {sorted(missing)}"
        )


def _check_coach_types(df_seats: pd.DataFrame, coach_types_path: Path) -> None:
    df_coach = pd.read_csv(coach_types_path)
    required_cols = {"train_type", "coach_no"}
    miss = required_cols - set(df_coach.columns)
    if miss:
        raise ValueError(f"В coach_types.csv нет колонок: {sorted(miss)}")

    # Нормализуем типы coach_no
    df_seats["coach_no"] = df_seats["coach_no"].astype(int)
    df_coach["coach_no"] = df_coach["coach_no"].astype(int)

    # Сколько раз каждая пара (train_type, coach_no) встречается в coach_types
    counts = df_coach.groupby(["train_type", "coach_no"]).size()

    missing_pairs: List[Tuple[str, int]] = []
    multi_pairs: List[Tuple[str, int, int]] = []

    pairs_in_seats = (
        df_seats[["train_type", "coach_no"]]
        .drop_duplicates()
        .to_records(index=False)
    )

    for train_type, coach_no in pairs_in_seats:
        key = (train_type, coach_no)
        if key not in counts.index:
            missing_pairs.append((train_type, int(coach_no)))
        else:
            cnt = int(counts.loc[key])
            if cnt != 1:
                multi_pairs.append((train_type, int(coach_no), cnt))

    if missing_pairs:
        raise ValueError(
            f"Для пар (train_type, coach_no) нет записи в coach_types.csv: {missing_pairs}"
        )
    if multi_pairs:
        raise ValueError(
            f"В coach_types.csv дубли по (train_type, coach_no): {multi_pairs}"
        )


def _check_duplicate_seats(df_seats: pd.DataFrame) -> None:
    """
    Нет дублей seat_no в рамках (train_type, coach_no).
    """
    if not {"train_type", "coach_no", "seat_no"}.issubset(df_seats.columns):
        raise ValueError("Для проверки дублей нужны колонки train_type, coach_no, seat_no.")

    # Нормализуем coach_no и seat_no
    df_seats["coach_no"] = df_seats["coach_no"].astype(int)
    df_seats["seat_no"] = df_seats["seat_no"].astype(int)

    dup_mask = df_seats.duplicated(subset=["train_type", "coach_no", "seat_no"], keep=False)
    if dup_mask.any():
        dups = (
            df_seats.loc[dup_mask, ["train_type", "coach_no", "seat_no"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(f"Найдены дубликаты мест (train_type, coach_no, seat_no): {dups}")


def _check_flag_columns(df_seats: pd.DataFrame) -> None:
    """
    Проверяем, что флажки содержат только допустимые значения:
    0/1, 0.0/1.0, '0'/'1', '', NaN, True/False.
    """
    allowed = {0, 1, 0.0, 1.0, "0", "1", "", True, False}

    for col in FLAG_COLUMNS:
        if col not in df_seats.columns:
            # для валидации это критично
            raise ValueError(f"В seatmaps отсутствует флажок '{col}'.")
        values = set(df_seats[col].dropna().unique())
        unexpected = {v for v in values if v not in allowed}
        if unexpected:
            raise ValueError(
                f"В колонке {col!r} найдены неожиданные значения: {sorted(unexpected)}"
            )


def validate_seatmaps(
    seatmaps_dir: Path,
    classes_path: Path,
    coach_types_path: Path,
) -> None:
    """
    Базовая валидация CSV-seatmaps и справочников.
    Бросает ValueError при критических ошибках.
    """
    if not seatmaps_dir.exists():
        raise FileNotFoundError(f"Каталог seatmaps не найден: {seatmaps_dir}")
    if not classes_path.exists():
        raise FileNotFoundError(f"classes_dict.csv не найден: {classes_path}")
    if not coach_types_path.exists():
        raise FileNotFoundError(f"coach_types.csv не найден: {coach_types_path}")

    df_seats = _load_seatmaps(seatmaps_dir)

    _check_service_classes(df_seats, classes_path)
    _check_coach_types(df_seats, coach_types_path)
    _check_duplicate_seats(df_seats)
    _check_flag_columns(df_seats)

    # На этом этапе можно добавить проверки по кардинальностям/МГН/животным и т.п.,
    # но это уже больше "бизнес-инварианты", их можно вынести в отдельный слой/fixtures.


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Валидация CSV seatmaps + reference"
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
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    classes_path = args.reference_dir / "classes_dict.csv"
    coach_types_path = args.reference_dir / "coach_types.csv"
    validate_seatmaps(
        seatmaps_dir=args.seatmaps_dir,
        classes_path=classes_path,
        coach_types_path=coach_types_path,
    )
