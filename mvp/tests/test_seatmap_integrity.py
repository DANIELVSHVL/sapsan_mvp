from pathlib import Path
import sys

import pandas as pd
import pytest

# Корень репозитория (/content/sapsan_mvp)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvp.core.validation import validate_seatmaps


SEATMAPS_DIR = ROOT / "mvp" / "data" / "seatmaps"
REF_DIR = ROOT / "mvp" / "data" / "reference"
CLASSES_PATH = REF_DIR / "classes_dict.csv"
COACH_TYPES_PATH = REF_DIR / "coach_types.csv"


# Фактические метрики по типам (то, что ты уже просмотрел глазами)
# Ключи такие же, как в именах файлов: sapsan_typeN_seats_v001.csv
EXPECTED = {
    "type1": {
        "total_rows": 552,
        "rows_by_coach": {1: 23, 2: 52, 3: 66, 4: 66, 5: 40, 6: 56, 7: 66, 8: 66, 9: 66, 10: 51},
        "free_seating_total": 24,
        "service_total": 10,
        "sellable_total": 518,
    },
    "type2": {
        "total_rows": 550,
        "rows_by_coach": {1: 23, 2: 48, 3: 66, 4: 66, 5: 40, 6: 56, 7: 66, 8: 66, 9: 66, 10: 53},
        "free_seating_total": 24,
        "service_total": 8,
        "sellable_total": 518,
    },
    "type3": {
        "total_rows": 546,
        "rows_by_coach": {1: 21, 2: 48, 3: 66, 4: 66, 5: 40, 6: 56, 7: 66, 8: 66, 9: 66, 10: 51},
        "free_seating_total": 24,
        "service_total": 8,
        "sellable_total": 514,
    },
    "type4": {
        "total_rows": 577,
        "rows_by_coach": {1: 48, 2: 52, 3: 66, 4: 66, 5: 40, 6: 56, 7: 66, 8: 66, 9: 66, 10: 51},
        "free_seating_total": 24,
        "service_total": 8,
        "sellable_total": 545,
    },
}


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Базовые метрики по CSV одного типа:
    - total_rows
    - rows_by_coach
    - free_seating_total
    - service_total
    - sellable_total (не service и не free_seating)
    """
    metrics = {}

    # Общее количество строк
    metrics["total_rows"] = len(df)

    # Распределение по вагонам
    metrics["rows_by_coach"] = (
        df["coach_no"].value_counts().sort_index().astype(int).to_dict()
    )

    # Свободная рассадка
    df_free = df[df["is_free_seating"] == 1]
    metrics["free_seating_total"] = len(df_free)

    # Служебные места
    df_service = df[df["is_service"] == 1]
    metrics["service_total"] = len(df_service)

    # Продаваемые места (грубый фильтр)
    df_sellable = df[
        (df["is_service"] != 1) &
        (df["is_free_seating"] != 1)
    ]
    metrics["sellable_total"] = len(df_sellable)

    return metrics


def test_validation_passes():
    """
    Базовая валидация CSV + reference для всех типов.
    Если здесь красное – значит поломали либо seatmaps, либо словари.
    """
    validate_seatmaps(SEATMAPS_DIR, CLASSES_PATH, COACH_TYPES_PATH)


@pytest.mark.parametrize("type_slug", ["type1", "type2", "type3", "type4"])
def test_each_type_has_10_nonempty_coaches(type_slug: str):
    """
    Минимальный инвариант для всех типов:
    - файл существует;
    - в нём есть вагоны 1..10;
    - ни один вагон не пустой.
    """
    csv_path = SEATMAPS_DIR / f"sapsan_{type_slug}_seats_v001.csv"
    if not csv_path.exists():
        pytest.skip(f"{csv_path.name} не найден, пропускаем тест для {type_slug}.")

    df = pd.read_csv(csv_path)
    coaches = sorted(df["coach_no"].unique())
    assert coaches == list(range(1, 11)), (
        f"{csv_path.name}: ожидались вагоны 1..10, а получили {coaches}"
    )

    counts = df["coach_no"].value_counts()
    empties = [c for c in range(1, 11) if c not in counts.index or counts[c] == 0]
    assert not empties, (
        f"{csv_path.name}: пустые или отсутствующие вагоны: {empties}"
    )


@pytest.mark.parametrize("type_slug, expected", list(EXPECTED.items()))
def test_metrics_match_expected(type_slug: str, expected: dict):
    """
    Проверка, что текущие CSV не разъехались с тем,
    что мы приняли как канон v001 (по факту).
    """
    csv_path = SEATMAPS_DIR / f"sapsan_{type_slug}_seats_v001.csv"
    if not csv_path.exists():
        pytest.skip(f"{csv_path.name} не найден, пропускаем тест для {type_slug}.")

    df = pd.read_csv(csv_path)
    metrics = compute_metrics(df)

    for key, exp_value in expected.items():
        assert key in metrics, f"{type_slug}: нет метрики {key!r}."
        got = metrics[key]
        assert got == exp_value, (
            f"{type_slug}: метрика {key!r} не совпадает.\n"
            f"Ожидалось: {exp_value}\n"
            f"Фактически: {got}"
        )
