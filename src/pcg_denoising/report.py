import re
import numpy as np

from datetime import datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from pcg_denoising.utils.evaluation import load_results

KIND_ORDER = ["AWGN", "APGN", "PhysioNet", "Speech"]
STAT_HEADERS = ["Min SI-SDR (dB)", "Max SI-SDR (dB)", "Mean SI-SDR (dB)", "Std SI-SDR (dB)",
                "Noisy SI-SDR (dB)", "Improvement (dB)", "Avg run time (s)"]
SCORE_FORMAT = "0.00"


def _std(x):
    return float(np.std(x)) # population std


def _style_header(cell):
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor="305496")
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _write_header(ws, headers):
    for col, h in enumerate(headers, start=1):
        _style_header(ws.cell(row=1, column=col, value=h))
        ws.column_dimensions[get_column_letter(col)].width = max(12, len(str(h)) + 2)
    ws.row_dimensions[1].height = 32
    ws.freeze_panes = "B2"


def _write_row(ws, row, values, bold=False):
    for col, v in enumerate(values, start=1):
        if isinstance(v, Path):
            v = str(v)
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        cell = ws.cell(row=row, column=col, value=v)
        if isinstance(v, float):
            cell.number_format = SCORE_FORMAT
        if bold:
            cell.font = Font(bold=True)


def _sheet_name(kind, dB):
    return re.sub(r"[\[\]:*?/\\]", "_", f"{kind}_{dB:g}dB")[:31]


def _stats(df):
    """Statistics of one group of recordings, as the list matching STAT_HEADERS (without std handling)."""
    return [df["sisdr"].min(), df["sisdr"].max(), df["sisdr"].mean(), _std(df["sisdr"]),
            df["sisdr_noisy"].mean(), df["sisdr_improvement"].mean(), df["run_time"].mean()]


def _conditions(df):
    keys = df[["kind", "dB"]].drop_duplicates().itertuples(index=False, name=None)
    order = lambda k: (KIND_ORDER.index(k[0]) if k[0] in KIND_ORDER else len(KIND_ORDER), k[0], k[1])
    return sorted(keys, key=order)


def write_report(csv_path, xlsx_path, settings):
    """Rebuild the whole workbook from results.csv.
    settings: ordered dict {"Transform": "STFT", "Learning Rate": 1e-3, ...}, shown on every row."""
    df = load_results(csv_path)
    if df.empty:
        print("No results to report")
        return None

    path = Path(xlsx_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            with open(path, "ab"):
                pass
        except PermissionError:
            raise PermissionError(f"Cannot write {path}. Close it in Excel and run again.") from None

    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"
    setting_names, setting_values = list(settings), list(settings.values())

    fold_headers = ["Fold", *setting_names, "Audios", *STAT_HEADERS]
    summary_headers = ["Noise", "Level (dB)", "Folds", "Audios", *setting_names, *STAT_HEADERS]
    _write_header(summary, summary_headers)
    summary_row = 2

    for kind, dB in _conditions(df):
        g = df[(df["kind"] == kind) & (df["dB"] == dB)]
        ws = wb.create_sheet(_sheet_name(kind, dB))
        _write_header(ws, fold_headers)

        row = 2
        fold_stats = []
        for fold, gf in g.groupby("fold"):
            stats = _stats(gf)
            fold_stats.append(stats)
            _write_row(ws, row, [int(fold), *setting_values, len(gf), *stats])
            row += 1

        arr = np.array(fold_stats, dtype=float)
        blank = [""] * len(setting_values)
        _write_row(ws, row, ["Average", *blank, "", *arr.mean(axis=0)], bold=True)
        _write_row(ws, row + 1, ["Std of fold means", *blank, "", *arr.std(axis=0)], bold=True)
        _write_row(ws, row + 2, ["Pooled", *blank, len(g), *_stats(g)], bold=True)
        notes = [
            "Average: mean of the fold values above (including the within-fold Std column).",
            "Std of fold means: how much each column varies from fold to fold.",
            "Pooled: all recordings of all folds together; its Std is over all recordings.",
        ]
        for i, text in enumerate(notes):
            ws.cell(row=row + 4 + i, column=1, value=text).font = Font(italic=True, color="595959")

        avg = arr.mean(axis=0)
        pooled_std = _std(g["sisdr"])
        _write_row(summary, summary_row,
                   [kind, dB, len(fold_stats), len(g), *setting_values,
                    avg[0], avg[1], avg[2], pooled_std, avg[4], avg[5], avg[6]])
        summary_row += 1

    notes = [
        "Min, Max, Mean, Noisy, Improvement and run time: averaged across folds.",
        "Std: pooled over all recordings of all folds (not the std of the fold means).",
        "Each noise kind and level has its own sheet with the per-fold detail.",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}",
    ]
    for i, text in enumerate(notes):
        summary.cell(row=summary_row + 1 + i, column=1, value=text).font = Font(italic=True, color="595959")

    tmp = path.with_name(path.stem + ".tmp.xlsx")
    wb.save(tmp)
    tmp.replace(path)
    print(f"Report: {len(wb.sheetnames) - 1} conditions -> {path}")
    return path