"""Synthetic SCAL report laid out like a typical Russian core-lab workbook.

Sheets: "капл давл" (capillary curves, one merged block per sample, side calculation table to
the right), "ОФП" (sample summary on top, kr curves below with sample IDs only in the summary's
column), "Коэф.вытес." (two-row header, "Скважина №" separator rows), "смачив." (Amott, two-row
header). IDs mix Cyrillic and Latin letters (РT vs РТ), Кв is in percent, Pc is in MPa without
units in the header. Numbers follow exact dependencies (TRUTH) unless noise > 0.
"""

from __future__ import annotations

import math

import numpy as np
import openpyxl

TRUTH = {
    "lg_kw": (-0.16, 0.82),      # lg Kw = a + b lg Kg
    "swi": (0.39, -0.078),       # Swi = a + b lg Kw
    "sor": (0.315, -0.069),      # Sor = a + b lg Kw
    "lg_k_poro": (-1.4, 14.0),   # lg Kg = a + b phi
    "swirr_pc": (0.52, -0.15),   # Swirr (Pc) = a + b lg K
    "J_exp": (6.0, 5.0),         # J = a exp(-b Swn), reservoir sigma*cos(theta) = 30 cos 30
    "krw_end": 0.2, "nw": 2.0, "now": 2.5,
}
PC_STEPS_MPA = [0, 0.025, 0.056, 0.1, 0.157, 0.226, 0.307, 0.402, 0.628, 0.903, 1.23]
SIGMA_RES, SIGMA_LAB = 30 * math.cos(math.radians(30)), 48 * math.cos(math.radians(30))

KR_SAMPLES = [  # (well, model no, id in ОФП, id in Коэф.вытес., gas k, horizon)
    ("31", 1, "011004010РT01H", "011004010РТ01H", 1.2, "-"),
    ("31", 2, "011004032РT01H", "011004032РТ01H", 2.0, "РТ-IV"),
    ("31", 3, "011004034РT03H", "011004034РТ03H", 3.5, "РТ-IV"),
    ("31", 4, "011004045РT01H", "011004045РТ01H", 6.0, "-"),
    ("32", 1, "011005004РT01H", "011005004РТ01H", 9.0, "РТ-V"),
    ("32", 2, "011005007РT01H", "011005007РТ01H", 14.0, "РТ-V"),
    ("32", 3, "011005016РT01H", "011005016РТ01H", 22.0, "РТ-V"),
    ("33", 1, "011010036РТ02H", "011010036РТ02H", 35.0, "РТ-IV"),
    ("33", 2, "011010037РТ01H", "011010037РТ01H", 50.0, "РТ-IV"),
]
TYPO_SAMPLE = "011005007РT01H"  # porosity 0.76 in ОФП, 0.076 in Коэф.вытес.
PC_SAMPLES = [("31", f"0110040{n}РT0{1 + n % 3}H", k)
              for n, k in zip(range(50, 58), (0.5, 0.9, 1.6, 3.0, 5.5, 9.0, 16.0, 30.0))]


def _lin(key, x):
    a, b = TRUTH[key]
    return a + b * x


def sample_values(k_gas, noise=0.0, rng=None):
    rng = rng or np.random.default_rng(0)
    e = (lambda s: rng.normal(0, s)) if noise else (lambda s: 0.0)
    lg_kw = _lin("lg_kw", math.log10(k_gas)) + e(0.05 * noise)
    kw = 10 ** lg_kw
    poro = (math.log10(k_gas) - TRUTH["lg_k_poro"][0]) / TRUTH["lg_k_poro"][1] + e(0.005 * noise)
    swi = _lin("swi", lg_kw) + e(0.01 * noise)
    sor = _lin("sor", lg_kw) + e(0.01 * noise)
    return {"kw": kw, "poro": poro, "swi": swi, "sor": sor,
            "kdisp": (1 - swi - sor) / (1 - swi)}


def _merge(ws, rng_str, value):
    first = rng_str.split(":")[0]
    ws[first] = value
    ws.merge_cells(rng_str)


def build(path, noise: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    wb = openpyxl.Workbook()

    # ---------------------------------------------------------------- капл давл
    ws = wb.active
    ws.title = "капл давл"
    ws.append(["№ скв.", "№ обр.", "Кп %", "Кпр. мД", "Кв", "Рс"])
    row = 2
    a, b = TRUTH["J_exp"]
    for i, (well, sid, k) in enumerate(PC_SAMPLES):
        poro = (math.log10(k) - TRUTH["lg_k_poro"][0]) / TRUTH["lg_k_poro"][1]
        swirr = _lin("swirr_pc", math.log10(k))
        factor = SIGMA_RES * 1e-3 / math.sqrt(k * 9.869233e-16 / poro)  # Pa per unit J
        start = row
        for p_mpa in PC_STEPS_MPA:
            if p_mpa == 0:
                sw = 1.0
            else:
                j = p_mpa * 1e6 * SIGMA_RES / SIGMA_LAB / factor
                swn = min(1.0, max(0.0, math.log(a / j) / b))
                sw = swirr + swn * (1 - swirr)
                if noise:
                    sw = min(1.0, sw + rng.normal(0, 0.01 * noise))
            ws.cell(row, 5, round(sw * 100, 2))
            ws.cell(row, 6, p_mpa)
            row += 1
        # the last step always reaches Swirr exactly (the plateau of the curve)
        ws.cell(row - 1, 5, round(swirr * 100, 2))
        _merge(ws, f"B{start}:B{row - 1}",
               f"обр. {sid}, Кп={str(round(poro * 100, 1)).replace('.', ',')}%, "
               f"Кпр={str(k).replace('.', ',')}мД")
        _merge(ws, f"C{start}:C{row - 1}", round(poro * 100, 2))
        _merge(ws, f"D{start}:D{row - 1}", k)
        if i == 0:
            first_row = start
    _merge(ws, f"A{first_row}:A{row - 1}", int(PC_SAMPLES[0][0]))
    # the engineer's side calculation (no sample column) and a parameter block
    ws["H20"], ws["I20"], ws["J20"], ws["K20"], ws["L20"] = "Pc_Mpa", "Sw", "poro", "perm", "Swn"
    for r in range(21, 30):
        for c in range(8, 13):
            ws.cell(r, c, round(0.1 * (r - 20) * (c - 7), 3))
    ws["AE5"], ws["AF5"], ws["AG5"] = "σ", 30, "dynes/cm"
    ws["AE6"], ws["AF6"], ws["AG6"] = "нефть", 688, "кг/м3"

    # --------------------------------------------------------------------- ОФП
    ws = wb.create_sheet("ОФП")
    ws.append(["№ скв. ", "№ модель", "№ образца", "Глубина, м ",
               "Глубина (привязанная глубина), м ", "Горизонт",
               "Проницаемость для газа *10-3 . мкм2  ", "Проницаемость по пластовой воде, мД",
               "Пористость. доли ед.", "Остаточная водонасыщенность. доли ед. ",
               "Котн. воды. доли ед.", "Котн. нефти. доли ед."])
    values = {}
    for well, model, sid, _, kg, horizon in KR_SAMPLES:
        v = sample_values(kg, noise, rng)
        values[sid] = v
        poro = 0.76 if sid == TYPO_SAMPLE else round(v["poro"], 3)
        ws.append([int(well), model, sid, 1800 + model, 1801 + model, horizon, kg,
                   round(v["kw"], 3), poro, round(v["swi"], 3), TRUTH["krw_end"], 1])
    ws.merge_cells("A2:A5")
    ws.merge_cells("A6:A8")
    ws.merge_cells("A9:A10")
    row = ws.max_row + 4
    ws.cell(row, 4, "Sw"), ws.cell(row, 5, "krw"), ws.cell(row, 6, "krow")
    ws.cell(row + 1, 5, "вода"), ws.cell(row + 1, 6, "нефть")
    row += 2
    for well, _, sid, _, _, _ in KR_SAMPLES:
        v = values[sid]
        swi, sor = round(v["swi"], 3), round(v["sor"], 3)
        ws.cell(row, 1, int(well))
        ws.cell(row, 3, sid)
        for sw in np.linspace(swi, 1 - sor, 7):
            sn = min(1.0, max(0.0, (sw - swi) / (1 - swi - sor)))
            ws.cell(row, 4, round(float(sw), 4))
            ws.cell(row, 5, round(TRUTH["krw_end"] * sn ** TRUTH["nw"], 5))
            ws.cell(row, 6, round((1 - sn) ** TRUTH["now"], 5))
            row += 1

    # ------------------------------------------------------------- Коэф.вытес.
    ws = wb.create_sheet("Коэф.вытес.")
    heads = ["№ модель", "№образца", "Горизонт", "Глубина, м", "Привязанная глубина, м",
             "Длина модели L, см", "Диаметр модели D, см", "Объем пор модели Vпор, см3",
             "Пористость, доли ед. ", "Проницаемость по пластовой воде, мД",
             "Остаточная водонасыщенность, Sвост д. ед.",
             "Остаточная нефтенасыщенность, Sвост д. ед.", "Коэффициент вытеснения, β, д. ед."]
    for c, h in enumerate(heads, 1):
        ws.cell(1, c, h)
        ws.merge_cells(start_row=1, start_column=c, end_row=2, end_column=c)
    ws.cell(3, 17, "ср.")
    row, current = 3, None
    for well, model, _, sid, kg, horizon in KR_SAMPLES:
        if well != current:
            _merge(ws, f"B{row}:M{row}", f"Скважина №{well}")
            row, current = row + 1, well
        v = values[next(s for w, m, s, _, _, _ in KR_SAMPLES if w == well and m == model)]
        swi, sor = round(v["swi"], 3), round(v["sor"], 3)
        ws.append([model, sid, horizon, 1800 + model, 1801 + model, 4.5, 3.77, 8.0,
                   round(v["poro"], 3), round(v["kw"], 3), swi, sor,
                   round((1 - swi - sor) / (1 - swi), 4), (1 - swi - sor) / (1 - swi)])
        row += 1

    # ------------------------------------------------------------------ смачив.
    ws = wb.create_sheet("смачив.")
    ws["A1"], ws["B1"], ws["C1"] = "№ скв.", "№ образца", "Интервал, м"
    ws["D1"], ws["F1"], ws["H1"] = ("Вытеснение водой", "Вытеснение нефтью",
                                    "Показатель смачиваемости Амотта")
    ws["D2"], ws["E2"] = "Самопроизвольное выход нефти, мл", "Принудительное выход нефти, мл"
    ws["F2"], ws["G2"] = "Самопроизвольное выход нефти, мл", "Принудительное выход нефти, мл"
    for rng_str in ("A1:A2", "B1:B2", "C1:C2", "D1:E1", "F1:G1", "H1:I2"):
        ws.merge_cells(rng_str)
    for i, (well, _, sid, _, _, _) in enumerate(KR_SAMPLES):
        ws.append([int(well) if i in (0, 4, 7) else None, sid, 1800 + i, 1.0, 4.0, 0.5, 4.5,
                   0.15 + 0.01 * i, "гидрофильная"])
    wb.save(path)
    return path
