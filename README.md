# relperm — интерактивные ОФП (Corey / LET) с экспортом в Eclipse и CMG

Строит кривые относительной фазовой проницаемости (ОФП) для систем вода–нефть и газ–нефть,
даёт крутить все параметры ползунками и выгружает готовые таблицы для симуляторов:

| Формат        | Ключевые слова              | Где работает                       |
|---------------|-----------------------------|------------------------------------|
| `eclipse`     | `SWOF` + `SGOF`             | Eclipse 100/300, OPM Flow, tNavigator |
| `eclipse2`    | `SWFN` + `SGFN` + `SOF3`    | Eclipse (family II), OPM, tNavigator  |
| `cmg`         | `*ROCKFLUID`, `*SWT`, `*SLT`| CMG IMEX / GEM / STARS             |

![interactive editor](docs/app.png)

Идея взята из скрипта [MouinAlmasoodi/Interactive_Relative_Permeability](https://github.com/MouinAlmasoodi/Interactive_Relative_Permeability)
(Mouin Almasoodi, PhD) и переписана в виде пакета.

## Установка

```bash
pip install -e .          # numpy + matplotlib
pip install -e ".[test]"  # + pytest
```

## Быстрый старт

```bash
relperm template -o my_field.json                      # шаблон конфига
relperm app my_field.json                              # интерактивный редактор
relperm app my_field.json --rock SAND --lab-wo examples/lab_water_oil.csv   # + керновые точки
relperm export my_field.json -f eclipse -o relperm.inc # SWOF/SGOF для всех типов пород
relperm export my_field.json -f cmg -o rockfluid.dat   # *SWT/*SLT
relperm plot my_field.json --log -o curves.png         # картинка для отчёта
```

Пример конфига с двумя типами пород (SATNUM 1, 2) — [`examples/two_rock_types.json`](examples/two_rock_types.json).
Порядок `rock_types` = номер таблицы (`SATNUM` в Eclipse, `*RPT` в CMG).

### Интерактивный редактор

* ползунки на все насыщенности, концевые точки и показатели Кори (в оригинале — только `Krg` и `ng`);
* точки, которые реально попадут в таблицу симулятора, видны поверх кривых;
* лог-шкала по kr — удобно смотреть «хвосты» krw/krow при адаптации;
* кривая доли воды `fw(Sw)` с ползунком `μo/μw` — сразу видно, как форма ОФП влияет на фронт Баклея–Леверетта;
* недопустимые комбинации (например `Swcr < Swl`) подсвечиваются красным и не ломают график;
* кнопки **Export Eclipse / Export CMG** пишут `<имя>_eclipse.inc` и `<имя>_cmg.dat`.

### Подбор Кори по керну

Линейная регрессия в лог-координатах (`ln kr = ln kr_max + n·ln Sn`), без scipy:

```bash
relperm fit examples/lab_water_oil.csv --phase ow --swl 0.2 --sorw 0.15
# kr_max = 0.8796, n = 2.629, RMSE = 0.003659 (6 points)
relperm fit lab.csv --phase w --swcr 0.2 --sorw 0.15 --krmax 0.45   # фиксировать концевую точку
```

`--phase`: `w`=krw, `ow`=krow, `g`=krg, `og`=krog. CSV может быть с `,` или `;` (и десятичной запятой).

### Python API

```python
from relperm import RockType, LET, export, fit_corey

rt = RockType(name="SAND", swl=0.18, swcr=0.2, sowcr=0.11, krwr=0.6, krwmax=1.0,
              nw=3, now=LET(L=2.5, E=1.2, T=1.5))
rt.water_oil_table(points=25)        # dict Sw, krw, krow, Pcow
rt.kro_stone2(sw=0.4, sg=0.2)        # трёхфазная kro, нормированный Stone II
rt.fractional_flow(0.5, mu_w=0.4, mu_o=3.0)
print(export([rt], "eclipse"))
```

## Модель

Насыщенности названы как в Eclipse (`SWL, SWCR, SOWCR, SGL, SGCR, SOGCR`):

```
krw  = krwr   · Sn^nw,  Sn = (Sw − Swcr)          / (1 − Swcr − Sorw)
krow = krocw  · Sn^now, Sn = (1 − Sw − Sorw)      / (1 − Swl − Sorw)
krg  = krgr   · Sn^ng,  Sn = (Sg − Sgcr)          / (1 − Sgcr − Swl − Sorg)
krog = krogcg · Sn^nog, Sn = (1 − Sg − Swl − Sorg) / (1 − Sgl − Swl − Sorg)
```

* Выше `1 − Sorw` krw линейно растёт до `krwmax` при Sw = 1 (аналогично krg до `krgmax` при Sg = 1 − Swl).
  По умолчанию `krwmax = krwr`, `krgmax = krgr`.
* Любой показатель (`nw, now, ng, nog`) можно заменить на LET: `{"L": 2, "E": 1.5, "T": 1.2}`.
* Критические насыщенности всегда добавляются в таблицу как узлы — симулятор не «размажет» точку начала течения.

## Что исправлено относительно оригинального скрипта

1. **Экспорта не было вовсе**, хотя README его обещал — теперь SWOF/SGOF, SWFN/SGFN/SOF3 и CMG SWT/SLT, несколько типов пород.
2. `(отрицательное число) ** n` давало NaN/отрицательные kr у остаточной нефти; это лечилось ручным
   `SWT['Krow'].iloc[-1] = 0` (chained assignment, в pandas ≥ 3 молча не работает). Теперь нормированная насыщенность обрезается в [0, 1].
3. SWOF начинался с захардкоженного `0.18` и заканчивался на `1 − Sorw`; Eclipse ждёт таблицу от `Swl` до 1, SGOF — от 0 до `1 − Swl`.
4. `Swcon` и `Swcrit` были склеены в одну переменную — теперь это разные параметры (`swl`, `swcr`).
5. Проверка согласованности: `krow(Swl)` должно равняться `krog(Sg=0)`, иначе Eclipse ругается при Stone — выводится предупреждение.
6. Валидация входных данных с понятными сообщениями, тесты (`pytest`), CI на GitHub Actions.

## Тесты

```bash
pytest -q
```
