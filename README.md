# Avito ranking hackathon 2026

Ранжирование кандидатов в рекомендательной ленте. Метрика — NDCG по лентам.
Дедлайн выбора финала: 09.12.2026 до 12:00.

## Ветки
- `main` — только проверенный код (прошёл валидатор, результат на val записан в `experiments/log.md`)
- `danya` — GBDT, ансамбль
- `avgust` — нейросеть
- `dima` — данные, метрика, валидация
- `anton` — признаки из истории

Правила: в чужие ветки не пушим; чужой код забираем через `main`; без `push --force`;
данные, модели и сабмиты в git не кладём (см. `.gitignore`).

## Структура
```
src/common/metric.py     NDCG / NDCG@10 / NDCG@30, сверены с эталоном на синтетике
src/common/data.py       чтение лент, признаки сходства/истории, ранги внутри ленты
src/common/validation.py временное разбиение val и групповые OOF-фолды
src/common/preparation.py подготовка таблиц и точный EDA по всем данным
src/gbdt/train.py        простой CatBoost YetiRank: train / predict (бейзлайн)
src/gbdt/run_exp.py      раннер экспериментов: CatBoost/LightGBM, позиция, режимы holdout/cv/final
src/gbdt/tune.py         Optuna по val (ES на pseudo-public, цель — pseudo-private)
src/gbdt/weights.py      IPW-веса против смещения позиции
src/gbdt/extra_features.py  подключение признаков Антона/Августа (контракт на формат — в docstring)
src/ensemble/blend.py    общий ансамбль N моделей: mean / подбор весов / стекинг по OOF
src/ensemble/select_final.py  выбор двух финальных решений
src/common/cv.py         pseudo-public/private (30/70 по времени) и временные фолды
src/nn/                  нейросетевой пайплайн Августа (docs/AVGUST_PIPELINE.md)
src/ensemble/blend_nn_gbdt.py  ранговый блендинг NN + GBDT (Август), сетка по весу
src/ensemble/stacking.py Ridge-стекинг NN + GBDT (Август)
eval/make_synthetic.py   крошечный синтетический датасет для проверки кода (не данные Авито)
scripts/validate_submission.py   проверка формата сабмита
scripts/check_dataset.py         проверка распакованного датасета
scripts/make_validation_split.py создание таблицы разбиения val
scripts/prepare_data.py          полный проход: Parquet + EDA
scripts/build_eda_notebook.py    исполняемый отчёт с графиками
scripts/run_baseline_check.py    небольшой CatBoost: весь val, test и валидатор
scripts/reproduce_data.py        одна команда для данных, EDA, фолдов и cosine-кеша
scripts/build_infeed_features.py ранги, z-score, размер и категории внутри ленты
src/common/embedding_catalogue.py общий каталог версий эмбеддингов
src/common/similarity.py         пакетные признаки сходства
src/common/history.py            признаки «кандидат × история пользователя» (46 шт.)
scripts/build_history_features.py кеш группы history
notebooks/01_data_eda.ipynb       выполненный EDA по всему датасету
tests/                          сверка с эталоном и проверки данных/сабмитов
experiments/log.md       таблица экспериментов
models/nn/, features/    зона Августа и Антона
```

## Запуск

Окружение данных и GBDT: Python 3.11, версии фиксируются в `uv.lock`.
Основные зависимости совпадают с предоставленным бейзлайном. PyTorch для
нейросети устанавливается отдельно из `competition/baseline/pyproject.toml`.

Из корня проекта в PowerShell:

```powershell
uv sync --locked --no-config --default-index https://pypi.org/simple
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Для сверки с эталоном нужен скачанный
`competition/baseline/calculate_metrics.py`; без него этот тест пропускается.
Проверки используют искусственные данные и не измеряют качество модели.

После распаковки `competition.zip` папки `train`, `val`, `test` должны
лежать непосредственно в `competition/`. Проверить структуру и создать фолды:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_dataset --data competition --sample-feeds 100
.\.venv\Scripts\python.exe -m scripts.make_validation_split --data competition --out runs/val_folds.parquet
.\.venv\Scripts\python.exe scripts/validate_submission.py competition/sample_submission.parquet runs/submission.parquet
```

Проверка датасета читает схемы всех шардов, размеры memmap и все `feed_id`
для поиска дублей и null в ключах/времени. Содержимое списков проверяется
только на первых `--sample-feeds` лентах каждого split; полный EDA нужен отдельно.

Таблица `val_folds.parquet`: `feed_id`, `timestamp`, `leaderboard_split`, `fold`.
Ранние ~30% лент — псевдо-Public, поздние ~70% — псевдо-Private.
Ленты с одинаковым timestamp остаются на одной стороне, поэтому доля может
отличаться от 30%. Фолды (по умолчанию 5, seed=42) случайные и группируются
по `feed_id`; все кандидаты ленты получают одно назначение через join по ключу.
Для OOF модель обучают без фолда, на котором получают предсказания.
Фолды не разделяют пользователей и не гарантируют обучение только на прошлом;
таргетные статистики считаются отдельно по прошлому или OOF.
Подбор по псевдо-Private раскрывает этот holdout, поэтому финальную оценку
устойчивости нужно интерпретировать с учётом использованных экспериментов.

Эталон считает DCG и IDCG с одинаковым отсечением для @10/@30.
`ndcg_idcg30` сохранён для совместимости как экспериментальный вариант,
он не входит в метрики предоставленного `calculate_metrics.py`.

## Подготовленные данные и полный EDA

```powershell
$env:POLARS_MAX_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -u -m scripts.prepare_data --data competition --out data/prepared --eda-out runs/eda
.\.venv\Scripts\python.exe -m scripts.build_eda_notebook
.\.venv\Scripts\python.exe -m scripts.make_validation_split --data competition --out data/prepared/val_folds.parquet
```

Подготовка обрабатывает по одному исходному шарду. При прерывании добавьте
`--resume` к той же команде: записанные части переиспользуются только при
совпадении версии формата, отпечатков источника/выходных файлов и числа строк.
EDA при возобновлении пересчитывается по всем исходным данным. Файлы записываются
через временный файл; `manifest.json` получает `complete=true` только после
всех проверок, включая точное совпадение ключей test с шаблоном сабмита.

```text
data/prepared/
  manifest.json
  train/                       # аналогично val/ и test/
    candidates/part-*.parquet   # одна строка = кандидат
    feeds/part-*.parquet        # одна строка = лента, история хранится один раз
runs/eda/
  summary.json                 # полный EDA, без выборки
  figures/                     # графики из ноутбука
notebooks/
  01_data_eda.ipynb             # код, выполненные таблицы и встроенные графики
  eda_summary.json              # переносимый снимок агрегатов для ноутбука
```

Ключ строки кандидата: `(split, feed_id, candidate_index)`, где candidate_index —
uint32, исходный номер кандидата внутри ленты с 0. Это технический ключ, не признак
модели. В train встречаются повторные item_id в одной ленте с разными pos и f1–f5;
они сохраняются без удаления или объединения меток. Число повторов и конфликтов
меток отражено в EDA, сами строки — в `runs/eda/duplicate_candidates/`.
Признаки уровня товара можно соединять по `(split, feed_id, item_id)` только
как many-to-one; для признаков конкретного вхождения нужен candidate_index.
В test ключи `(feed_id, item_id)` дополнительно сверяются с шаблоном сабмита.

Колонки: `feed_id`, `timestamp`,
`user_id`, `item_id`, `item_id__embedding_idx` — int32; `category_id` — int8;
f1–f5 — float32; target — int8 только в train/val, pos — int16 только в train.
`split` — строка. Таблица feeds содержит `(split, feed_id)`, timestamp, user_id,
user_embedding, все поля history и feed_size. Истории присоединяют по ключу ленты.
Это исходные признаки: сходство с историей и обученные признаки сохраняются
отдельными группами. Таблицы эмбеддингов остаются в `competition/<split>/item_embeddings/`;
их относительный путь указан через `raw_root_relative` в манифесте. При передаче
команде подготовленных таблиц нужно также передать исходные таблицы эмбеддингов.

Ленивое чтение без загрузки всего train в память:

```python
from src.common.preparation import scan_candidates

val = scan_candidates("data/prepared", "val")
print(val.select("feed_id", "candidate_index", "item_id", "f1", "target").limit(5).collect())
```

EDA включает размеры лент/историй, доли позитивов и лент без позитивов, CTR по
позиции, пропуски эмбеддингов, точные пересечения user/item ID, распределения
признаков/категорий/типов событий и объём по датам UTC. Нулевые user/item ID
исключены из уникальных объектов и пересечений, их вхождения посчитаны отдельно.
Таблицы и исходные данные игнорируются Git; ноутбук содержит только агрегаты.

На полном train используйте чтение по частям или отбор целых лент: существующий
`load_split()` загружает весь split в память, даже если подготовка была потоковой.

## Обучение

### Команда воспроизведения для команды

Каждый участник скачивает `competition.zip` с платформы соревнования и
распаковывает его в `competition/`. Дополнительный уровень `competition/competition/`
определяется автоматически. Из корня репозитория:

```powershell
uv run --locked python -m scripts.reproduce_data --raw competition --resume
```

Полная проверочная цепочка от сырых файлов до валидированного сабмита одной
командой, включая обучение cosine-baseline на одном train-шарде:

```powershell
uv run --locked python -m scripts.reproduce_data --raw competition --resume --baseline-out runs/reproduced-baseline-001
```

Для этого режима нужен предоставленный `competition/baseline/calculate_metrics.py`.
Используются seed=42, фиксированные версии зависимостей, CPU=2 потока, depth=6,
до 100 итераций с выбором по псевдо-Public. Выходная папка модели должна быть
новой. Псевдо-Private оценивается после fit; полноценное обучение на всём train
этот проверочный режим не подразумевает. Дополнительные группы признаков
включаются отдельным вызовом модельного скрипта после сравнения качества.

Команда проверяет и подготавливает все исходные таблицы, EDA-ноутбук в `runs/eda/`,
разбиение val, общий каталог эмбеддингов и cosine-кеш для первого train-шарда,
**всего val и всего test**. Подготовленные базовые таблицы при этом полные,
все 512 train-шардов. Статистики внутри ленты можно дополнительно построить
через `--infeed data/features/infeed`; по умолчанию необязательная группа
не вычисляется. Проверенный кеш переиспользуется; повреждённые файлы
не считаются готовыми. Передача архивов и тяжёлых таблиц через Git не нужна.

По умолчанию cosine-кеш соответствует небольшому проверочному прогону.
Полная материализация cosine-признаков train на текущем CPU может занять
несколько часов. Она запускается явно с `--all-train-features`; промежуточный
объём — через `--feature-train-parts N`. Покрытие каждого split записано в
`data/features/cosine/manifest.json`: `complete` относится к запрошенному
покрытию, `all_train_parts` показывает наличие всех train-шардов. Потребитель
читает перечень файлов из манифеста и отклоняет незакешированные шарды.

Полные базовые таблицы и обе группы признаков одной командой (общая загрузка
векторов для четырёх шардов использует больше RAM):

```powershell
uv run --locked python -m scripts.reproduce_data --raw competition --resume --all-train-features --prefetch-parts 4 --infeed data/features/infeed
```

### Каталог эмбеддингов и признаки сходства

Для одного item_id векторы разных split могут различаться. Полный побайтовый
аудит подтвердил различия у 17,30% общих товаров train/val, 27,92% train/test
и 20,60% val/test. Поэтому ключ версии — **`(split, item_id)`**. Каталог
`data/embedding_catalogue/` объединяет 83 260 664 товара, плотные индексы и
Parquet-таблицу item_id с индексами всех трёх split. Векторы хранятся в исходных
memmap-файлах, каталог ссылается на них по относительному пути; повторная
копия 11 ГБ векторов не создаётся. Внутри split проверена согласованность
всех соответствий item_id и векторов.

```python
from src.common.embedding_catalogue import EmbeddingCatalogue

store = EmbeddingCatalogue("data/embedding_catalogue")
vectors = store.gather("val", item_ids, original_missing=(embedding_indices == -1))
```

Cosine-группа `data/features/cosine/` имеет ключ `(split, feed_id, candidate_index)`
и контрольную колонку item_id. Признаки: наличие эмбеддинга кандидата,
cos(user, item), длина истории и для последних 10/50 **исходных событий** —
число доступных эмбеддингов, mean/max cosine. Пропуски исключаются внутри
выбранного окна, не заменяются более старыми событиями. При пустой истории
cosine равен 0. Математика выполняется в float32; исходный `-1` сохраняется,
версии и пропуски не подменяются данными других split. Изменённый кеш
проверяется по исходным файлам, версии расчёта и числу строк.

Самостоятельная сборка и расширение кеша:

```powershell
uv run --locked python -m scripts.build_embedding_catalogue --data competition --out data/embedding_catalogue --resume
uv run --locked python -m scripts.build_similarity_features --train-parts 1 --resume
```

При расширении cosine-кеша проверенные готовые шарды сохраняются: последующий
запуск на одном шарде не исключает остальные из манифеста. Отпечаток численной
формулы и параметров отделён от способа чтения; у каждого файла записана версия
производителя. Поддерживается миграция ранее опубликованного стандартного кеша.

Режим `--reader auto` читает большой train-memmap последовательно и удерживает
в RAM только нужные векторы кандидатов и первых 50 событий. Для небольших
val/test остаётся чтение через отображаемые окна. На одном реальном train-шарде
проверено побитовое совпадение девяти Float32-признаков с прежним кешем.
Скорость зависит от файлового кеша и носителя; замер сохраняется отдельно:

```powershell
uv run --locked python -m scripts.benchmark_similarity --out runs/cosine-reader-benchmark.json
```

Полный cosine-кеш train можно построить с общим чтением векторов для четырёх
шардов за проход. Это сокращает повторное чтение большого memmap и увеличивает
потребление RAM; `--prefetch-parts 1` оставляет обработку по одному шарду.
Прогресс сохраняется атомарно после каждого готового файла, а записи val/test
сохраняются и при прерывании во время расчёта train.

```powershell
uv run --locked python -m scripts.build_similarity_features --resume --prefetch-parts 4
```

После завершения полный кеш проверяется отдельной командой. Для всех строк
сверяются ключи, типы, конечность чисел, доступность эмбеддинга и счётчики
истории; на трёх лентах каждого шарда и всех лентах с повторными item_id
формулы независимо пересчитываются в Float64. Полнота численной проверки
явно ограничена этой выборкой.

```powershell
uv run --locked python -m scripts.audit_similarity_features --out runs/cosine_cache_audit_full.json
```

### Проверка моделей

Увеличение обучающей выборки проверяется отдельно от изменения признаков:

```powershell
uv run --locked python -m scripts.build_similarity_features --train-parts 2 --resume
uv run --locked python -m scripts.run_baseline_check --feature-root data/features/cosine --train-parts 2 --out runs/baseline-cosine-2parts-001
uv run --locked python -m scripts.compare_baselines --baseline runs/baseline-cosine-001 --candidate runs/baseline-cosine-2parts-001 --allow-train-growth --out runs/baseline-cosine-2parts-001/comparison.json
```

Для такого сравнения обязательны те же признаки, параметры, val-строки и
назначения Public/Private; train должен быть строгим расширением исходных
шардов. Без `--allow-train-growth` несовпадающий train отклоняется. В `run.json`
пишутся этап выполнения, доступная RAM и пиковый рабочий набор процесса —
это измерение, а не жёсткий лимит памяти.

Группа `data/features/infeed/` содержит 12 признаков только из текущей ленты:
размер, долю категории кандидата и для f1–f5 — нормированный ранг и z-score.
Ранги возрастающие: средний ранг при равенстве / размер ленты. Z-score использует
генеральное стандартное отклонение (`ddof=0`), для постоянной и одиночной ленты
равен 0. Target, исходная позиция, другие ленты и ID кандидатов в расчёте не
участвуют. Повторные item_id остаются отдельными кандидатами с candidate_index.
Старый общий загрузчик использует те же формулы, включая конечный z-score для
одиночных лент.

```powershell
uv run --locked python -m scripts.build_infeed_features --prepared data/prepared --out data/features/infeed --resume
```

Для быстрого сравнения на одном train-шарде, полном val и test:

```powershell
uv run --locked python -m scripts.build_infeed_features --prepared data/prepared --out data/features/infeed-check --train-parts 1 --resume
```

Параметр `--feature-root` можно повторять: группы соединяются по ключу
конкретного вхождения кандидата, дубли имён признаков отклоняются. Сравнение
cosine-модели с добавлением признаков внутри ленты:

```powershell
uv run --locked python -m scripts.run_baseline_check --feature-root data/features/cosine --feature-root data/features/infeed-check --out runs/baseline-infeed-001
uv run --locked python -m scripts.compare_baselines --baseline runs/baseline-cosine-001 --candidate runs/baseline-infeed-001 --out runs/baseline-infeed-001/comparison.json
```

Проверка полной цепочки на подготовленных таблицах (один train-шард,
100 итераций, CPU, f1–f5 и категория):

```powershell
.\.venv\Scripts\python.exe -u -m scripts.run_baseline_check --data data/prepared --raw competition --out runs/baseline-check-001
.\.venv\Scripts\python.exe -u -m scripts.run_baseline_check --data data/prepared --raw competition --feature-root data/features/cosine --out runs/baseline-cosine-001
```

Нужны `data/prepared/val_folds.parquet` и скачанный эталон метрики.
Параметры размера прогона: `--train-parts`, `--iterations`, `--depth`,
`--seed`, `--threads`. Лучший чекпойнт выбирается только по NDCG@30
псевдо-Public; псевдо-Private не передаётся в обучение или early stopping.
Все ленты val оцениваются с float32-скорами; full/@10/@30 сверяются с эталоном
отдельно для всего val и обеих временных частей. Это проверочный прогон
на небольшой части train, не финальное обучение.

Выходы: `model.cbm`, `validation_predictions.parquet`, `submission.parquet`,
`run.json` с параметрами, коммитом/хешами кода, метриками и результатом валидатора.
Val-предсказания — holdout, не OOF: псевдо-Public участвовал в выборе итерации.
Сабмит создаётся локально. Скрипт не отправляет его на платформу и не
перезаписывает существующую папку прогона.

Обучение и инференс (bash, после активации окружения):

```bash
source .venv/bin/activate

## Запуск GBDT-раннера (Даня)
```bash
export PYTHONPATH=.
python -m src.gbdt.train train --data /path/to/competition --out runs/gbdt-01
python -m src.gbdt.train predict --data /path/to/competition \
    --model runs/gbdt-01/model.cbm --out runs/gbdt-01/submission.parquet
python scripts/validate_submission.py /path/to/competition/sample_submission.parquet runs/gbdt-01/submission.parquet
```
Синтетика для отладки (формат соревнования, смещение позиции в train, несколько частей; НЕ данные Авито):
`python eval/make_synthetic.py --out data/synth --feeds 20000 --parts 4` и тот же запуск с `--data data/synth`.
GPU: `--task-type GPU`. Для быстрых экспериментов: `--max-parts 2`.

## Формат общих артефактов
- Предсказание модели: parquet `feed_id:int32, item_id:int32, score:float32`.
- Для ансамбля также нужны OOF-предсказания на val в том же формате (с колонкой `target`).

## Эксперименты GBDT (зона Дани)
```bash
export PYTHONPATH=.
D=/path/to/competition
# 1. сравнение лоссов и режимов позиции (кэш признаков ускоряет повторные запуски; --n-jobs — процессы)
python -m src.gbdt.run_exp --data $D --out runs/cb-yeti   --cache runs/cache --n-jobs 4 --log
python -m src.gbdt.run_exp --data $D --out runs/cb-pair   --cache runs/cache --loss YetiRankPairwise --log
python -m src.gbdt.run_exp --data $D --out runs/lgb       --cache runs/cache --kind lightgbm --log
python -m src.gbdt.run_exp --data $D --out runs/lgb-ipw   --cache runs/cache --kind lightgbm --pos-mode ipw --log
python -m src.gbdt.run_exp --data $D --out runs/cb-posf   --cache runs/cache --pos-mode feature --log
python -m src.gbdt.run_exp --data $D --out runs/cb-top50  --cache runs/cache --pos-mode topn --topn 50 --log
# 2. подбор параметров, затем перепроверка лучших режимом cv
python -m src.gbdt.tune --data $D --out runs/tune-lgb --kind lightgbm --trials 40 --cache runs/cache
python -m src.gbdt.run_exp --data $D --out runs/lgb-tuned --kind lightgbm --cache runs/cache $(cat runs/tune-lgb/best_cli.txt)
# 3. OOF на val для ансамбля, затем финальное обучение на train+val
python -m src.gbdt.run_exp --data $D --out runs/lgb-cv --kind lightgbm --mode cv --iterations-from runs/lgb-tuned/metrics.json ...
python -m src.gbdt.run_exp --data $D --out runs/lgb-final --kind lightgbm --mode final --val-weight 3 --iterations-from runs/lgb-tuned/metrics.json ...
# 4. ансамбль и выбор финала
python -m src.ensemble.blend --data $D --runs runs/lgb-cv runs/cb-cv runs/nn-cv --out runs/blend
python -m src.ensemble.select_final --data $D --runs runs/lgb-cv runs/cb-cv runs/blend --safe runs/lgb-cv runs/cb-cv
python scripts/validate_submission.py $D/sample_submission.parquet runs/blend/test_pred.parquet
```
Заметки:
- CatBoost с парными лоссами (YetiRank, PairLogit) **игнорирует веса объектов**, поэтому IPW и `--val-weight` работают с LightGBM или с `--loss LambdaMart`; раннер сам остановится с ошибкой.
- `--log` дописывает строку в `experiments/log.md` — включать только на реальных данных.
- Метрики `priv_*` — на pseudo-private (поздние 70% val по времени); early stopping идёт по pseudo-public.
- Все числа на синтетике (`eval/make_synthetic.py`) проверяют только работоспособность кода.

## Установка на Windows (PowerShell)
Единое окружение команды — Python 3.11 и версии из `uv.lock` (код Димы не работает с polars 2.x и Python 3.12).
```powershell
winget install Python.Python.3.11
winget install astral-sh.uv          # затем перезапустить PowerShell
uv sync --locked --no-config --default-index https://pypi.org/simple
.\.venv\Scripts\python.exe -m unittest discover -s tests
```
Если `.venv` уже создан другой версией Python — удалить папку `.venv` перед `uv sync`.
В PowerShell команды запускаются по одной строке (без `&&`); перенос строки — обратный апостроф.

## Быстрый старт для нейросети (Август)

Подробная техническая документация архитектуры и контрактов: [docs/AVGUST_PIPELINE.md](docs/AVGUST_PIPELINE.md).

**Рекомендации по железу:**
- **CPU (Windows / локальная разработка):** добавляйте `--num-workers 0 --batch-size 32`.
- **GPU (Linux / облако):** добавляйте `--amp --num-workers 4 --batch-size 128` (включает смешанную точность fp16 и параллельную загрузку).

### 1. Проверка формата реальных данных (5 секунд)
```bash
python -u scripts/quick_check_real_data.py --data /path/to/competition
```

### 2. Полный пайплайн одной командой (проверка + smoke + train + finetune + OOF + сабмит)
```bash
# Вариант для GPU:
python -u scripts/run_full_pipeline.py \
    --data /path/to/competition \
    --amp --scheduler cosine --warmup-epochs 1 \
    --epochs 10 --finetune-epochs 3 --batch-size 128 --num-workers 4 \
    --out models/nn/pipeline_run

# Вариант для CPU:
python -u scripts/run_full_pipeline.py \
    --data /path/to/competition \
    --epochs 10 --finetune-epochs 3 --batch-size 32 --num-workers 0 \
    --out models/nn/pipeline_run
```

### 3. Экспорт нейро-фичей для GBDT Дани (`features/avgust/`)
Обучает модель K-Fold (честный OOF на `train` без утечки таргета) и сохраняет `train.parquet`, `val.parquet`, `test.parquet` с колонками `x_avgust_*`:
```bash
python -u src/nn/export_gbdt_features.py \
    --data /path/to/competition \
    --folds 4 --epochs 8 --batch-size 64 \
    --out features
```

### 4. Взвешенный сид-ансамбль нейросетей (`seed_ensemble.py`)
Обучает 3 модели (сиды 42, 43, 44), подбирает оптимальные веса по `val` и генерирует сабмит:
```bash
python -u src/nn/seed_ensemble.py \
    --data /path/to/competition \
    --seeds 42 43 44 --epochs 10 --batch-size 64 \
    --mixup-alpha 0.2 --temperature 1.5 --agg rank --predict-test \
    --out models/nn/ensemble_v1
```

### 5. Инференс в формате Дани (`val_pred.parquet` / `test_pred.parquet`)
Для подключения в `select_final.py` и `blend.py` Дани используйте флаг `--save-as-danya`:
```bash
python -u src/nn/predict.py --data /path/to/competition --split val --model models/nn/exp/best_model.pt --save-as-danya --out runs/nn-01
python -u src/nn/predict.py --data /path/to/competition --split test --model models/nn/exp/best_model.pt --save-as-danya --out runs/nn-01
```

### 6. Блендинг и стекинг с GBDT (Даня + Август)
```bash
# Ранговый блендинг (подбор весов по сетке на val + сабмит на test):
python -u src/ensemble/blend_nn_gbdt.py \
    --val-nn runs/nn-01/oof_val.parquet \
    --val-gbdt runs/gbdt-01/oof_val.parquet \
    --test-nn runs/nn-01/submission.parquet \
    --test-gbdt runs/gbdt-01/submission.parquet \
    --out runs/ensemble/submission_blend.parquet

# Ridge-стекинг на [nn_score, gbdt_score, f1..f5] с GroupKFold по feed_id:
python -u src/ensemble/stacking.py \
    --data /path/to/competition \
    --val-nn runs/nn-01/oof_val.parquet \
    --val-gbdt runs/gbdt-01/oof_val.parquet \
    --test-nn runs/nn-01/submission.parquet \
    --test-gbdt runs/gbdt-01/submission.parquet \
    --alpha 10.0 \
    --out runs/ensemble/submission_stack.parquet
```

## GBDT-раннер на подготовленных данных Димы
Сначала данные и кеши признаков (см. раздел Димы выше), для N train-шардов:
```powershell
uv run --locked python -m scripts.reproduce_data --raw competition --resume --feature-train-parts 8 --infeed data/features/infeed
```
Затем эксперименты (`--prepared` читает готовые таблицы, `--feature-root` можно повторять,
`--train-parts` — сколько train-шардов, равномерно по времени; кеш должен их покрывать):
```powershell
$F = "--data competition --prepared data/prepared --feature-root data/features/cosine --feature-root data/features/infeed --train-parts 8"
uv run --locked python -m src.gbdt.run_exp $F.Split() --out runs/cb8 --kind catboost --log
uv run --locked python -m src.gbdt.run_exp $F.Split() --out runs/lgb8 --kind lightgbm --log
uv run --locked python -m src.gbdt.run_exp $F.Split() --out runs/lgb8-ipw --kind lightgbm --pos-mode ipw --log
uv run --locked python -m src.gbdt.run_exp $F.Split() --out runs/cb8-cv --kind catboost --mode cv --log
uv run --locked python -m src.ensemble.blend --data competition --runs runs/cb8-cv runs/lgb8-cv --out runs/blend
uv run --locked python scripts/validate_submission.py competition/sample_submission.parquet runs/blend/test_pred.parquet
```
Метрика выбора везде — NDCG@30 (как в эталоне); early stopping — по pseudo-public, оценка — по pseudo-private.
Режим `cv` добавляет val в обучение: val собран так же, как test (без смещения позиции), поэтому это главный кандидат на прирост.

## Признаки истории (группа `history`)
46 признаков связи кандидата с историей пользователя: был ли товар в истории, сколько раз, как давно и с какими
типами событий; доля категории кандидата в истории (вся, последние 10/50, с затуханием, по типам событий);
профиль пользователя по типам событий. Только история самой ленты — без target, pos и других лент.
```powershell
uv run --locked python -m scripts.build_history_features --train-parts 8 --resume
uv run --locked python -m src.gbdt.run_exp --data competition --prepared data/prepared --train-parts 8 `
    --feature-root data/features/cosine --feature-root data/features/infeed --feature-root data/features/history `
    --out runs/cb8-hist --kind catboost --log
```
Каждый кеш должен покрывать конкретные шарды, выбранные раннером через
`--train-parts`. Полный кеш подходит для любой меньшей выборки. Наборы из 8 и 32
шардов выбираются через `linspace` и могут пересекаться частично. Расширение
history сохраняет все проверенные файлы. Кеш проверяет отпечатки таблиц
кандидатов и историй, прогресс сохраняется после каждого шарда.

Аудит проверяет ключи, схемы, конечность чисел и отпечатки всех покрытых файлов.
Все 46 формул независимо пересчитываются на трёх лентах каждого шарда и
дополнительно на лентах с повторными item_id. Таблица событий использует
условие `h_item_evtN > 0`, сохраняет число кандидатов, позитивов и лент,
проверяет весь val и выбранные шарды train. Группы событий могут пересекаться.

```powershell
uv run --locked python -m scripts.audit_history_features --out runs/history_cache_audit.json
uv run --locked python -m scripts.analyse_history_events --train-parts 8 --out experiments/history_event_target_008
```

Для связи событий с категорией кандидата доступен отдельный режим.
Условие `h_cat_evtN > 0` означает, что категория кандидата встречалась
в истории с типом события N. Таблица содержит доли target=1 для этих групп
на выбранном train и полном val. Группы могут пересекаться. На результаты
влияют состав категорий, длина истории и позиция в train. Значения описывают
статистическую связь с целью, их интерпретация требует учёта этих факторов.

```powershell
uv run --locked python -m scripts.analyse_history_events --entity category --train-parts 32 --out experiments/history_category_event_target_032
```

## Статистики товаров и категорий

Группа `itemstats` содержит 32 Float32-признака. Для товара и категории считаются
показы, клики, сглаженный CTR, ожидаемые клики по позициям, сглаженное отношение
наблюдаемых кликов к ожидаемым, число событий в историях и частоты типов 1–10.
Повторные события в историях учитываются как отдельные вхождения.

Для модели на восьми train-шардах источником служат остальные 504 шарда.
По этим же источникам оценивается CTR каждой позиции и общий prior CTR.
Для CTR используется `(clicks + alpha * prior) / (impressions + alpha)`.
Для поправки на позицию используется
`(clicks + alpha * prior) / (expected_clicks + alpha * prior)`.
Значение `alpha` по умолчанию равно 20. Для неизвестных товаров CTR равен prior,
а отношение кликов равно 1. При нулевом знаменателе отношение также равно 1.

Группа предназначена для конкретного набора обучающих шардов. При изменении
набора нужно выбрать отдельную выходную папку, например `itemstats-032`.
Манифест содержит списки исключённых и исходных шардов, параметры сглаживания,
отпечатки всех входов и версию реализации. Присоединение к train проверяет,
что обучающие шарды исключены из источников статистик. Target val/test
используется только при последующей оценке модели.

DuckDB хранит агрегаты и временные данные в папке кеша. Готовые таблицы
и Parquet переиспользуются через `--resume`. Лимит памяти и число потоков
можно менять при возобновлении. Для реального запуска задан лимит 1536 МиБ
и один поток после достижения лимита 1024 МиБ при агрегации историй.

```powershell
uv sync --locked
uv run --locked python -m scripts.build_itemstats_features --train-parts 8 --out data/features/itemstats-008 --memory-mb 1536 --threads 1 --resume
uv run --locked python -m scripts.audit_itemstats_features --cache data/features/itemstats-008 --out runs/itemstats_008_audit.json
```

Для обучающей выборки из 32 шардов источниками служат оставшиеся 480.
Этот набор имеет собственный кеш и собственный список исключений.

```powershell
uv run --locked python -m scripts.build_itemstats_features --train-parts 32 --out data/features/itemstats-032 --memory-mb 2048 --threads 1 --resume
uv run --locked python -m scripts.audit_itemstats_features --cache data/features/itemstats-032 --out runs/itemstats_032_audit.json
```

Кеш подключается к GBDT через `--feature-root data/features/itemstats-008`.
Для модели на 32 шардах используется `--feature-root data/features/itemstats-032`.
Аудит проверяет все строки кеша и раздельность источников. Для трёх кандидатов
каждого файла все признаки независимо пересчитываются по всем исходным
donor-шардам через Polars и Python. Полнота численной проверки ограничена
этой выборкой.

## Пользовательские статистики

Группа `userstats` содержит 16 признаков `is_user_*` по тем же формулам,
что статистики товаров. Для пользователя считаются показы, клики, сглаженный
CTR, ожидаемые клики по позициям, сглаженное отношение кликов к ожидаемым,
общее число событий в донорских историях и частоты типов 1–10. Счётчики
суммируют вхождения событий в историях донорских лент. Повторяющиеся
события учитываются при каждом вхождении.

Для неизвестного пользователя используются нулевые счётчики, общий prior CTR
и нейтральное отношение кликов 1. Alpha по умолчанию равно 20. Кеши для
8 и 32 обучающих шардов имеют отдельные папки и списки исключений.
Источниками служат соответственно остальные 504 и 480 train-шардов.

```powershell
uv run --locked python -m scripts.build_userstats_features --train-parts 8 --out data/features/userstats-008 --memory-mb 512 --threads 1 --resume
uv run --locked python -m scripts.build_userstats_features --train-parts 32 --out data/features/userstats-032 --memory-mb 512 --threads 1 --resume
uv run --locked python -m scripts.audit_userstats_features --cache data/features/userstats-008 --out runs/userstats_008_audit.json
uv run --locked python -m scripts.audit_userstats_features --cache data/features/userstats-032 --out runs/userstats_032_audit.json
```

Группа подключается через соответствующий `--feature-root`. Проверяется
исключение обучающих шардов из источников. Аудит сверяет ключи и числа всех
строк, соответствие пользователя между кандидатами и историей каждой
донорской ленты, а также независимо пересчитывает признаки для трёх
кандидатов каждого выходного файла по всем исходным шардам.

## Воспроизведение модели одной командой

После установки зависимостей из `uv.lock` и распаковки исходного архива
общая команда готовит таблицы, EDA, временное разбиение val, каталог
эмбеддингов, выбранные группы признаков и CatBoost. Затем она проверяет
метрики по эталону, загружает сохранённую модель для предсказания test
и запускает валидатор сабмита.

```powershell
uv run --locked python -m scripts.reproduce_ranking --raw competition --train-parts 8 --infeed --history --itemstats --out runs/reproduced-joint-008 --resume
```

Для воспроизведения варианта с пользовательскими статистиками добавьте
`--userstats` и задайте новое имя в `--out`. Без флагов дополнительных групп
используется cosine. Число обучающих шардов общее для кешей и модели.
Для статистик оставляются отдельные донорские шарды, поэтому этот режим
требует хотя бы одного train-шарда сверх обучающей выборки.

`--iterations`, `--depth`, `--seed` и `--threads` задают параметры модели.
По умолчанию это 100 итераций, глубина 6, seed 42 и два CPU-потока.
Чекпойнт выбирается по pseudo-Public. `--memory-mb` задаёт лимит
агрегаций DuckDB, по умолчанию 2048 МиБ при одном потоке.

В папке прогона сохраняются `model.cbm`, `run.json`, `submission.parquet`,
а также `val_pred.parquet` и `test_pred.parquet` в командном формате для
ансамбля. Файл `pipeline.json` фиксирует группы, отпечатки предсказаний
и завершение всех этапов. `--resume` переиспользует проверенные данные
и признаки. Для нового обучения нужна новая папка `--out`.

Команда проверена сквозным синтетическим тестом с восемью группами,
138 признаками, раздельными донорскими шардами и повторным запуском.
Конфигурация окончательной командной модели будет зафиксирована после
её выбора.

## Совместная встречаемость категорий

Группа `category_cooc` — первый категорийный вариант item2item по историям.
Для каждой пары категорий считается число донорских лент, в историях которых
присутствуют обе категории. Повторные события одной категории в донорской
истории дают одно присутствие. Матрица симметрична.

Оценка пары равна `C(i,j) / sqrt(C(i,i) * C(j,j))`. Для одинаковых известных
категорий она равна 1, для категории без наблюдений — 0. Четыре признака
`cc_sum_10`, `cc_max_10`, `cc_sum_50`, `cc_max_50` суммируют и максимизируют
эту оценку по первым 10 и 50 событиям истории текущей ленты. Порядок событий
и повторения в этих окнах сохраняются.

Доноры исключают все выбранные обучающие шарды. Кеши на 8 и 32 шарда
имеют отдельные папки. Загрузчик проверяет соответствие обучающей выборки.
Агрегаты сохраняются после каждого донорского шарда, признаки — после
каждого выходного файла. Числа всех пар независимо проверяются пересечениями
битовых наборов донорских лент. Формулы проверяются на трёх лентах каждого
файла, а ключи, типы, диапазоны и отпечатки — на всех строках.

```powershell
uv run --locked python -m scripts.build_category_cooc_features --train-parts 8 --out data/features/category-cooc-008 --resume
uv run --locked python -m scripts.build_category_cooc_features --train-parts 32 --out data/features/category-cooc-032 --resume
uv run --locked python -m scripts.audit_category_cooc_features --cache data/features/category-cooc-008 --out runs/category_cooc_008_audit.json
uv run --locked python -m scripts.audit_category_cooc_features --cache data/features/category-cooc-032 --out runs/category_cooc_032_audit.json
```

Подключение выполняется через соответствующий `--feature-root`.

## Cosine по типам событий

Группа `event_cosine` использует последние 50 исходных событий истории и
типы 1, 2 и 10. Для каждого типа сохраняются число доступных векторов,
среднее и максимум cosine с кандидатом, всего девять Float32-признаков.
Типы 1 и 2 выбраны как частые события train, тип 10 включён по запросу Дани.
Окно отсекается до фильтрации по типу и наличию вектора. При отсутствии
подходящих векторов среднее и максимум равны 0, отрицательный максимум
сохраняется. Используются исходные векторы каждого split и маски `-1`.

Кеш расширяется с сохранением проверенных файлов. Выборки на 8 и 32 шарда
имеют частичное пересечение, поэтому общий кеш содержит их объединение.
Весь val/test вычисляется один раз. `--prefetch-parts` объединяет чтение
векторов для нескольких шардов и по умолчанию равно 4.

```powershell
uv run --locked python -m scripts.build_event_cosine_features --train-parts 8 --resume
uv run --locked python -m scripts.build_event_cosine_features --train-parts 32 --resume
uv run --locked python -m scripts.audit_event_cosine_features --train-parts 8 --out runs/event_cosine_008_audit.json
uv run --locked python -m scripts.audit_event_cosine_features --train-parts 32 --out runs/event_cosine_032_audit.json
```

## Ранги сильных признаков

Группа `feature_ranks` добавляет нормированные ранги `cos_user`,
`cos_hist_max_10`, `cos_hist_max_50` и `is_item_ctr`. Ранг возрастающий,
при равенстве используется средний ранг, делённый на размер ленты.
Ранг `h_cat_share` уже доступен в history как `h_cat_share_feedrank`.
CTR берётся из itemstats с тем же исключением обучающих шардов.
Отдельные кеши содержат результаты для когорт 8 и 32.

```powershell
uv run --locked python -m scripts.build_feature_ranks --train-parts 8 --itemstats data/features/itemstats-008 --out data/features/feature-ranks-008 --resume
uv run --locked python -m scripts.build_feature_ranks --train-parts 32 --itemstats data/features/itemstats-032 --out data/features/feature-ranks-032 --resume
uv run --locked python -m scripts.audit_feature_ranks --itemstats data/features/itemstats-008 --cache data/features/feature-ranks-008 --out runs/feature_ranks_008_audit.json
uv run --locked python -m scripts.audit_feature_ranks --itemstats data/features/itemstats-032 --cache data/features/feature-ranks-032 --out runs/feature_ranks_032_audit.json
```

## Время и возраст товара

Группа `time` содержит час UTC (0–23), день недели UTC (понедельник 0),
число дней после первого появления товара среди кандидатов донорского train
и флаг наличия такого прошлого появления. Это наблюдаемый возраст в данных.
Используется минимальный timestamp донорского кандидата строго раньше
текущей ленты. Если прошлое появление отсутствует, возраст и флаг равны 0.
Идентификаторы товаров 0 и отрицательные значения получают этот же вариант.
Обучающие шарды исключаются из доноров, target используется только моделью.

```powershell
uv run --locked python -m scripts.build_time_features --train-parts 8 --out data/features/time-008 --resume
uv run --locked python -m scripts.build_time_features --train-parts 32 --out data/features/time-032 --resume
uv run --locked python -m scripts.audit_time_features --cache data/features/time-008 --out runs/time_008_audit.json
uv run --locked python -m scripts.audit_time_features --cache data/features/time-032 --out runs/time_032_audit.json
```

Для новых кешей действует исключительная блокировка записи, которая
освобождается операционной системой при завершении процесса. Повторный
запуск переиспользует проверенные части. Аудит проверяет все строки
структурно и пересчитывает формулы на трёх лентах каждого файла независимо
от реализации расчёта признаков. Возраст сверяется по всем донорским шардам
для товаров из этих лент.

Все группы подключаются через `--feature-root`. В `scripts.reproduce_ranking`
доступны флаги `--category-cooc`, `--event-cosine`, `--feature-ranks`,
`--time-features` и `--item2item`. Для рангов itemstats строится как зависимость. Полный
набор его колонок добавляется в модель при явном флаге `--itemstats`.

## Item2item по историям товаров

Группа `item2item` содержит четыре Float32-признака. Для последних 10 и 50
событий истории вычисляются сумма и максимум совместной встречаемости товара
кандидата с товарами истории. Сходство пары равно числу донорских лент,
содержащих оба товара в истории, делённому на квадратный корень из произведения
чисел лент для каждого товара. Повторы товара внутри донорской истории
учитываются один раз. Повторы в истории текущего пользователя сохраняются.
Окно отсекается до обработки отсутствующих товаров. Совпадение товара с собой
имеет сходство 1 при наличии донорских наблюдений. Неизвестные товары и
неположительные идентификаторы дают 0. Target в расчёте отсутствует.

Для каждой когорты из доноров исключаются все её train-шарды. Индекс хранит
отсортированные списки донорских лент для нужных товаров. Запросы используют
разреженные произведения с локальным сжатием идентификаторов лент. Индекс
сохраняется в трёх массивах NumPy, читаемых через mmap. DuckDB ограничивает
память построения и сохраняет таблицы для возобновления после прерывания.
Файлы признаков сохраняются после каждого шарда. Кеши 8 и 32 независимы.

Аудит проверяет все строки признаков, структуру всего индекса и независимо
собирает списки донорских лент для товаров из трёх проверяемых лент каждого
файла. Для этих товаров просмотрены все донорские истории. Формулы
пересчитываются пересечениями множеств, с допусками rtol/atol=2e-6.

```powershell
uv run --locked python -m scripts.build_item2item_features --train-parts 8 --out data/features/item2item-008 --resume
uv run --locked python -m scripts.build_item2item_features --train-parts 32 --out data/features/item2item-032 --resume
uv run --locked python -m scripts.audit_item2item_features --cache data/features/item2item-008 --out runs/item2item_008_audit.json
uv run --locked python -m scripts.audit_item2item_features --cache data/features/item2item-032 --out runs/item2item_032_audit.json
```

По умолчанию `--memory-mb=1536` ограничивает DuckDB, а
`--candidate-batch=64` задаёт размер запроса к индексу. Группа подключается
через соответствующий `--feature-root` или флаг `--item2item` в
`scripts.reproduce_ranking`. Зависимость SciPy зафиксирована в uv.lock.

## Четыре временных OOF-фолда для joint + item2item

Конфигурация содержит прежние 105 признаков joint-008 и четыре item2item,
всего 109. Для сравнения зафиксированы параметры joint-008 и его 92 дерева,
выбранные ранее по pseudo-Public. Настройка `--folds 4` задаёт четыре
последовательных временных блока val. Каждый блок предсказывается моделью,
обученной на восьми train-шардах и остальных трёх блоках val. Все кандидаты
одной ленты попадают в один фолд. Это блочный OOF с обучением на остальных
блоках. Донорские признаки itemstats и item2item используют исходный train с
исключением всех восьми обучающих шардов и сохраняются одинаковыми для фолдов.

Подготовка данных и пяти групп признаков выполняется после распаковки
исходного архива в `competition`. Флаг `--resume` переиспользует совпадающие
готовые файлы.

```powershell
uv sync --locked
uv run --locked python -m scripts.reproduce_data --raw competition --feature-train-parts 8 --infeed data/features/infeed --resume
uv run --locked python -m scripts.build_history_features --train-parts 8 --resume
uv run --locked python -m scripts.build_itemstats_features --train-parts 8 --out data/features/itemstats-008 --resume
uv run --locked python -m scripts.build_item2item_features --train-parts 8 --out data/features/item2item-008 --resume
uv run --locked python -m scripts.audit_item2item_features --cache data/features/item2item-008 --out runs/item2item_008_audit.json
```

Запуск четырёх моделей выполняет владелец основного обучения. Папка вывода
должна быть отдельной для каждого эксперимента. `used_ram_limit` задаёт
ориентир CatBoost, общий расход памяти включает загруженные таблицы и пулы.

```powershell
$cvOut = 'runs/dima-joint-item2item-008-cv'
New-Item -ItemType Directory -Path $cvOut | Out-Null
@{ thread_count = 2; used_ram_limit = '1024mb' } | ConvertTo-Json | Set-Content -Encoding ascii "$cvOut/catboost-params.json"
uv run --locked python -m src.gbdt.run_exp --data competition --prepared data/prepared --train-parts 8 --feature-root data/features/cosine --feature-root data/features/infeed --feature-root data/features/history --feature-root data/features/itemstats-008 --feature-root data/features/item2item-008 --kind catboost --mode cv --folds 4 --iterations 92 --depth 6 --lr 0.1 --seed 42 --model-params "$cvOut/catboost-params.json" --out $cvOut
uv run --locked python scripts/validate_submission.py competition/sample_submission.parquet "$cvOut/test_pred.parquet"
```

На выходе получаются `val_pred.parquet` с OOF-score и target для всех
1 389 971 кандидатов val, `test_pred.parquet` для 2 904 806 кандидатов test
и `metrics.json`. Test-score усредняется по моделям после z-score внутри
каждой ленты. Для ансамбля используются эти два Parquet, веса выбираются
по pseudo-Public. Метрики и статус Public платформы заносятся владельцем
прогона в `experiments/log.md`.

### Последовательный OOF с дисковыми пулами

`scripts.run_joint_oof` обрабатывает те же временные фолды последовательно.
Объединённые признаки сохраняются по шардам, обучающий TSV сортируется через
DuckDB с ограничением памяти. Для каждого фолда CatBoost читает один
обучающий набор, квантует и сохраняет его на диск. Предсказания val/test
считаются по шардам. Чтение исходного пула и обучение расходуют дополнительную
память сверх лимита менеджера DuckDB.

Сохраняются модель каждого фолда, контрольные точки CatBoost и промежуточные
предсказания. `--resume` проверяет источники и переиспользует законченные
фолды. При прерывании после сохранения модели повторяется расчёт её
предсказаний. Число итераций фиксировано, held-out блок исключается при
подготовке обучающего пула. Метрики сверяются с эталоном, итоговый test
проверяется валидатором. Параметры и отпечатки входят в `run.json`.

```powershell
uv run --locked python -m scripts.run_joint_oof --train-parts 8 --folds 4 --iterations 92 --depth 6 --learning-rate 0.1 --seed 42 --threads 2 --memory-mb 512 --feature-root data/features/cosine --feature-root data/features/infeed --feature-root data/features/history --feature-root data/features/itemstats-008 --feature-root data/features/item2item-008 --out runs/dima-joint-item2item-008-cv-001 --resume
```

Итоговые `val_pred.parquet` и `test_pred.parquet` имеют командный формат.
Дополнительно сохраняются ключи вхождений кандидатов и назначение фолдов.
`test_pred` усредняет четыре предсказания с z-score внутри каждой ленты.

Описательные срезы готового OOF по длине истории, размеру ленты и покрытию
эмбеддингами можно получить следующей командой. Для test сохраняются
характеристики покрытия.

```powershell
uv run --locked python -m scripts.analyse_oof_slices --run runs/dima-joint-item2item-008-cv-001 --out runs/joint_item2item_oof_slices_008.json
```

При готовых item2item-008 и weighted-item2item-008 можно сравнить
однофакторные сигналы разных нормировок на фиксированной Public-подвыборке.

```powershell
uv run --locked python -m scripts.analyse_item2item_normalization --out runs/item2item-normalization-diagnostics.json
```

## Item2item с весами истории

Группа `weighted_item2item` использует проверенный индекс `item2item-008`
или `item2item-032`. Для окон 10/50 рассчитываются среднее и максимум
сходства с весами `2^(-position/10)`. Три канала используют всю историю,
только события типа 2 и только события типа 10, всего 12 признаков.
Веса событий заданы индикаторами принадлежности каналу. Среднее делится
на сумму весов всей истории соответствующего канала, включая неизвестные
товары. Это нормирует сигнал на взвешенную длину истории. Нормировка на
популярность кандидата и товара истории сохраняется из исходного индекса.

Индивидуальные timestamps событий в данных отсутствуют. Давность здесь
определяется позицией события, период полураспада равен десяти позициям.
Веса применяются к истории запроса, донорские совместные встречаемости
берутся из родительского индекса. Повторы и исходный порядок сохраняются.
Неизвестные товары и пустые каналы дают 0.

```powershell
uv run --locked python -m scripts.build_weighted_item2item_features --parent data/features/item2item-008 --train-parts 8 --out data/features/weighted-item2item-008 --resume
uv run --locked python -m scripts.build_weighted_item2item_features --parent data/features/item2item-032 --train-parts 32 --out data/features/weighted-item2item-032 --resume
uv run --locked python -m scripts.audit_weighted_item2item_features --parent data/features/item2item-008 --parent-audit experiments/item2item_008_audit.json --cache data/features/weighted-item2item-008 --out runs/weighted_item2item_008_audit.json
uv run --locked python -m scripts.audit_weighted_item2item_features --parent data/features/item2item-032 --parent-audit experiments/item2item_032_audit.json --cache data/features/weighted-item2item-032 --out runs/weighted_item2item_032_audit.json
```

Аудит требует совпадения родительского manifest с его независимым аудитом.
Веса и формулы пересчитываются скалярно через пересечения списков лент,
ключи и числовые диапазоны проверяются на всех строках нового кеша.

## Cosine с затуханием по истории

Группа `recency_cosine` содержит четыре признака. Для исходных окон 10/50
вычисляются взвешенный средний cosine и cosine с нормированным взвешенным
профилем истории. Веса равны `2^(-position/10)`. Профиль собирается из
единичных векторов товаров истории. Среднее делится на сумму весов
доступных векторов. Пропуски сохраняют исходные позиции, повторы учитываются
повторно. Нулевые профили и недоступные векторы кандидатов дают 0.
Используются исходные версии векторов каждого split.

```powershell
uv run --locked python -m scripts.build_recency_cosine_features --train-parts 8 --resume
uv run --locked python -m scripts.build_recency_cosine_features --train-parts 32 --resume
uv run --locked python -m scripts.audit_recency_cosine_features --train-parts 8 --out runs/recency_cosine_008_audit.json
uv run --locked python -m scripts.audit_recency_cosine_features --train-parts 32 --out runs/recency_cosine_032_audit.json
```

Кеш объединяет нужные train-шарды и содержит весь val/test.
`--prefetch-parts` по умолчанию равен 2. Аудит независимо пересчитывает
формулы в Float64 по исходным векторам на трёх лентах каждого файла,
структурные проверки охватывают все строки.

## Расширенный контекст ленты

Группа `feed_context` содержит 34 признака. Для f1–f5 добавлены отклонения
от максимума и среднего ленты. Для `cos_user`, `is_item_ctr` и четырёх
item2item-сигналов добавлены отклонения и z-score с выборочным стандартным
отклонением. Для item2item также добавлены средние ранги, делённые на размер
ленты. Ранги cosine/CTR уже доступны в `feature_ranks`, z-score и ранги
f1–f5 доступны в `infeed`. Два взаимодействия равны `cos_user * is_item_ctr`
и `cos_hist_max_50 * is_item_ctr`.

При постоянном признаке и в ленте из одного кандидата z-score равен 0.
Вычисления используют Float64, результат сохраняется в Float32 с насыщением
диапазона для экстремальных значений. Донорские когорты itemstats и item2item
должны совпадать с выбранными train-шардами.

```powershell
uv run --locked python -m scripts.build_feed_context_features --itemstats data/features/itemstats-008 --item2item data/features/item2item-008 --train-parts 8 --out data/features/feed-context-008 --resume
uv run --locked python -m scripts.build_feed_context_features --itemstats data/features/itemstats-032 --item2item data/features/item2item-032 --train-parts 32 --out data/features/feed-context-032 --resume
uv run --locked python -m scripts.audit_feed_context_features --itemstats data/features/itemstats-008 --item2item data/features/item2item-008 --cache data/features/feed-context-008 --out runs/feed_context_008_audit.json
uv run --locked python -m scripts.audit_feed_context_features --itemstats data/features/itemstats-032 --item2item data/features/item2item-032 --cache data/features/feed-context-032 --out runs/feed_context_032_audit.json
```

## Иерархический и свежий CTR

Сглаженный CTR с глобальным prior уже входит в itemstats. Новая группа
`bayesian_ctr` содержит 14 признаков. Для всей доступной донорской истории
и окон 3/7 дней сохраняются CTR категории, CTR товара с prior категории,
разность CTR товара и категории и `log1p` числа показов товара. Сглаживание
на обоих уровнях по умолчанию равно 20. Глобальная частота окна задаёт prior
категории, её posterior задаёт prior товара. Холодный товар получает posterior
категории. При отсутствии прошлых донорских наблюдений CTR равен 0.

Обучающие шарды исключены из доноров. Используются завершённые UTC-дни
строго раньше дня запроса. Свежие окна заканчиваются в
`min(query_day - 1, last_donor_day)`. Это сохраняет последние дни доноров
для val/test, где даты запроса находятся позже окончания train. Добавлены
число дней после последнего допустимого донорского дня и флаг наличия
прошлых наблюдений. По данным соревнования train заканчивается 19.09,
val охватывает 21–24.09, test — 25–29.09. Даты приведены в UTC.

```powershell
uv run --locked python -m scripts.build_bayesian_ctr_features --train-parts 8 --out data/features/bayesian-ctr-008 --resume
uv run --locked python -m scripts.build_bayesian_ctr_features --train-parts 32 --out data/features/bayesian-ctr-032 --resume
uv run --locked python -m scripts.audit_bayesian_ctr_features --cache data/features/bayesian-ctr-008 --out runs/bayesian_ctr_008_audit.json
uv run --locked python -m scripts.audit_bayesian_ctr_features --cache data/features/bayesian-ctr-032 --out runs/bayesian_ctr_032_audit.json
```

Дневные таблицы и lookup-файлы имеют контрольные точки для resume. Аудит
заново считает общие и категорийные наблюдения всех доноров и дневные
наблюдения для проверяемых товаров. Posterior и окна пересчитываются
независимыми суммами. Ключи, диапазоны и отпечатки проверяются полностью.

Для этой группы `--memory-mb` по умолчанию равен 1536. Manifest явно
фиксирует использование прошлых донорских target, данные запроса задают
ключи и timestamps. Суммы по окнам и присоединение lookup выполняются по
одному дню запросов. Это ограничивает объём одновременно агрегируемых
ключей. Дневные счётчики доноров переиспользуются для всех дат.

## Бэггинг на непересекающихся train-блоках

`--train-start` задаёт начальную позицию в списке train-шардов manifest,
нумерация начинается с нуля. Вместе с `--train-parts 32` значения 0, 32, 64
выбирают блоки 0..31, 32..63, 64..95. При отсутствии `--train-start`
сохраняется выбор linspace. Каждый из этих блоков охватывает временной
диапазон train. Непересечение источников и feed_id проверяется отдельно.

Параметр поддерживается сборщиками cosine, infeed, history, itemstats,
item2item, weighted item2item и дисковым OOF-раннером. Для каждого блока
донорские статистики и индекс используют остальные 480 train-шардов.
Кеши разных блоков размещаются в отдельных каталогах. Cosine и infeed
могут использовать общий кеш с покрытием всех выбранных блоков.

Общий раннер `python -m src.gbdt.run_exp` также принимает `--train-start`
вместе с `--prepared` для CatBoost и LightGBM. Он загружает выбранный блок
в память. Дисковый `scripts.run_joint_oof` готовит обучающие пулы по файлам.

Если общего cosine/infeed на первые 96 шардов ещё нет, его можно собрать
следующими командами. Эмбеддинг-каталог и prepared предварительно создаются
через `scripts.reproduce_data`.

```powershell
uv run --locked python -m scripts.build_similarity_features --train-start 0 --train-parts 96 --out data/features/cosine-blocks-096 --prefetch-parts 2 --resume
uv run --locked python -m scripts.build_infeed_features --train-start 0 --train-parts 96 --out data/features/infeed-blocks-096 --resume
```

Далее выполняется предварительная проверка покрытия и конфигурации.
В примере используются указанные выше общие кеши. Для уже полного кеша
можно передать `data/features/cosine` и `data/features/infeed`.

```powershell
uv run --locked python -m scripts.prepare_train_blocks --cosine data/features/cosine-blocks-096 --infeed data/features/infeed-blocks-096 --out runs/train-blocks-prepare-001 --starts 0 32 64 --train-parts 32 --weighted-item2item --plan-only
uv run --locked python -m scripts.prepare_train_blocks --cosine data/features/cosine-blocks-096 --infeed data/features/infeed-blocks-096 --out runs/train-blocks-prepare-001 --starts 0 32 64 --train-parts 32 --weighted-item2item --resume
```

Подготовка последовательно строит history, itemstats, item2item и
weighted item2item, выполняет независимые аудиты и проверяет загрузку
реального train/val/test-файла для каждого блока. Конфигурация содержит
121 признак, при отключении `--weighted-item2item` — 109. Прогресс,
аудиты и OOF-команды сохраняются в `runs/train-blocks-prepare-001`.
`--resume` переиспользует проверенные этапы. Повторная проверка полностью
готового плана сохраняет его содержимое и SHA-256.

Для владельца обучения доступна последовательная команда OOF.

```powershell
uv run --locked python -m scripts.run_train_blocks_oof --plan runs/train-blocks-prepare-001/run.json
```

Три блока и четыре фолда дают 12 моделей. Блоки обучаются в отдельных
последовательных процессах, память процесса освобождается перед следующим
блоком. Параметры по умолчанию соответствуют joint-008, depth=6, lr=0.1,
seed=42, 92 итерации. `--only block-000-032` выбирает один готовый блок.
Каждый блок выдаёт OOF val и полный test с одинаковыми ключами кандидатов.
Итоговые модели и предсказания записываются в `runs/dima-blocks-oof`.
Перед обучением повторно проверяются кеши, входные отпечатки и аудиты.
Оценка памяти входного квантованного пула приведена в
[расчёте блоков](experiments/train_block_bagging_feasibility_001.json).

После агрегации itemstats сохраняет отсортированные lookup-массивы товаров
и категорий. Признаки кандидатов извлекаются прямым поиском, значения
сохраняются в Float32. Массивы входят в отпечатки артефактов кеша.
Готовый `--resume` проверяет входы и выходы и возвращает исходный manifest.
Поддерживаются известные SQL-реализации с совпадающими формулами агрегации
и lookup. При восстановлении частичного кеша для каждого Parquet
сохраняется SHA-256 реализации, которая его произвела.

Побитовое сопоставление со всеми строками готового SQL-кеша и замеры
материализации выполняются следующей командой.

```powershell
uv run --locked python -m scripts.benchmark_itemstats_lookup --cache data/features/train-blocks/block-000-032/itemstats --out runs/itemstats-lookup-check
```

Команда сохраняет временные lookup-массивы и отчёт в указанном каталоге.
Реальный результат проверки 19,6 млн строк опубликован в
[отчёте](experiments/itemstats_lookup_benchmark_001.json).
