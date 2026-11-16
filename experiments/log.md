# Журнал экспериментов

Все числа — только на реальных данных. Синтетика проверяет код, а не качество.
Псевдо-Public/Private — временные части локального val; колонка Public относится к платформе соревнования.

| Дата | Автор | Коммит кода | Что сделано | val NDCG (full / @10 / @30) | Псевдо-Public (full / @30) | Псевдо-Private (full / @30) | Public платформы | Примечание |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11.11.2026 | Дима | `0eb8c43` | CatBoost YetiRank; f1–f5 + категория; 1 train-шард, 476,301 кандидатов / 6,132 лент; 60 деревьев, seed=42 | 0.328563 / 0.188423 / 0.259971 | 0.331998 / 0.265654 | 0.327090 / 0.257534 | — | Весь val: 14,423 лент. Чекпойнт выбран по псевдо-Public; Private не передавался в fit. Сверка с эталоном и валидатор сабмита пройдены. |
| 11.11.2026 | Дима | `ff159ef` | Тот же CatBoost + cosine-группа (9 признаков), тот же train-шард; 34 дерева, seed=42 | 0.357691 / 0.225219 / 0.295816 | 0.363805 / 0.304976 | 0.355070 / 0.291889 | — | Весь val; выбор итерации только на псевдо-Public. Эталон и валидатор пройдены; новая группа на исходных версиях split. |
| 11.11.2026 | Дима | `da4c0a6` | Cosine + 12 признаков внутри ленты; тот же train-шард, 40 деревьев, seed=42 | 0.356414 / 0.225054 / 0.294685 | 0.363243 / 0.304885 | 0.353487 / 0.290312 | — | Эталон и валидатор пройдены. Улучшение не подтверждено, группа опциональная. |

| 12.11.2026 | Дима | `f948754` | Cosine, 2 train-шарда; 958,087 кандидатов / 12,265 лент; 80 дерева, seed=42 | 0.359267 / 0.227615 / 0.297908 | 0.365115 / 0.307071 | 0.356759 / 0.293979 | — | Расширение train при тех же признаках/параметрах. Весь val, выбор чекпойнта только по псевдо-Public. Эталон и валидатор пройдены. |

| 13.11.2026 | Дима | `3f9c669` | Cosine + 32 itemstats; 1 train-шард, 42 дерева; статистики из 504 других шардов, alpha=20 | 0.360929 / 0.231232 / 0.299849 | 0.368352 / 0.310286 | 0.357746 / 0.295374 | — | Выбор чекпойнта по pseudo-Public. Эталон и валидатор пройдены. Источники статистик исключают 8 шардов, включая обучающий. |
| 13.11.2026 | Дима | `6c20d45` | Cosine + itemstats + 16 userstats; тот же 1 train-шард, 63 признака, 41 дерево; alpha=20 | 0.361030 / 0.230818 / 0.300081 | 0.368273 / 0.310417 | 0.357925 / 0.295649 | — | Выбор чекпойнта по pseudo-Public. Эталон и валидатор пройдены. Прирост от userstats не подтверждён: 95% интервал разности включает ноль. |
| 13.11.2026 | Дима | `43c13cd` | Описательный анализ событий по категории кандидата на 32 train-шардах и всём val | — | — | — | — | Все десять групп h_cat_evtN > 0 содержат наблюдения. |
| 13.11.2026 | Дима | `055f644` | Cosine + infeed + history + itemstats, 8 train-шардов, 105 признаков, 92 дерева | 0.363236 / 0.233965 / 0.302709 | 0.369146 / 0.311774 | 0.360702 / 0.298823 | — | Чекпойнт выбран по pseudo-Public. Эталон и валидатор пройдены. |
| 13.11.2026 | Дима | `1a2e241` | Тот же набор и 8 шардов с добавлением userstats, 121 признак, 30 деревьев | 0.362064 / 0.232770 / 0.301274 | 0.368073 / 0.310522 | 0.359489 / 0.297308 | — | Добавление userstats дало разность Private -0.001514, 95% интервал [-0.003391, 0.000188]. |
| 15.11.2026 | Дима | `12fbd63` | Cosine + itemstats + 4 category_cooc, 1 train-шард, 51 признак, 19 деревьев | 0.358392 / 0.228104 / 0.297019 | 0.367835 / 0.309221 | 0.354344 / 0.291788 | — | Парное сравнение с cosine + itemstats. Разность Private -0.003586, 95% интервал [-0.005724, -0.001622]. |
| 15.11.2026 | Дима | `0abfc5e` | Cosine + itemstats + event_cosine, 1 train-шард, 56 признаков, 26 деревьев | 0.361203 / 0.230450 / 0.299675 | 0.370352 / 0.311196 | 0.357281 / 0.294735 | — | Разность Private к тому же контролю -0.000639, 95% интервал [-0.002739, 0.001313]. |
| 15.11.2026 | Дима | `0abfc5e` | Cosine + itemstats + feature_ranks, 1 train-шард, 51 признак, 41 дерево | 0.358649 / 0.228131 / 0.297069 | 0.364648 / 0.305674 | 0.356077 / 0.293380 | — | Разность Private -0.001994, 95% интервал [-0.004546, 0.000332]. |
| 15.11.2026 | Дима | `0abfc5e` | Cosine + itemstats + time, 1 train-шард, 51 признак, 20 деревьев | 0.359153 / 0.228357 / 0.298057 | 0.366723 / 0.308271 | 0.355908 / 0.293678 | — | Разность Private -0.001696, 95% интервал [-0.004099, 0.000624]. |

Подробные параметры, хеши кода и данных, метрики и результат проверки: [baseline_check_001.json](baseline_check_001.json). Коммит в таблице — чистый коммит, на котором выполнялся прогон.

Воспроизведение из корня проекта после подготовки данных:

```powershell
.\.venv\Scripts\python.exe -u -m scripts.run_baseline_check --data data/prepared --raw competition --out runs/baseline-check-001 --train-parts 1 --iterations 100 --depth 6 --seed 42 --threads 2
```

Если папка прогона уже существует, задайте новое имя в `--out`. Модель и предсказания находятся локально в `runs/baseline-check-001/`. Это проверочный прогон на 0,193% кандидатов train; обучение на полном train и OOF на val здесь не выполнялись.


## Cosine-группа: сравнение с baseline

Тот же train-срез (476 301 кандидат / 6 132 ленты), параметры и разбиение val. NDCG@30 на псевдо-Private: 0.257534 → 0.291889; прирост 0.034355 (13.34% относительно baseline). Парный bootstrap по 10 095 лентам, 1000 повторов, seed=42: 95% интервал разности [0.030372, 0.038557]. Это оценка на фиксированном val для моделей с малым train, а не результат обучения на полном train.

[Параметры cosine-прогона](baseline_cosine_001.json), [парное сравнение](cosine_comparison_001.json), [полный аудит эмбеддингов](embedding_catalogue_audit.json).

```powershell
.\.venv\Scripts\python.exe -u -m scripts.run_baseline_check --data data/prepared --raw competition --feature-root data/features/cosine --out runs/baseline-cosine-001 --train-parts 1 --iterations 100 --depth 6 --seed 42 --threads 2
.\.venv\Scripts\python.exe -m scripts.compare_baselines --baseline runs/baseline-check-001 --candidate runs/baseline-cosine-001 --out runs/baseline-cosine-001/comparison.json
```

Cosine-кеш покрывает один train-шард и весь val/test. Полные исходные таблицы и общий каталог построены для всех split. Расширение train-кеша запускается отдельно; текущий эксперимент использует только заявленное покрытие.


## Признаки внутри ленты: сравнение с cosine-моделью

При совпадающих train-срезе, параметрах и val-назначениях NDCG@30 псевдо-Private: 0.291889 → 0.290312. Разность -0.001577 (-0.54%). Парный bootstrap по лентам, 1000 повторов: 95% интервал [-0.004025, 0.000615]. Улучшение на этом малом train не подтверждено. Проверочный baseline по умолчанию остаётся cosine-моделью; группа статистик доступна отдельно для экспериментов на других объёмах train и моделях.

[Параметры прогона](baseline_infeed_001.json), [парное сравнение](infeed_comparison_001.json). Предсказания — holdout, не OOF.

```powershell
uv run --locked python -m scripts.run_baseline_check --feature-root data/features/cosine --feature-root data/features/infeed-check --out runs/baseline-infeed-001
uv run --locked python -m scripts.compare_baselines --baseline runs/baseline-cosine-001 --candidate runs/baseline-infeed-001 --out runs/baseline-infeed-001/comparison.json
```


## Полный кеш и сквозная воспроизводимость

Признаки внутри ленты материализованы для всех 640 файлов: 246 745 836 кандидатов train, 1 389 971 val, 2 904 806 test; всего 251 040 613 строк, около 7,1 ГиБ. Проверены число строк, схемы и отпечатки каждого файла; формулы дополнительно сверены независимо на 22 реальных лентах, включая ленту с повторным товаром. [Аудит полного кеша](infeed_cache_audit.json).

Единая команда `scripts.reproduce_data --baseline-out` проверена на реальных данных с переиспользованием проверенных кешей. Cosine-baseline обучен заново на одном train-шарде. Схемы, ключи и все значения val/test-предсказаний совпали с исходным прогоном; Float32-скоры совпали побитово. Метрики совпали, валидатор пройден. [Проверка воспроизводимости](reproduction_check_001.json).

```powershell
uv run --locked python -m scripts.reproduce_data --raw competition --resume --baseline-out runs/reproduced-baseline-001
```

Обычная команда не материализует необязательную группу внутри ленты. Для её полного построения или проверки существующего кеша добавьте `--infeed data/features/infeed`. Default-модель остаётся cosine-baseline; негативный/неопределённый результат новой группы не скрывается и не заменяет принятый baseline.


## Расширение train до двух шардов — 12.11.2026

Cosine-кеш покрывает 2/512 train-шардов (958,087 кандидатов), а также весь val/test. Проверены число строк и отпечатки всех 130 файлов. Запуск `--train-parts 1 --resume` сохраняет оба готовых train-шарда и не перезаписывает Parquet. [Аудит покрытия](cosine_cache_audit_002.json).

Чтение большого train-memmap заменено последовательным проходом с удержанием только нужных int8-векторов. На одном реальном шарде старый способ занял 164,05 с, новый — 20,75 с (наблюдаемое отношение 7,91). Порядок запуска был «старый, затем новый», поэтому файловый кеш мог повлиять на время. Все девять Float32-признаков совпали с прежним кешем побитово. После явного выравнивания окон это проверено повторно на 476 301 кандидате; нестандартный batch=3 проверен тестом с пересечением границы 128 лент. [Замер и проверка](cosine_reader_benchmark_001.json).

Это сравнение объёма обучения: 1 → 2 train-шарда, то есть 0,193% → 0,388% кандидатов train. Набор признаков, параметры, val-ключи, таргеты и временные назначения совпадают. Чекпойнты выбирались только по псевдо-Public; псевдо-Private оценён после обучения. NDCG@30 псевдо-Private: 0.291889 → 0.293979; разность +0.002090 (+0.72%). Парный bootstrap по 10 095 лентам, 1000 повторов, seed=42: 95% интервал [-0.000129, 0.004226]. Интервал включает ноль: устойчивый прирост пока не подтверждён. Это оценка на фиксированном val для малого train; она не переносится автоматически на полный train или лидерборд.

Прогон выполнен с чистого коммита `f948754`. Позднейшее исправление окон при нестандартном батче не меняет использованные в нём признаки с batch=16. Пиковый рабочий набор процесса — 1.036 ГиБ; длительность всей цепочки — 807.0 с. Полный test-сабмит на 2,904,806 строк прошёл валидатор; загрузки на платформу не было. Полная материализация cosine для train остаётся отдельным шагом.

[Параметры и метрики](baseline_cosine_2parts_001.json), [сравнение роста train](train_growth_comparison_001.json).

```powershell
uv run --locked python -m scripts.build_similarity_features --train-parts 2 --resume
uv run --locked python -m scripts.run_baseline_check --feature-root data/features/cosine --train-parts 2 --out runs/baseline-cosine-2parts-001
uv run --locked python -m scripts.compare_baselines --baseline runs/baseline-cosine-001 --candidate runs/baseline-cosine-2parts-001 --allow-train-growth --out runs/baseline-cosine-2parts-001/comparison.json
```


## Полный cosine-кеш — 12.11.2026

Завершена материализация всех 640 файлов с девятью Float32-признаками. Train содержит 246,745,836 строк в 512 шардах, val — 1,389,971 в 64, test — 2,904,806 в 64. Всего 251,040,613 вхождений кандидатов, размер кеша 4.936 ГиБ.

Полный аудит прошёл. Для каждой строки проверены ключи и порядок относительно подготовленных таблиц, схема, конечность чисел, доступность эмбеддинга и счётчики исходной истории. Девять формул независимо пересчитаны в Float64 на 1,921 лентах (153,230 кандидатов), включая 1 ленты с повторными item_id. Численная проверка охватывает эту выборку. Допуски абсолютной и относительной ошибки составляют 2e-6. Длительность аудита — 429.1 с.

Расчёт возобновлялся по сохранённому манифесту. Последний проход использовал `--prefetch-parts 16`, пиковый рабочий набор процесса составил 1.197 ГиБ. Версия численных формул сохранена, все файлы имеют сведения о производителе. Аудит хранит отпечаток итогового манифеста и версию проверяющего скрипта.

[Отчёт полного аудита](cosine_cache_audit_full.json). Полные cosine- и within-feed-признаки готовы для командных моделей. Подготовка после скачивания исходного архива воспроизводится одной командой, размер группы чтения можно выбрать по доступной RAM.

```powershell
uv run --locked python -m scripts.reproduce_data --raw competition --resume --all-train-features --prefetch-parts 4 --infeed data/features/infeed
uv run --locked python -m scripts.audit_similarity_features --out runs/cosine_cache_audit_full.json
```


## History-кеш и типы событий — 12.11.2026

Дима подключил общий код `origin/danya` (`e79674c`) в ветку `dima`. Фактическая группа history содержит 46 признаков. В прежнем описании было указано 49; README исправлен по списку колонок кода.

Построены выборки на 8 (3,850,759 кандидатов) и 32 (15,373,984 кандидатов) train-шардах, val/test покрыты полностью. Первый расчёт занял 86.91 с, расширение — 131.27 с. Медианное время расчёта train-шарда в итоговом кеше — 4.289 с. Наборы выбираются через linspace и пересекаются частично, поэтому кеш сохраняет объединение из 38 train-шардов. Файлы и метаданные проверяются по отпечаткам кандидатов и историй; прогресс сохраняется после каждого шарда.

Аудит прошёл для 22,561,433 строк. Проверены ключи, схема, отсутствие null/NaN/inf, отпечатки обоих источников и файлов признаков. Число совпадений кандидата с товаром в истории независимо проверено на всех строках. Все 46 формул независимо пересчитаны на 498 лентах (50,326 строк), допуски rtol/atol=2e-6. Повторные item_id дополнительно проверены синтетическим тестом с сохранением отдельных candidate_index.

Таблицы `h_item_evtN > 0 → доля target=1` рассчитаны для обеих выборок train и всего val. Во всех десяти группах нет наблюдений. Полная проверка исходных историй подтвердила ноль совпадений кандидата с товаром собственной истории на 38 покрытых train-шардах и всём val/test. Поэтому вероятность target по этим группам оценить невозможно. Типы 1–10 присутствуют в историях, признаки категорий и профиля событий рассчитаны. Первые и последние ленты первого файла каждого split дополнительно сверены с сырым архивом.

Загрузчик `src.gbdt.prepared.load_prepared` проверен на реальном train-шарде: 476 301 строка, 73 признака (6 базовых + 9 cosine + 12 infeed + 46 history), соединение строго по вхождению кандидата. Отдельно подтверждено покрытие нужных шардов для размеров 8 и 32.

Проверки — 62 unittest-теста инфраструктуры и history, все прошли. В общем uv-окружении отсутствуют PyTorch и pytest; нейросетевой pytest-модуль `tests/test_edge_cases.py` в этот прогон не входил.

[Время и покрытие](history_materialization_001.json), [аудит 8 шардов](history_cache_audit_008.json), [итоговый аудит](history_cache_audit_032.json), [интеграция с GBDT](history_gbdt_integration_001.json), [события на 8 шардах](history_event_target_008.md), [события на 32 шардах](history_event_target_032.md).

```powershell
uv run --locked python -m scripts.build_history_features --train-parts 8 --resume
uv run --locked python -m scripts.build_history_features --train-parts 32 --resume
uv run --locked python -m scripts.audit_history_features --out runs/history_cache_audit.json
uv run --locked python -m scripts.analyse_history_events --train-parts 32 --out experiments/history_event_target_032
```


## Itemstats — 13.11.2026

Реализованы 32 признака товаров и категорий. Для модели на восьми train-шардах агрегаты рассчитаны по остальным 504 шардам, содержащим 242,895,077 показов и 5,362,045 позитивов. Общий prior CTR равен 0.02207556. По этим же источникам рассчитана кривая CTR для 300 позиций. Target val/test не участвует в статистиках; идентификаторы кандидатов используются для ограничения размера таблицы lookup.

Для каждой сущности вычислены показы, клики, CTR со сглаживанием alpha=20, ожидаемые клики по позициям, сглаженное отношение кликов к ожидаемым, число событий в историях и частоты типов 1–10. Повторные события учитываются как отдельные вхождения. Неизвестные сущности получают нулевые счётчики, prior CTR и нейтральное отношение 1. Значение alpha выбрано до оценки модели и в этом эксперименте не подбиралось.

Кеш покрывает 8 train-шардов и весь val/test: 8,145,536 строк, 0.251 ГиБ Parquet. Исходный лимит DuckDB 1024 МиБ с двумя потоками был достигнут при агрегации историй. Расчёт возобновлён с сохранённых таблиц при лимите 1536 МиБ и одном потоке. Успешный проход занял 358.36 с, пиковый рабочий набор процесса составил 1.661 ГиБ.

Полный аудит ключей, схемы, конечности чисел, источников и раздельности шардов прошёл. Для 408 кандидатов (три строки каждого файла) все 32 значения независимо пересчитаны через Polars и Python по всем 504 исходным шардам. Допуски rtol/atol=2e-6. Повторно сверены общий prior и CTR всех позиций. Длительность аудита — 79.47 с.

В test у 53.71% строк есть показы товара в источниках, у 51.36% — события товара в исходных историях. Это доли вхождений кандидатов.

Интеграция с `src.gbdt.prepared.load_prepared` проверена на 476 301 строке со всеми четырьмя группами и 105 признаками. Попытка использовать этот itemstats-кеш для выбранных 32 обучающих шардов отклоняется, поскольку часть из них входит в источники. Для другой обучающей выборки нужен отдельный кеш с новым исключением шардов.

Проверочный CatBoost YetiRank обучен на том же одном шарде и с теми же параметрами, что исходная cosine-модель: 100 максимальных итераций, depth=6, lr=0.1, seed=42, CPU=2. Использовано 47 признаков (6 базовых, 9 cosine, 32 itemstats). Выбрано 42 дерева по pseudo-Public. Статистики исключают восемь шардов, включая этот обучающий шард. NDCG@30 pseudo-Private вырос 0.291889 → 0.295374, разность +0.003485 (+1.19%). Парный bootstrap по 10 095 лентам, 1000 повторов, seed=42 дал 95% интервал [0.000296, 0.006773]. Оценка относится к этому фиксированному val и проверочному обучению на одном шарде.

Full/@10/@30 совпали с предоставленным эталоном. Сабмит на 2 904 806 строк прошёл валидатор и сохранён локально. Общий прогон — 64 unittest-теста инфраструктуры, history и itemstats, все прошли. Нейросетевой pytest-модуль требует отдельного окружения с PyTorch и pytest.

[Материализация и покрытие](itemstats_materialization_008.json), [аудит](itemstats_008_audit.json), [интеграция](itemstats_gbdt_integration_001.json), [модель](baseline_itemstats_001.json), [парное сравнение](itemstats_comparison_001.json).

```powershell
uv run --locked python -m scripts.build_itemstats_features --train-parts 8 --out data/features/itemstats-008 --memory-mb 1536 --threads 1 --resume
uv run --locked python -m scripts.audit_itemstats_features --cache data/features/itemstats-008 --out runs/itemstats_008_audit.json
uv run --locked python -m scripts.run_baseline_check --feature-root data/features/cosine --feature-root data/features/itemstats-008 --train-parts 1 --out runs/baseline-itemstats-001
uv run --locked python -m scripts.compare_baselines --baseline runs/baseline-cosine-001 --candidate runs/baseline-itemstats-001 --out runs/baseline-itemstats-001/comparison.json
```


## Itemstats для 32 train-шардов — 13.11.2026

Построен отдельный кеш `data/features/itemstats-032` с 32 Float32-признаками товаров и категорий. Из источников исключены все 32 выбранных обучающих шарда. Статистики рассчитаны по остальным 480 train-шардам: 231,371,852 показов и 5,106,943 позитивов. Prior CTR и кривая CTR позиций рассчитаны по тем же источникам, alpha=20. Кеш `itemstats-008` сохранён с прежним отпечатком манифеста.

Покрытие нового кеша — 15,373,984 кандидатов train в 32 шардах, 1,389,971 val и 2,904,806 test. Всего 19,668,761 строк, 0.631 ГиБ Parquet. Расчёт занял 506.45 с при лимите DuckDB 2048 МиБ и 1 потоке. Пиковый рабочий набор процесса составил 2.160 ГиБ. Медианное время записи train-шарда — 1.773 с.

Аудит проверил ключи и порядок всех строк, схему, конечность чисел, отпечатки источников и раздельность обучающих и исходных шардов. Для 480 кандидатов все 32 признака независимо пересчитаны через Polars/Python по всем 480 исходным шардам, rtol/atol=2e-6. Общие числа показов и позитивов, prior CTR и CTR каждой позиции также совпали. Длительность аудита — 80.17 с.

В test у 53.37% вхождений кандидатов есть показы товара в источниках, у 51.02% — события товара в историях источников.

Проверено покрытие выбранных 32 шардов всеми четырьмя группами. GBDT-загрузчик прочитал один реальный train-шард со всеми 105 признаками; дополнительно проверено присоединение признаков для другого шарда из выбранного набора. Попытка использовать itemstats-032 для несовместимого набора из 8 шардов получила ожидаемый отказ.

[Материализация](itemstats_materialization_032.json), [полный аудит кеша](itemstats_032_audit.json), [проверка GBDT-загрузчика](itemstats_gbdt_integration_032.json).

```powershell
uv run --locked python -m scripts.build_itemstats_features --train-parts 32 --out data/features/itemstats-032 --memory-mb 2048 --threads 1 --resume
uv run --locked python -m scripts.audit_itemstats_features --cache data/features/itemstats-032 --out runs/itemstats_032_audit.json
```

## Userstats — 13.11.2026

Реализованы 16 Float32-признаков `is_user_*`: показы, клики, CTR со сглаживанием alpha=20, ожидаемые клики по позициям, сглаженное отношение кликов к ожидаемым, число событий в донорских историях и частоты типов 1–10. Использованы те же формулы, что в itemstats. Повторные события в историях разных донорских лент считаются повторно. Для неизвестных пользователей счётчики нулевые, CTR равен общему prior, отношение кликов равно 1. Alpha задана до оценки модели и не подбиралась.

Отдельные кеши для 8 и 32 выбранных train-шардов построены по остальным 504 и 480 шардам соответственно. Target исключённых обучающих шардов и val/test не участвует в расчёте. Списки источников, отпечатки кандидатов и историй, исходный код и формулы зафиксированы в manifest. Несовместимый обучающий набор отклоняется загрузчиком. Существующие itemstats-кеши сохранили прежние отпечатки.

Кеш `userstats-008` содержит 8 145 536 строк, 35 988 477 байт Parquet; расчёт занял 30.95 с, пиковый рабочий набор — 526 802 944 байта. Кеш `userstats-032` содержит 19 668 761 строку, 86 074 994 байта; расчёт занял 39.58 с, пик — 557 281 280 байт. DuckDB: лимит 512 МиБ, один поток. Весь val/test покрыт в обоих кешах. В test показы пользователя известны для 33.65% / 32.80% вхождений кандидатов соответственно; доли относятся к строкам, а не уникальным пользователям.

Оба аудита прошли: ключи, типы, конечность значений, источники и исключения проверены на всех строках. Формулы независимо пересчитаны по всем донорским шардам для трёх кандидатов каждого выходного файла: 408 строк / 375 пользователей и 480 строк / 446 пользователей. Общие показы, позитивы, prior и CTR всех позиций также сверены. Соответствие пользователя между кандидатами и метаданными истории проверено для каждой донорской ленты. Аудиты заняли 20.66 и 23.86 с, допуски rtol/atol=2e-6.

Проверено покрытие обоих обучающих наборов всеми пятью группами. `src.gbdt.prepared.load_prepared` прочитал реальный train-шард (476 301 строка, 6 132 ленты) со 121 признаком. Для обоих кешей проверен отказ на несовместимой обучающей выборке.

Выполнен один проверочный CatBoost-прогон с добавлением userstats к cosine + itemstats-008: тот же обучающий шард, параметры и val, 63 признака, 41 дерево. Чекпойнт выбран только по pseudo-Public. NDCG@30 pseudo-Private: 0.295374 → 0.295649, разность +0.000275 (+0.093%). Парный bootstrap по 10 095 лентам, 1000 повторов, seed=42: 95% интервал [-0.001836, 0.002119]. Прирост не подтверждён; группа остаётся опциональной до отдельной проверки на большем train. Private не использовался для выбора параметров. Метрики full/@10/@30 сверены с эталоном; сабмит на 2 904 806 строк прошёл валидатор после загрузки сохранённой модели. На платформу он не отправлялся.

66 unittest-тестов общей инфраструктуры, history, itemstats и userstats прошли. Проверены исключение обучающих target, неизвестный пользователь, повторные кандидаты, нулевой prior, восстановление после прерывания и обнаружение повреждённых чисел. Модуль `tests/test_edge_cases.py` с необязательными PyTorch/pytest в этот прогон не входил.

[Материализация 8](userstats_materialization_008.json), [материализация 32](userstats_materialization_032.json), [аудит 8](userstats_008_audit.json), [аудит 32](userstats_032_audit.json), [интеграция GBDT](userstats_gbdt_integration_001.json), [прогон модели](baseline_userstats_001.json), [парное сравнение](userstats_comparison_001.json).

```powershell
uv run --locked python -m scripts.build_userstats_features --train-parts 8 --out data/features/userstats-008 --memory-mb 512 --threads 1 --resume
uv run --locked python -m scripts.build_userstats_features --train-parts 32 --out data/features/userstats-032 --memory-mb 512 --threads 1 --resume
uv run --locked python -m scripts.audit_userstats_features --cache data/features/userstats-008 --out runs/userstats_008_audit.json
uv run --locked python -m scripts.audit_userstats_features --cache data/features/userstats-032 --out runs/userstats_032_audit.json
uv run --locked python -m scripts.run_baseline_check --data data/prepared --raw competition --feature-root data/features/cosine --feature-root data/features/itemstats-008 --feature-root data/features/userstats-008 --out runs/baseline-userstats-001 --train-parts 1 --iterations 100 --depth 6 --seed 42 --threads 2
uv run --locked python -m scripts.compare_baselines --baseline runs/baseline-itemstats-001 --candidate runs/baseline-userstats-001 --out runs/baseline-userstats-001/comparison.json
```

## GBDT на 8 шардах (Даня) — 12–13.11.2026

| дата | ветка | - | модель | pseudo-private NDCG full / idcg30 / @30 | Public | примечание |
|---|---|---|---|---|---|---|
| 2026-11-12 | danya | - | catboost/default pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3596/0.3596/0.2973 | - | out=runs/cb8; feats=$B (f+cosine+infeed), 8 parts |
| 2026-11-12 | danya | - | catboost/default pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3600/0.3601/0.2986 | - | out=runs/cb8-cv-B; feats=$B (f+cosine+infeed), 8 parts; iters 165 from cb8; Public submission 2026-11-12 |
| 2026-11-12 | danya | - | catboost/default pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3602/0.3602/0.2984 | - | out=runs/cb8-hist; feats=$H (+history), 8 parts |
| 2026-11-12 | danya | - | lightgbm/default pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3500/0.3501/0.2859 | - | out=runs/lgb8-hist; feats=$H (+history), 8 parts |
| 2026-11-12 | danya | - | lightgbm/default pos=ipw mode=holdout extra=- depth=8 lr=0.05 | priv: 0.2778/0.2779/0.1960 | - | out=runs/lgb8-ipw; feats=$H (+history), 8 parts; ipw-max 10, rejected (non-monotone CTR by pos) |
| 2026-11-12 | danya | - | lightgbm/default pos=feature mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3496/0.3497/0.2851 | - | out=runs/lgb8-posf; feats=$H (+history), 8 parts |
| 2026-11-12 | danya | - | lightgbm/default pos=topn mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3497/0.3498/0.2858 | - | out=runs/lgb8-top50; feats=$H (+history), 8 parts |
| 2026-11-12 | danya | - | catboost/YetiRankPairwise pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3604/0.3605/0.2985 | - | out=runs/cb8-pair; feats=$H (+history), 8 parts; 14 min vs 8.7 |
| 2026-11-13 | danya | - | catboost/default pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3619/0.3620/0.3005 | - | out=runs/cb8-cv; feats=$H (+history), 8 parts; iters 323 from cb8-hist |
| 2026-11-13 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3565/0.3566/0.2953 | - | out=runs/lgb8-cv; feats=$H (+history), 8 parts; iters 126 from lgb8-hist |
| 2026-11-13 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3562/0.3563/0.2948 | - | out=runs/lgb8-cv-w3; feats=$H (+history), 8 parts; val-weight 3 |
| 2026-11-13 | danya | - | catboost/default pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3595/0.3595/0.2976 | - | out=runs/cb8-is; feats=$H + itemstats-008, 8 parts; vs cb8-hist priv -0.0008 [-0.0041, +0.0026], rejected |
| 2026-11-13 | danya | - | blend weights cb8-cv 0.667 + lgb8-cv 0.333 (z-score, weights fit on pseudo-public) | priv: -/-/0.3022 | - | out=runs/blend-cb-lgb; vs cb8-cv priv +0.0017 [+0.0000, +0.0035]; validator OK |
| 2026-11-13 | danya | - | catboost/default pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3620/0.3621/0.2996 | - | out=runs/cb16-hist; feats=$H, 16 parts; 15.4 min, peak RAM 16.4 GB; vs cb8-hist priv +0.0012 [-0.0010, +0.0034] |
| 2026-11-13 | danya | - | catboost/default pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3627/0.3627/0.3008 | - | out=runs/cb32-hist; feats=$H, 32 parts; 49 min, peak RAM 30.6 GB; vs cb8-hist priv +0.0024 [+0.0003, +0.0045], pub -0.0020 |
| 2026-11-13 | danya | - | lightgbm/default pos=none mode=holdout extra=- depth=4 lr=0.02244 | priv: 0.3562/0.3562/0.2932 | - | out=runs/lgb8-tuned; feats=$H, 8 parts; Optuna 40 trials selected on pseudo-public (depth 4, lr 0.022, min_data_in_leaf 256, lambda_l2 28.8); vs lgb8-hist priv +0.0073 [+0.0038, +0.0108] |
| 2026-11-13 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=4 lr=0.02244 | priv: 0.3599/0.3599/0.2991 | - | out=runs/lgb8-tuned-cv; feats=$H, 8 parts; tuned params; vs lgb8-cv priv +0.0037 [+0.0007, +0.0069], accepted |
| 2026-11-13 | danya | - | blend weights cb8-cv 0.333 + lgb8-tuned-cv 0.667 | priv: -/-/0.3015 | - | out=runs/blend-cb-lgbt; vs blend-cb-lgb priv -0.0007 [-0.0025, +0.0012], noise |
| 2026-11-13 | danya | - | blend weights cb8-cv 0.667 + lgb8-cv 0.167 + lgb8-tuned-cv 0.167 | priv: -/-/0.3014 | - | out=runs/blend-3; vs blend-cb-lgb priv -0.0008 [-0.0019, +0.0002], noise; blend-cb-lgb stays the submission |
| 2026-11-13 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3566/0.3567/0.2942 | - | out=runs/lgb8-cv-s1 |
| 2026-11-13 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3569/0.3569/0.2951 | - | out=runs/lgb8-cv-s2 |
| 2026-11-15 | danya | - | lightgbm/default pos=none mode=holdout extra=- depth=4 lr=0.02244 | priv: 0.3532/0.3533/0.2888 | - | out=runs/lgb32-tuned; feats=$H, 32 parts; tuned params, best_iter=141 on pseudo-public |
| 2026-11-15 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=4 lr=0.02244 | priv: 0.3590/0.3591/0.2968 | - | out=runs/lgb32-tuned-cv; feats=$H, 32 parts; iters 141 from lgb32-tuned; vs lgb8-tuned-cv priv -0.0022 [-0.0044, +0.0000], rejected |
| 2026-11-15 | danya | - | catboost/default pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3612/0.3612/0.2998 | - | out=runs/cb32-is; feats=$H + itemstats-032, 32 parts; vs cb32-hist priv -0.0010 [-0.0044, +0.0025], rejected |
| 2026-11-15 | danya | - | seed-avg z-score lgb8-cv + lgb8-cv-s1 + lgb8-cv-s2 mode=cv depth=8 lr=0.05 | priv: -/-/0.2987 | - | out=runs/lgb8-cv-seeds; feats=$H, 8 parts, 3 seeds; pub 0.3104; vs lgb8-cv priv +0.00333 [+0.00101, +0.00576] (p=0.998), accepted; validator OK |
| 2026-11-15 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=4 lr=0.02244 | priv: 0.3601/0.3601/0.2979 | - | out=runs/lgb32-tuned-cv298; feats=$H, 32 parts; tuned params, iters 298 from lgb8-tuned; pub 0.3104; vs lgb8-tuned-cv priv -0.00111 [-0.00320, +0.00105] (p=0.151), rejected |
| 2026-11-15 | danya | - | catboost/QuerySoftMax pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3621/0.3621/0.3004 | - | out=runs/cb32-qsm; feats=$H, 32 parts, GPU (border_count=254, one_hot_max_size=100); best_iter=188; pub 0.3117; vs cb32-hist priv -0.00044 [-0.00208, +0.00138] (p=0.298) |
| 2026-11-15 | danya | - | catboost/QuerySoftMax pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3624/0.3624/0.3007 | - | out=runs/cb32-qsm-cv; feats=$H, 32 parts, GPU, iters 188 from cb32-qsm; pub 0.3114; vs cb32-hist priv -0.00010 [-0.00179, +0.00170], vs cb8-cv +0.00022 [-0.00198, +0.00239]; select_final B; validator OK |
| 2026-11-15 | danya | - | catboost/default pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3615/0.3616/0.2994 | - | out=runs/cb32-yeti-gpu; feats=$H, 32 parts, GPU (border_count=254, one_hot_max_size=100), no-test; best_iter=688; pub 0.3113; vs cb32-hist priv -0.00137 [-0.00287, +0.00019] |
| 2026-11-15 | danya | - | catboost/default pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3617/0.3618/0.3000 | - | out=runs/cb32-cv; feats=$H, 32 parts, GPU, iters 688 from cb32-yeti-gpu (~10.9 min); pub 0.3115; vs cb32-hist priv -0.00082 [-0.00237, +0.00081], vs cb8-cv -0.00049 [-0.00259, +0.00156], rejected; validator OK |
| 2026-11-15 | danya | - | blend weights cb8-cv 0.500 + lgb8-cv-seeds 0.500 (z-score, weights fit on pseudo-public) | priv: -/-/0.3010 | - | out=runs/blend-cb-lgb-seeds; pub 0.3155; vs blend-cb-lgb priv -0.00116 [-0.00269, +0.00027] (p=0.067), rejected; validator OK |
| 2026-11-15 | danya | - | blend weights cb32-cv 0.333 + lgb32-tuned-cv 0.667 (z-score, weights fit on pseudo-public) | priv: -/-/0.2994 | - | out=runs/blend32-sec23; pub 0.3116; vs blend-cb-lgb priv -0.00277 [-0.00512, -0.00061] (p=0.006), rejected; validator OK |
| 2026-11-15 | danya | - | blend weights cb8-cv 0.333 + cb32-cv 0.167 + cb32-qsm-cv 0.167 + lgb8-cv-seeds 0.333 (z-score, weights fit on pseudo-public) | priv: -/-/0.3018 | - | out=runs/blend32; pub 0.3164; vs blend-cb-lgb priv -0.00036 [-0.00195, +0.00119] (p=0.334), not accepted over blend-cb-lgb; validator OK |
| 2026-11-15 | danya | - | dima baseline-joint-008-retry-001 (catboost/YetiRank mode=holdout depth=6 lr=0.1 seed=42, 92 trees) | priv: 0.3607/0.3607/0.2988 | - | out=runs/dima-joint008; feats=$H + itemstats-008 (105, 504 donor parts), 8 train parts; pub 0.3118; vs cb8-cv priv -0.00166 [-0.00504, +0.00171] (p=0.164); validator OK |
| 2026-11-15 | danya | - | blend weights cb8-cv 0.200 + lgb8-cv 0.400 + dima-joint008 0.400 (z-score, weights fit on pseudo-public) | priv: -/-/0.3042 | - | out=runs/blend-cb-lgb-dima; pub 0.3173; vs blend-cb-lgb priv +0.00199 [-0.00014, +0.00421] (p=0.965, all +0.00224); validator OK |
| 2026-11-15 | danya | - | blend weights cb8-cv 0.133 + cb32-cv 0.067 + cb32-qsm-cv 0.267 + lgb8-cv-seeds 0.267 + dima-joint008 0.267 (z-score, weights fit on pseudo-public) | priv: -/-/0.3045 | - | out=runs/blend32-dima; pub 0.3184; vs blend-cb-lgb priv +0.00232 [+0.00024, +0.00446] (p=0.986, all +0.00280), accepted; new best submission; validator OK |
| 2026-11-15 | danya | - | catboost/default pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3601/0.3601/0.2980 | - | out=runs/cb32-is-us; feats=$H + itemstats-032 + userstats-032 (121), 32 parts, GPU; best_iter=280; pub 0.3114; vs cb32-hist priv -0.00283 [-0.00622, +0.00072] (p=0.051), rejected |
| 2026-11-15 | danya | - | catboost/default pos=none mode=holdout extra=- depth=8 lr=0.11063 | priv: 0.3617/0.3618/0.2999 | - | out=runs/cb32-tuned; feats=$H, 32 parts, GPU, Optuna 25 trials on 8 parts (best pub=0.31423, l2_leaf_reg=8.142, random_strength=0.320); best_iter=390; pub 0.3108 |
| 2026-11-15 | danya | - | catboost/default pos=none mode=cv extra=- depth=8 lr=0.11063 | priv: 0.3615/0.3615/0.2999 | - | out=runs/cb32-tuned-cv; feats=$H, 32 parts, GPU, tuned params, iters 390 from cb32-tuned; pub 0.3115; vs cb32-cv priv -0.00005 [-0.00086, +0.00080], vs cb32-hist -0.00086 [-0.00234, +0.00075], rejected; validator OK |
| 2026-11-15 | danya | - | catboost/QuerySoftMax pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3621/0.3622/0.3004 | - | out=runs/cb32-qsm-cv-s1; feats=$H, 32 parts, GPU, seed=1, iters 188; pub 0.3123 |
| 2026-11-15 | danya | - | catboost/QuerySoftMax pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3616/0.3616/0.3001 | - | out=runs/cb32-qsm-cv-s2; feats=$H, 32 parts, GPU, seed=2, iters 188; pub 0.3112 |
| 2026-11-15 | danya | - | seed-avg z-score cb32-qsm-cv + s1 + s2 mode=cv depth=8 lr=0.05 | priv: -/-/0.3006 | - | out=runs/cb32-qsm-cv-seeds; feats=$H, 32 parts, 3 seeds; pub 0.3120; vs cb32-qsm-cv priv -0.00009 [-0.00091, +0.00074] (all +0.00011), vs cb32-hist -0.00019 [-0.00186, +0.00158]; validator OK |
| 2026-11-15 | danya | - | catboost/default pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3615/0.3615/0.3005 | - | out=runs/cb8-is-cv; feats=$H8+itemstats-008 (105), GPU, iters 249 from cb8-is; pub 0.3140, all 0.3046; vs dima-joint008 priv +0.00171 [-0.00004, +0.00350], vs cb8-cv +0.00005 [-0.00317, +0.00302] |
| 2026-11-15 | danya | - | catboost/QuerySoftMax pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3624/0.3625/0.3010 | - | out=runs/cb8-is-qsm; feats=$H8+itemstats-008 (105), GPU, best_iter=248; pub 0.3127, all 0.3045; --no-test |
| 2026-11-15 | danya | - | catboost/QuerySoftMax pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3650/0.3651/0.3045 | - | out=runs/cb8-is-qsm-cv; feats=$H8+itemstats-008 (105), GPU, iters 248 from cb8-is-qsm; pub 0.3163, all 0.3080; vs dima-joint008 priv +0.00565 [+0.00372, +0.00763], vs cb8-cv +0.00399 [+0.00088, +0.00694], accepted |
| 2026-11-15 | danya | - | lightgbm/default pos=none mode=holdout extra=- depth=4 lr=0.02244 | priv: 0.3594/0.3594/0.2970 | - | out=runs/lgb8-is-tuned; feats=$H8+itemstats-008 (105), CPU, best_iter=140; pub 0.3130; --no-test |
| 2026-11-15 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=4 lr=0.02244 | priv: 0.3634/0.3635/0.3027 | - | out=runs/lgb8-is-tuned-cv; feats=$H8+itemstats-008 (105), CPU, iters 140 from lgb8-is-tuned; pub 0.3150, all 0.3064; vs lgb8-tuned-cv priv +0.00361 [+0.00083, +0.00637], vs dima-joint008 +0.00384 [+0.00128, +0.00631], accepted |

## Связь типов событий с категорией кандидата — 13.11.2026

В `scripts.analyse_history_events` добавлен режим `--entity category`. Условие `h_cat_evtN > 0` выбирает вхождения кандидатов, категория которых встречалась в истории пользователя с типом события N. Используются уже проверенные признаки history. Дробные доли категорий преобразуются в условие присутствия; в таблице считаются кандидаты и позитивы. Исходный режим по товарам сохранён.

Проанализированы 15 373 984 кандидата / 196 954 ленты train и весь val из 1 389 971 кандидата / 14 423 лент. Итоговые числа строк и позитивов совпали с исходным анализом по товарам. Проверены отпечатки источников, однозначность соединения по вхождению кандидата, конечность и диапазон долей. В отчёт записаны хеши подготовленного датасета, history-манифеста и кода анализа.

Все десять категорийных групп содержат наблюдения. На val доля target=1 составляет 1.65–2.21% при общей доле 1.58%. Для типа 10 в группе 51 338 кандидатов, 1 135 позитивов, 954 ленты. Доля равна 2.2108%, отношение к общей доле — 1.398. На train для типа 10 — 484 860 кандидатов и 14 329 позитивов, доля 2.9553% против общей 2.2111%.

Оценки описывают связь с target в выбранных группах. Группы пересекаются, на результат влияют состав категорий, длина истории и исходная позиция train. Для интерпретации значений кодов событий требуется дополнительная проверка состава групп. Этот анализ использует полный val как EDA. Исходные группы совпадения конкретного товара `h_item_evtN > 0` по-прежнему пусты.

Пять целевых unittest-тестов history и анализа событий прошли. Регрессионный тест проверяет дробные доли 0.25/0.5/0.75, подсчёт отдельных candidate_index при одинаковом item_id, позитивы, отсутствующую группу и отклонение долей выше единицы. Результаты реального анализа дополнительно сверены с исходными общими счётчиками train/val.

[Таблица событий по категориям](history_category_event_target_032.md), [полные агрегаты и отпечатки](history_category_event_target_032.json).

```powershell
uv run --locked python -m scripts.analyse_history_events --entity category --train-parts 32 --out experiments/history_category_event_target_032
```

## Совместные группы на 8 шардах — 13.11.2026

Завершены два заранее заданных прогона на одинаковых 3 850 759 кандидатах / 49 200 лентах train. Первая модель использует cosine, infeed, history и itemstats-008, всего 105 признаков. Вторая добавляет 16 userstats-008. Параметры общие — CatBoost YetiRank, максимум 100 итераций, depth=6, lr=0.1, seed=42, два CPU-потока. Чекпойнты выбраны только по 4 328 лентам pseudo-Public. Таргетные статистики исключают все восемь обучающих шардов.

Сравнение подтвердило одинаковые параметры, обучающие шарды, val-кандидатов, цели и назначение holdout. Хеши общего кода и метаданные общих групп совпадают. Модели обучены на чистых коммитах `055f644` и `1a2e241`. Оба отчёта подтверждают совпадение full/@10/@30 с эталоном и успешную проверку сабмитов на 2 904 806 строках после загрузки сохранённых моделей.

Модель со 105 признаками сохранила 92 дерева и получила NDCG@30 pseudo-Public 0.311774, pseudo-Private 0.298823. Вариант с userstats сохранил 30 деревьев, получил pseudo-Public 0.310522, pseudo-Private 0.297308. Разность Private равна -0.001514. Парный bootstrap по 10 095 лентам, 1000 повторов, seed=42 дал 95% интервал [-0.003391, 0.000188]. Польза userstats в этом прогоне не подтверждена. Модель со 105 признаками также имеет более высокий pseudo-Public, который использовался для выбора чекпойнта.

Успешные прогоны заняли 3251 и 1912 секунд. Пиковые рабочие наборы составили 6 231 179 264 и 7 773 200 384 байта. Первый терминальный запуск прервался на этапе fit. Завершённая пара выполнена отдельным фоновым процессом Windows, результаты находятся в папках с суффиксом `retry-001`.

Для обеих моделей подготовлены `val_pred.parquet` на 1 389 971 строку и `test_pred.parquet` на 2 904 806 строк в командном формате. Проверены конечность Float32-скоров и точное сохранение значений после записи. Модели и предсказания лежат локально в `runs/baseline-joint-008-retry-001/` и `runs/baseline-joint-userstats-008-retry-001/`. Хеши моделей и предсказаний сохранены в агрегированном отчёте передачи.

После завершения прогонов повторена общая проверка. Все 68 unittest-тестов прошли за 20.546 с. Модуль `tests/test_edge_cases.py` с необязательными PyTorch/pytest исключён из этого прогона.

[Прогон со 105 признаками](baseline_joint_008.json), [прогон со 121 признаком](baseline_joint_userstats_008.json), [парное сравнение](userstats_comparison_008.json), [состав и отпечатки файлов передачи](joint_gbdt_handoff_008.json).

```powershell
uv run --locked python -m scripts.reproduce_ranking --raw competition --train-parts 8 --infeed --history --itemstats --out runs/reproduced-joint-008 --resume
uv run --locked python -m scripts.reproduce_ranking --raw competition --train-parts 8 --infeed --history --itemstats --userstats --out runs/reproduced-joint-userstats-008 --resume
uv run --locked python -m scripts.compare_baselines --baseline runs/reproduced-joint-008 --candidate runs/reproduced-joint-userstats-008 --out runs/reproduced-joint-userstats-008/comparison.json
```

## Категорийный вариант item2item — 15.11.2026

По новому приоритету Дани реализована группа `category_cooc`. В исходном train 583 586 378 событий истории. Первый вариант использует компактную матрицу категорий. Каждая донорская лента даёт одно присутствие категории и одно совместное присутствие пары категорий. Оценка пары равна `C(i,j) / sqrt(C(i,i)*C(j,j))`. Матрица симметрична, одинаковые известные категории имеют оценку 1, категории без наблюдений — 0.

Для кандидата считаются сумма и максимум этих оценок по первым 10 и 50 событиям собственной истории, всего четыре Float32-признака. Повторения в истории запроса учитываются при каждом вхождении. Направленные последовательные переходы категорий и пары конкретных товаров требуют отдельной реализации.

Отдельные кеши исключают выбранные 8 или 32 обучающих шарда. Источниками служат 504 или 480 оставшихся train-шардов, используются только категории историй донорских лент. Прогресс агрегации сохраняется после каждого донорского шарда, готовые Parquet — после каждого выходного файла. Регрессионные тесты проверяют восстановление после обоих видов прерывания, ручные формулы, повторные события, неизвестные категории, одинаковые item_id с разными candidate_index, независимость от target и отклонение несовместимой обучающей выборки.

Кеш на 8 шардах покрывает 8 145 536 строк, занимает 65 493 230 байт и построен за 89.34 с при пиковом рабочем наборе 339 152 896 байт. Кеш на 32 шардах покрывает 19 668 761 строку, занимает 159 323 038 байт и построен за 75.98 с при пике 346 415 104 байта. Весь val/test покрыт в обоих случаях.

Независимые аудиты заняли 101.23 и 99.69 с. Все совместные частоты пересчитаны по исходным историям через битовые наборы донорских лент и их пересечения. Во всех строках проверены ключи, типы, конечность, диапазоны и отпечатки. Формулы дополнительно сверены на 408 лентах / 37 043 кандидатах и 480 лентах / 40 831 кандидате. В донорских историях найдены 54 категории. Нормальный GBDT-загрузчик проверен со всеми пятью группами и 109 признаками на реальном train-шарде, покрытие обеих выборок и отказы на несовместимых наборах подтверждены.

Для согласованного проверочного обучения использован тот же один train-шард, что в готовом контроле cosine + itemstats, 476 301 кандидат / 6 132 ленты. Параметры общие — CatBoost YetiRank, максимум 100 итераций, depth=6, lr=0.1, seed=42, два CPU-потока. Вариант с category_cooc получил 51 признак и 19 деревьев, выбранных по pseudo-Public. Сравнение подтвердило одинаковые данные, параметры и val. Pseudo-Public NDCG@30 составил 0.309221 против 0.310286, pseudo-Private — 0.291788 против 0.295374. Разность Private -0.003586, парный bootstrap по 10 095 лентам, 1000 повторов, seed=42 дал 95% интервал [-0.005724, -0.001622]. В этом проверочном прогоне группа ухудшила результат. Она доступна как опциональная группа. Оценка относится к этому train-срезу и фиксированному val.

Прогон занял 75.95 с. Full/@10/@30 сверены с эталоном, сабмит на 2 904 806 строк прошёл валидатор после загрузки модели. Общая проверка содержит 70 unittest-тестов, все прошли за 38.032 с. Необязательный нейросетевой pytest-модуль `tests/test_edge_cases.py` исключён из этого прогона.

[Материализация 8](category_cooc_materialization_008.json), [материализация 32](category_cooc_materialization_032.json), [аудит 8](category_cooc_008_audit.json), [аудит 32](category_cooc_032_audit.json), [GBDT-интеграция](category_cooc_gbdt_integration_001.json), [модель](baseline_category_cooc_001.json), [парное сравнение](category_cooc_comparison_001.json).

```powershell
uv run --locked python -m scripts.build_category_cooc_features --train-parts 8 --out data/features/category-cooc-008 --resume
uv run --locked python -m scripts.build_category_cooc_features --train-parts 32 --out data/features/category-cooc-032 --resume
uv run --locked python -m scripts.audit_category_cooc_features --cache data/features/category-cooc-008 --out runs/category_cooc_008_audit.json
uv run --locked python -m scripts.audit_category_cooc_features --cache data/features/category-cooc-032 --out runs/category_cooc_032_audit.json
uv run --locked python -m scripts.run_baseline_check --feature-root data/features/cosine --feature-root data/features/itemstats-008 --feature-root data/features/category-cooc-008 --train-parts 1 --iterations 100 --depth 6 --seed 42 --threads 2 --out runs/baseline-category-cooc-001
uv run --locked python -m scripts.compare_baselines --baseline runs/baseline-itemstats-001 --candidate runs/baseline-category-cooc-001 --out runs/baseline-category-cooc-001/comparison.json
```

Для передачи Дане подготовлен архив `dima_joint008_predictions_2026-11-15.zip` с исходными `val_pred.parquet` и `test_pred.parquet` модели со 105 признаками. Файлы сверены с опубликованными ранее SHA-256 и повторно проверены после упаковки. Размер ZIP составляет 32 236 939 байт, SHA-256 `3b3b43c7f77891aacc1816ae837a396601ba6f9553858adb1927b2aed59ce0d4`. По просьбе пользователя архив сохранён локально в папке «Загрузки».

## Event-cosine, ранги и время — 15.11.2026

По приоритетам Дани реализованы три опциональные группы. `event_cosine` содержит среднее и максимум cosine по типам событий 1, 2 и 10, а также число доступных векторов каждого типа. Используется окно первых 50 исходных событий, затем фильтрация по типу и маске вектора. Группа содержит девять Float32-признаков. Типы 1 и 2 выбраны как частые события train, тип 10 включён по запросу Дани. Все векторы берутся из исходной таблицы соответствующего split. Отрицательный максимум сохраняется, отсутствующие векторы дают нулевой счётчик и нулевые cosine.

Общий кеш `event-cosine` покрывает объединение из 38 train-шардов, нужных для выборок 8 и 32, и весь val/test. В нём 22 561 433 строки, 358 818 576 байт Parquet. Завершающий вызов расширения до выборки из 32 шардов занял 186.48 с, пик рабочего набора этого вызова составил 712 847 360 байт. Ранее рассчитанные файлы переиспользованы. Чтение векторов группируется по четырём шардам. Проверка размера векторного файла использует целочисленное произведение Python, корректное для train-файла 10 439 416 832 байта на Windows.

`feature_ranks` содержит четыре нормированных средних ранга внутри ленты для `cos_user`, `cos_hist_max_10`, `cos_hist_max_50`, `is_item_ctr`. Источник CTR исключает нужные обучающие шарды. Ранг `h_cat_share` уже присутствует в history под именем `h_cat_share_feedrank`. Отдельные кеши на 8 и 32 шарда покрывают 8 145 536 и 19 668 761 строку, занимают 122 813 112 и 301 439 609 байт. Построение заняло 24.70 и 42.94 с. Ресурсные поля исходных манифестов могут обновляться при resume, поэтому зависимость отслеживает числовую сигнатуру, идентичность itemstats и отпечатки используемых файлов.

`time` содержит час UTC, день недели UTC с понедельника 0, возраст в днях после первого появления товара среди донорских кандидатов и флаг такого прошлого появления. Минимальный донорский timestamp должен быть строго раньше текущего. Для отсутствующего прошлого и идентификаторов товаров 0 или меньше возраст и флаг равны 0. Это наблюдаемый возраст в данных. Кеши занимают 54 546 610 и 134 816 923 байта, покрытие такое же, как у рангов. Построение заняло 65.67 и 103.03 с. Пики рабочих наборов составили 529 866 752 и 922 198 016 байт. Таблица первых появлений сохраняется в DuckDB и переиспользуется при resume.

Все шесть аудитов прошли. Ключи, схема, диапазоны, конечность значений и отпечатки проверены на всех строках соответствующей когорты. Независимые формулы проверены на 408 лентах / 40 695 кандидатах для восьми шардов и 480 лентах / 47 528 кандидатах для 32. Для event-cosine используется Float64-расчёт по исходным векторам, для рангов — границы равных значений в отсортированном массиве, для возраста — независимый минимум по всем 504 или 480 донорским шардам для товаров из проверяемых лент. Допуски rtol/atol=2e-6.

Кеши защищены исключительной блокировкой записи, освобождаемой ОС при завершении процесса. Регрессионные проверки покрывают отрицательные cosine, сохранение масок и исходного окна, повторные item_id с разными candidate_index, возобновление после прерывания, расширение общего кеша, одновременную запись, изменения векторов, равные ранги, несовместимые донорские когорты, UTC около начала эпохи и строгое исключение будущих и одновременных появлений товара. Все 75 общих unittest-тестов прошли за 26.119 с. Необязательный нейросетевой pytest-модуль `tests/test_edge_cases.py` исключён из этого прогона. Сквозной тест одной команды прошёл с восемью группами и 138 признаками, включая повторный запуск.

На реальном train-шарде GBDT-загрузчик прочитал все девять групп и 142 признака. Проверены покрытие выборок 8 и 32 и отказы на несовместимых когортах рангов и времени. В `reproduce_ranking` добавлены флаги новых групп, а в отчёт проверочного обучения — хеши их кода.

Выполнено по одному заранее заданному проверочному обучению на 476 301 кандидате / 6 132 лентах train. Для всех трёх использован готовый контроль cosine + itemstats с NDCG@30 pseudo-Private 0.295374. Параметры общие — максимум 100 итераций, depth=6, lr=0.1, seed=42, два CPU-потока. Выбор чекпойнта выполнен по pseudo-Public. Метаданные двух общих групп, обучающие шарды, параметры, val-ключи, цели и holdout совпали с контролем.

| Добавленная группа | Признаков модели | Деревьев | pseudo-Public @30 | pseudo-Private @30 | Разность Private | 95% парный интервал |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| event_cosine | 56 | 26 | 0.311196 | 0.294735 | -0.000639 | [-0.002739, 0.001313] |
| feature_ranks | 51 | 41 | 0.305674 | 0.293380 | -0.001994 | [-0.004546, 0.000332] |
| time | 51 | 20 | 0.308271 | 0.293678 | -0.001696 | [-0.004099, 0.000624] |

Парный bootstrap выполнен по 10 095 лентам, 1000 повторов, seed=42. Все интервалы включают ноль. Положительный эффект на этом малом train-срезе и фиксированном val не подтверждён. Группы доступны опционально для последующей проверки владельцами основных моделей. Full/@10/@30 совпали с эталоном, все три сабмита на 2 904 806 строк прошли валидатор после загрузки сохранённых моделей. Длительность прогонов составила 90.61, 118.62 и 84.72 с соответственно.

[Материализация и контроль исходных групп](additional_feature_materialization_001.json), [проверка загрузчика](additional_gbdt_integration_001.json), [event-cosine 8](event_cosine_008_audit.json), [event-cosine 32](event_cosine_032_audit.json), [ранги 8](feature_ranks_008_audit.json), [ранги 32](feature_ranks_032_audit.json), [время 8](time_008_audit.json), [время 32](time_032_audit.json).

[Прогон event-cosine](baseline_event_cosine_001.json), [сравнение event-cosine](event_cosine_comparison_001.json), [прогон рангов](baseline_feature_ranks_001.json), [сравнение рангов](feature_ranks_comparison_001.json), [прогон времени](baseline_time_001.json), [сравнение времени](time_comparison_001.json).

Команды построения и аудита для обеих когорт приведены в README. Проверочные модели воспроизводятся с добавлением одной группы к общему контролю, например

```powershell
uv run --locked python -m scripts.run_baseline_check --feature-root data/features/cosine --feature-root data/features/itemstats-008 --feature-root data/features/event-cosine --out runs/baseline-event-cosine-001 --train-parts 1 --iterations 100 --depth 6 --seed 42 --threads 2
uv run --locked python -m scripts.compare_baselines --baseline runs/baseline-itemstats-001 --candidate runs/baseline-event-cosine-001 --out runs/baseline-event-cosine-001/comparison.json
```

Для рангов дополнительный корень равен `data/features/feature-ranks-008`, для времени — `data/features/time-008`. Для каждого прогона нужна отдельная новая папка вывода.

15.11 проверен актуальный журнал ветки Дани. Заполненных оценок Public платформы в нём пока нет. Сверка порядка моделей ожидает этих записей. [Журнал ветки danya](https://github.com/shertwg/avito-hackaton2026/blob/danya/experiments/log.md)

## Item2item по конкретным товарам — Дима, 15.11.2026

Добавлены четыре признака `i2i_sum_10`, `i2i_max_10`, `i2i_sum_50`,
`i2i_max_50`. Сходство пары равно числу донорских историй, содержащих оба
товара, делённому на корень из произведения чисел историй для каждого товара.
Каждый товар учитывается один раз внутри донорской ленты. Повторы текущей
истории сохраняются, окна отсекаются до обработки неизвестных товаров.
Неизвестные и неположительные ID дают 0, известная пара товара с собой — 1.
Target в расчёте отсутствует. Донорские группы исключают все train-шарды
соответствующей когорты.

Индекс хранит отсортированные списки донорских лент для нужных товаров в
трёх mmap-массивах. Сходства вычисляются разреженными произведениями с
локальным сжатием ID лент. DuckDB сохраняет таблицы для resume, файлы
признаков имеют отдельные контрольные точки и исключительную блокировку
записи. Зафиксирована прямая зависимость SciPy. Ограничение DuckDB 1536 МБ
относится к его менеджеру памяти, общий рабочий набор процесса приведён ниже.

| Когорта | Строк train/val/test | Товаров в индексе | Связей товар–лента | Parquet, байт | Время, с | Пик рабочего набора, байт |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 008 | 3850759/1389971/2904806 | 5453051 | 95507441 | 56253840 | 536.34 | 2602283008 |
| 032 | 15373984/1389971/2904806 | 10931497 | 143172991 | 152734706 | 1002.72 | 3824844800 |

Оба кеша покрывают весь val/test. Независимые аудиты прошли. Структура всего индекса, ключи, схемы, диапазоны, конечность значений и отпечатки проверены полностью. Для когорты 008 списки лент для 53774 проверяемых товаров заново собраны по всем 504 донорским шардам, формулы сверены на 408 лентах / 40695 кандидатах. Для когорты 032 списки лент для 63001 проверяемых товаров заново собраны по всем 480 донорским шардам, формулы сверены на 480 лентах / 47528 кандидатах. Независимая числовая сверка ограничена этой выборкой товаров.

Все 79 общих unittest-тестов прошли за 35.680 с. Необязательный нейросетевой
pytest-модуль `tests/test_edge_cases.py` исключён. Новые тесты покрывают
дубликаты, неизвестные ID, нормировку, окна, пустой индекс, независимость от
target, донорские исключения, повреждение индекса и возобновление после
прерывания таблицы, публикации массивов и файла признаков. Сквозное
синтетическое воспроизведение прошло с девятью группами и 142 признаками.
Обычный GBDT-загрузчик прочитал реальный train-шард со всеми десятью
группами и 146 признаками для обеих когорт. Неподходящая донорская когорта
item2item отклонена.

### Один проверочный прогон

Использованы тот же один train-шард, 476 301 кандидат / 6 132 ленты,
и готовый контроль cosine + itemstats. Метаданные обеих исходных групп
полностью совпадают. Кандидат содержит 51 признак. Параметры общие
CatBoost YetiRank, максимум 100 итераций, depth=6, lr=0.1, seed=42,
два CPU-потока. Чекпойнт выбран по pseudo-Public.

С item2item pseudo-Public NDCG@30=0.311299892, pseudo-Private=0.298266319. Контроль Private=0.295374228, разность +0.002892091 (+0.979%). 95% парный bootstrap-интервал [+0.000069901, +0.005920777], 1000 повторов по 10 095 лентам, seed=42. В сохранённой модели 31 дерево. Положительный интервал получен на фиксированном малом train-срезе. Проверку в основных командных моделях ведут их владельцы. Группа подключается явно. Полные метрики full/@10/@30 совпали с эталоном. Сабмит на 2 904 806 кандидатов прошёл валидатор после загрузки сохранённой модели.

[Материализация](item2item_materialization_001.json), [аудит 8](item2item_008_audit.json),
[аудит 32](item2item_032_audit.json), [проверка GBDT](item2item_gbdt_integration_001.json),
[модель](baseline_item2item_001.json), [парное сравнение](item2item_comparison_001.json).

```powershell
uv run --locked python -m scripts.build_item2item_features --train-parts 8 --out data/features/item2item-008 --resume
uv run --locked python -m scripts.build_item2item_features --train-parts 32 --out data/features/item2item-032 --resume
uv run --locked python -m scripts.audit_item2item_features --cache data/features/item2item-008 --out runs/item2item_008_audit.json
uv run --locked python -m scripts.audit_item2item_features --cache data/features/item2item-032 --out runs/item2item_032_audit.json
uv run --locked python -m scripts.run_baseline_check --feature-root data/features/cosine --feature-root data/features/itemstats-008 --feature-root data/features/item2item-008 --train-parts 1 --iterations 100 --depth 6 --seed 42 --threads 2 --out runs/baseline-item2item-001
uv run --locked python -m scripts.compare_baselines --baseline runs/baseline-itemstats-001 --candidate runs/baseline-item2item-001 --out runs/baseline-item2item-001/comparison.json
```

Для воспроизведения с сырых данных добавлен флаг `--item2item` в
`scripts.reproduce_ranking`. В журнале ветки `danya`, проверенном через
GitHub API 15.11, заполненных значений Public платформы пока нет.
Сверка порядка моделей ожидает этих записей. Финальный ансамбль остаётся
поздним пунктом после выбора командной конфигурации с 01.12.

## Командный ансамбль и передача item2item в OOF — Дима, 15.11.2026

Даня подтвердил получение архива joint-008, совпадение SHA-256 и успешную
проверку val/test. Его журнал содержит новые принятые ансамбли с этой моделью.
По сообщению Дани `blend-cb-lgb-dima` получил pseudo-Private 0.30419,
pseudo-Public 0.31729. Новый лучший командный сабмит `blend32-dima` получил
pseudo-Private 0.30452 и pseudo-Public 0.31837. Парная разность к прежнему
0.30220 равна +0.00232 с интервалом [+0.00024, +0.00446]. Значения команды
взяты из сообщения Дани, записи ветки danya проверены через GitHub API.
Public платформы в этих строках ожидает заполнения.
[Журнал Дани](https://github.com/shertwg/avito-hackaton2026/blob/danya/experiments/log.md)

По выбору пользователя подготовлена команда для запуска владельцем модели.
В README добавлен `src.gbdt.run_exp --mode cv --folds 4` с пятью группами
cosine, infeed, history, itemstats-008, item2item-008, всего 109 признаков.
Параметры сохраняют joint-008, depth=6, lr=0.1, seed=42, 92 итерации,
два CPU-потока. Число итераций зафиксировано по прежнему чекпойнту joint-008.
Каждая из четырёх моделей использует train и три блока val, предсказывает
оставшийся блок val. Test-score усредняется после z-score внутри ленты.

Команда прошла синтетическую сквозную проверку через настоящий CV-раннер
с теми же пятью группами и 109 признаками. Для каждого обучения перехват
вызова fit проверил отсутствие предсказываемого блока среди обучающих лент.
Все ленты val предсказаны ровно один раз, ключи и target сохранены, test
прошёл валидатор. Проверка использовала по две итерации и depth=2.
Выполнение CV на реальных данных передано Дане.
[Проверка CV](item2item_cv_smoke_001.json)

Перед слиянием повторно прошли все 79 общих unittest-тестов за 40.685 с.
Необязательный нейросетевой pytest-модуль `tests/test_edge_cases.py` исключён.

## Этап 3: OOF-модели `itemstats-008 + item2item-008` и ансамбли (Даня) — 15.11.2026

Аудит `data/features/item2item-008` (`runs/item2item_008_audit.json`): `complete: true`, 504 донорских шарда, 5,453,051 item, 95,507,441 postings, 408 лент.

| дата | ветка | - | модель | pseudo-private NDCG full / idcg30 / @30 | Public | примечание |
|---|---|---|---|---|---|---|
| 2026-11-15 | danya | - | catboost/default pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3644/0.3644/0.3036 | - | out=runs/cb8-is-i2i; feats=$H8+itemstats-008+item2item-008 (109), GPU, best_iter=339; pub 0.3152, all 0.3071; --no-test |
| 2026-11-15 | danya | - | catboost/default pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3647/0.3648/0.3045 | - | out=runs/cb8-is-i2i-cv; feats=109, GPU, iters 339; pub 0.3164, all 0.3081; vs cb8-is-cv priv +0.00395 [+0.00120, +0.00657] (p=0.998), accepted; validator OK |
| 2026-11-15 | danya | - | catboost/QuerySoftMax pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3662/0.3663/0.3051 | - | out=runs/cb8-is-i2i-qsm; feats=109, GPU, best_iter=349; pub 0.3167, all 0.3086; --no-test |
| 2026-11-15 | danya | - | catboost/QuerySoftMax pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3662/0.3662/0.3055 | - | out=runs/cb8-is-i2i-qsm-cv; feats=109, GPU, iters 349; pub 0.3189, all 0.3095; vs cb8-is-qsm-cv priv +0.00101 [-0.00174, +0.00376], vs blend32-dima +0.00096 [-0.00182, +0.00375]; validator OK |
| 2026-11-15 | danya | - | lightgbm/default pos=none mode=holdout extra=- depth=4 lr=0.02244 | priv: 0.3649/0.3650/0.3045 | - | out=runs/lgb8-is-i2i-tuned; feats=109, CPU; pub 0.3173, all 0.3083; --no-test |
| 2026-11-15 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=4 lr=0.02244 | priv: 0.3656/0.3656/0.3055 | - | out=runs/lgb8-is-i2i-tuned-cv; feats=109, CPU; pub 0.3155, all 0.3085; vs lgb8-is-tuned-cv priv +0.00284 [+0.00018, +0.00547] (p=0.981), accepted; validator OK |
| 2026-11-15 | danya | - | blend weights 6 OOF (cb32-qsm-cv-seeds + cb32-tuned-cv + lgb8-cv-seeds + cb8-is-i2i-cv + cb8-is-i2i-qsm-cv + lgb8-is-i2i-tuned-cv) | priv: -/-/0.3061 | - | out=runs/blend-oof-pure; pub 0.3201 (mean: pub 0.3194, priv 0.3069); vs blend32-dima priv +0.00160 [-0.00018, +0.00340] (all +0.00162 [+0.00006, +0.00317]), vs blend-cb-lgb +0.00392 [+0.00171, +0.00615]; validator OK |
| 2026-11-15 | danya | - | blend weights 7 models (6 OOF + dima-joint008) | priv: -/-/0.3075 | - | out=runs/blend-team-v3; pub 0.3200; vs blend32-dima priv +0.00294 [+0.00129, +0.00463] (p=1.000), vs blend-cb-lgb +0.00526 [+0.00303, +0.00754] (p=1.000), Финал A (select_final); validator OK |

## Этап 4: сиды, Optuna-тюнинг на 109 признаках, 32 шарда `item2item-032` и нейросеть Августа (Даня) — 15–16.11.2026

Аудит `data/features/item2item-032` (`runs/item2item_032_audit.json`): `complete: true`, 480 донорских шардов, 10,931,497 items, 143,172,991 postings, 480 лент.
Импортирована OOF-нейросеть Августа (`nn-avgust-oof4-predictions.zip` -> `runs/nn-oof4`, 1,389,971 строк val без пропусков и дублей, validator OK).

| дата | ветка | - | модель | pseudo-private NDCG full / idcg30 / @30 | Public | примечание |
|---|---|---|---|---|---|---|
| 2026-11-15 | danya | - | catboost/QuerySoftMax pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3672/0.3673/0.3065 | - | out=runs/cb8-is-i2i-qsm-cv-s1; seed=1, iters 349, pub 0.3177, all 0.3099 |
| 2026-11-15 | danya | - | catboost/QuerySoftMax pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3661/0.3661/0.3058 | - | out=runs/cb8-is-i2i-qsm-cv-s2; seed=2, iters 349, pub 0.3181, all 0.3095 |
| 2026-11-15 | danya | - | seed-avg mean cb8-is-i2i-qsm-cv + s1 + s2 | priv: -/-/0.3066 | - | out=runs/cb8-is-i2i-qsm-cv-seeds; pub 0.3180, all 0.3100; vs cb8-is-i2i-qsm-cv priv +0.00116 [+0.00016, +0.00213] (p=0.991), accepted; validator OK |
| 2026-11-15 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=4 lr=0.02244 | priv: 0.3653/0.3654/0.3052 | - | out=runs/lgb8-is-i2i-tuned-cv-s1; seed=1, pub 0.3127, all 0.3075 |
| 2026-11-15 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=4 lr=0.02244 | priv: 0.3662/0.3663/0.3057 | - | out=runs/lgb8-is-i2i-tuned-cv-s2; seed=2, pub 0.3156, all 0.3087 |
| 2026-11-15 | danya | - | seed-avg mean lgb8-is-i2i-tuned-cv + s1 + s2 | priv: -/-/0.3055 | - | out=runs/lgb8-is-i2i-tuned-cv-seeds; pub 0.3148, diff 0.00000 vs seed 0 |
| 2026-11-15 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=4 lr=0.02024 | priv: 0.3667/0.3668/0.3066 | - | out=runs/lgb8-is-i2i-opt-cv; Optuna runs/tune-lgb-i2i (best pub 0.3193), 435 iters, pub 0.3171, all 0.3097; vs lgb8-is-i2i-tuned-cv all +0.00119 [+0.00015, +0.00224] (p=0.987); validator OK |
| 2026-11-15 | danya | - | catboost/QuerySoftMax pos=none mode=cv extra=- depth=9 lr=0.03199 | priv: 0.3668/0.3669/0.3063 | - | out=runs/cb8-is-i2i-qsm-opt-cv; Optuna runs/tune-cb-qsm-i2i (best pub 0.3178), 404 iters, pub 0.3172, all 0.3096; validator OK |
| 2026-11-15 | danya | - | blend weights 7 OOF (без Dima) | priv: -/-/0.3075 | - | out=runs/blend-oof-pure-v4; pub 0.3207, all 0.3114 (mean: priv 0.3080) |
| 2026-11-16 | danya | - | nn-oof4 (Август, 4-fold OOF) | priv: -/-/0.3007 | - | out=runs/nn-oof4; pub 0.3144, all 0.3048; validator OK |
| 2026-11-16 | danya | - | blend weights 9 models (7 OOF + nn-oof4 + dima-joint008) | priv: -/-/0.3079 | - | out=runs/blend-team-nn-v4; pub 0.3203, all 0.3116 |
| 2026-11-15 | danya | - | catboost/QuerySoftMax pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3669/0.3669/0.3061 | - | out=runs/cb32-is-i2i-qsm; feats=$H32+itemstats-032+item2item-032 (109), GPU, best_iter=923; pub 0.3181, all 0.3097; --no-test |
| 2026-11-16 | danya | - | catboost/QuerySoftMax pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3677/0.3677/0.3077 | - | out=runs/cb32-is-i2i-qsm-cv; feats=109, 32 parts, GPU, iters 923; pub 0.3167, all 0.3104; vs cb8-is-i2i-qsm-cv priv +0.00220 [+0.00041, +0.00396] (p=0.992), Финал B (select_final), accepted; validator OK |
| 2026-11-16 | danya | - | lightgbm/default pos=none mode=holdout extra=- depth=4 lr=0.02024 | priv: 0.3637/0.3638/0.3029 | - | out=runs/lgb32-is-i2i-opt; 32 parts, CPU, pub 0.3164, all 0.3069; --no-test |
| 2026-11-16 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=4 lr=0.02024 | priv: 0.3644/0.3644/0.3034 | - | out=runs/lgb32-is-i2i-opt-cv; 32 parts, CPU, pub 0.3153, all 0.3070; rejected vs lgb8-is-i2i-opt-cv |
| 2026-11-16 | danya | - | blend weights 8 OOF (cb32-is-i2i-qsm-cv + cb8-is-i2i-qsm-cv-seeds + cb8-is-i2i-qsm-opt-cv + cb32-qsm-cv-seeds + lgb8-cv-seeds + nn-oof4) | priv: -/-/0.3090 | 0.3622419578 | out=runs/blend-oof-v5; pub 0.3218, all 0.3128; vs blend32-dima priv +0.00444 [+0.00229, +0.00659] (p=1.000), vs blend-team-v3 priv +0.00150 [-0.00005, +0.00309] (all +0.00161 [+0.00030, +0.00289], p=0.992); Public LB 0.36224 (~320 место); validator OK |
| 2026-11-16 | danya | - | blend weights 9 models (8 OOF + dima-joint008) | priv: -/-/0.3087 | - | out=runs/blend-team-v5; pub 0.3218, all 0.3126; validator OK |

## Этап 5: все 9 групп признаков (`category-cooc`, `event-cosine`, `feature-ranks`, `time`) на 8 и 32 шардах (Даня) — 16.11.2026

Аудиты пройдены (`"complete": true`): `category-cooc-008/032`, `event-cosine` (8/32), `feature-ranks-008/032`, `time-008/032` (`runs/*_audit.json`). Всего признаков: 130.

| дата | ветка | - | модель | pseudo-private NDCG full / idcg30 / @30 | Public | примечание |
|---|---|---|---|---|---|---|
| 2026-11-16 | danya | - | catboost/QuerySoftMax pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3658/0.3658/0.3048 | - | out=runs/cb8-all-qsm; 9 групп (130 признаков), 8 parts, GPU, best_iter=291; pub 0.3163, all 0.3083; --no-test |
| 2026-11-16 | danya | - | catboost/QuerySoftMax pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3668/0.3669/0.3066 | - | out=runs/cb8-all-qsm-cv; 130 feats, 8 parts, GPU, iters 291; pub 0.3180, all 0.3100; vs cb8-is-i2i-qsm-cv priv +0.00110 [-0.00052, +0.00276] (p=0.908); validator OK |
| 2026-11-16 | danya | - | lightgbm/default pos=none mode=holdout extra=- depth=4 lr=0.02024 | priv: 0.3645/0.3645/0.3042 | - | out=runs/lgb8-all-opt; 130 feats, 8 parts, CPU, pub 0.3176, all 0.3082; --no-test |
| 2026-11-16 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=4 lr=0.02024 | priv: 0.3654/0.3655/0.3050 | - | out=runs/lgb8-all-opt-cv; 130 feats, 8 parts, CPU, pub 0.3159, all 0.3083 |
| 2026-11-16 | danya | - | catboost/QuerySoftMax pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3672/0.3672/0.3069 | - | out=runs/cb32-all-qsm; 130 feats, 32 parts, GPU, best_iter=403; pub 0.3177, all 0.3101; --no-test |
| 2026-11-16 | danya | - | catboost/QuerySoftMax pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3680/0.3680/0.3077 | - | out=runs/cb32-all-qsm-cv; 130 feats, 32 parts, GPU, iters 403; pub 0.3173, all 0.3106; vs cb32-is-i2i-qsm-cv pub +0.00066, all +0.00023, Финал B (select_final); validator OK |
| 2026-11-16 | danya | - | catboost/default pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3642/0.3642/0.3028 | - | out=runs/cb32-all-yeti; 130 feats, 32 parts, GPU, best_iter=341; pub 0.3163, all 0.3068; --no-test |
| 2026-11-16 | danya | - | catboost/default pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3650/0.3650/0.3041 | - | out=runs/cb32-all-yeti-cv; 130 feats, 32 parts, GPU, iters 341; pub 0.3161, all 0.3077; validator OK |
| 2026-11-16 | danya | - | blend weights 7 OOF (без 73-фич балласта) | priv: -/-/0.3078 | - | out=runs/blend-oof-v6; pub 0.3213 (mean: priv 0.3088) |
| 2026-11-16 | danya | - | blend weights 9 OOF (cb32-all-qsm-cv + cb32-is-i2i-qsm-cv + cb8-all-qsm-cv + cb8-is-i2i-qsm-cv-seeds + lgb8-is-i2i-opt-cv + cb32-qsm-cv-seeds + lgb8-cv-seeds + nn-oof4) | priv: -/-/0.3098 | - | out=runs/blend-oof-v7; pub 0.3223, all 0.3136 (mean: pub 0.3205, priv 0.3093); vs blend-oof-v5 priv +0.00085 [-0.00028, +0.00192] (p=0.935, all p=0.952), Финал A (select_final); validator OK |
| 2026-11-16 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=4 lr=0.02359 | priv: 0.3669/0.3669/0.3071 | - | out=runs/lgb8-all-opt2-cv; Optuna runs/tune-lgb-all (130 feats, min_data=252, best pub 0.3187), 361 iters, pub 0.3173, all 0.3101 |
| 2026-11-16 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=4 lr=0.02359 | priv: 0.3661/0.3662/0.3061 | - | out=runs/lgb32-all-opt2-cv; 32 parts, pub 0.3153, all 0.3089; уступает 8-шардовой lgb8-all-opt2-cv |
| 2026-11-16 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=4 lr=0.02359 | priv: 0.3661/0.3661/0.3066 | - | out=runs/lgb8-all-opt2-cv-s1; seed=1, pub 0.3173, all 0.3098 |
| 2026-11-16 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=4 lr=0.02359 | priv: 0.3669/0.3669/0.3074 | - | out=runs/lgb8-all-opt2-cv-s2; seed=2, pub 0.3188, all 0.3109 |
| 2026-11-16 | danya | - | seed-avg mean lgb8-all-opt2-cv + s1 + s2 | priv: -/-/0.3071 | - | out=runs/lgb8-all-opt2-cv-seeds; pub 0.3185 (+0.00122 vs seed 0, p=0.952), all 0.3105; validator OK |
| 2026-11-16 | danya | - | catboost/QuerySoftMax pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3677/0.3678/0.3076 | - | out=runs/cb32-all-qsm-cv-s1; seed=1, 130 feats, 32 parts, GPU, iters 403, pub 0.3188, all 0.3109; validator OK |
| 2026-11-16 | danya | - | blend weights 9 OOF (cb32-all-qsm-cv-s1 + lgb8-all-opt2-cv-seeds + cb32-is-i2i-qsm-cv + cb8-all-qsm-cv + nn-oof4 ...) | priv: -/-/0.3091 | - | out=runs/blend-oof-v8; pub 0.3223 (mean: pub 0.3205, priv 0.3098) |
| 2026-11-16 | danya | - | catboost/QuerySoftMax pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3652/0.3652/0.3041 | - | out=runs/cb8-all-fi-qsm; 152 feats (--feed-interactions), 8 parts, GPU, best_iter=306; pub 0.3165, all 0.3078; --no-test |
| 2026-11-16 | danya | - | catboost/QuerySoftMax pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3675/0.3676/0.3074 | - | out=runs/cb8-all-fi-qsm-cv; 152 feats, 8 parts, GPU, iters 306; pub 0.3194, all 0.3110; vs cb8-all-qsm-cv all +0.00101 (p=0.909) |
| 2026-11-16 | danya | - | lightgbm/default pos=none mode=cv extra=- depth=4 lr=0.02359 | priv: 0.3671/0.3672/0.3072 | - | out=runs/lgb8-all-fi-opt2-cv; 152 feats (--feed-interactions), 8 parts, CPU, pub 0.3183, all 0.3105 |
| 2026-11-16 | danya | - | catboost/QuerySoftMax pos=none mode=holdout extra=- depth=8 lr=0.05 | priv: 0.3670/0.3671/0.3066 | - | out=runs/cb32-all-fi-qsm; 152 feats, 32 parts, GPU, best_iter=518; pub 0.3181, all 0.3101; --no-test |
| 2026-11-16 | danya | - | catboost/QuerySoftMax pos=none mode=cv extra=- depth=8 lr=0.05 | priv: 0.3679/0.3679/0.3074 | - | out=runs/cb32-all-fi-qsm-cv; 152 feats, 32 parts, GPU, iters 518; pub 0.3193, all 0.3110 (all_ndcg_full 0.3703); validator OK |
| 2026-11-16 | danya | - | blend weights 10 OOF (152-feat + 130-feat + 73-feat + nn-oof4) | priv: -/-/0.3100 | - | out=runs/blend-oof-v9; pub 0.3219, all 0.3136; validator OK |
| 2026-11-16 | danya | - | blend mean 10 OOF (152-feat + 130-feat + 73-feat + nn-oof4) | priv: -/-/0.3107 | - | out=runs/blend-oof-v9-mean; pub 0.3210, all 0.3138; vs blend-oof-v5 priv +0.00175 [+0.00027, +0.00324] (p=0.992), Финал A (select_final); validator OK |
| 2026-11-16 | danya | - | nn-is-oof4 (Август, 4-fold OOF с itemstats) | priv: -/-/0.3016 | - | out=runs/nn-is-oof4; pub 0.3131, all 0.3050; vs nn-oof4 priv +0.00091 [-0.00186, +0.00387]; validator OK |
| 2026-11-16 | danya | - | blend weights 8 OOF (пул blend-oof-v5 с nn-is-oof4 вместо nn-oof4) | priv: -/-/0.3095 | - | out=runs/blend-oof-v5-nnis; pub 0.3234, all 0.3137; vs blend-oof-v5 priv +0.00059 [-0.00096, +0.00224] |
| 2026-11-16 | danya | - | blend weights 10 OOF (пул v9 с nn-is-oof4 вместо nn-oof4) | priv: -/-/0.3096 | - | out=runs/blend-oof-v10; pub 0.3241 (mean: pub 0.3212, priv 0.3102); validator OK |
| 2026-11-16 | danya | - | blend mean 11 OOF (пул v9 + nn-oof4 + nn-is-oof4) | priv: -/-/0.3103 | - | out=runs/blend-oof-v10-bothnn; pub 0.3217, all 0.3137; validator OK |
