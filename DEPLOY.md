# Развёртывание Streamlit (finál-прогноз)

Подробная инструкция: как поднять приложение, подключить PostgreSQL и выложить код на GitHub / Streamlit Cloud.

---

## Что вы разворачиваете

| Компонент | Назначение |
|-----------|------------|
| **Streamlit** (`app.py`) | UI: вкладки «Прогноз» и «Загрузка данных» |
| **PostgreSQL** | Хранение загруженных Excel/txt (tab3-таблицы) |
| **finál-пайплайн** | `prognoz_vseh_parametrov.py` + `finale_pipeline_extracted.py` (XGBoost, 15 месяцев) |

Без PostgreSQL приложение **не запустится** (импорт вкладок обращается к БД).

---

## Вариант A (рекомендуется): Docker — приложение + Postgres на одном сервере

Подходит для: ваш PC, VPS (Timeweb, Selectel, Hetzner, …), внутренний сервер.

### A.1. Требования

- Docker Desktop (Mac/Windows) или Docker Engine + Compose (Linux)
- Git
- Порты свободны: **8510** (сайт), **15432** (Postgres снаружи, опционально)

### A.2. Клонирование и первый запуск

```bash
git clone https://github.com/Eva-Evans/Forcast.git
cd Forcast

# Если контейнер herd-db уже был создан раньше:
docker compose down
# при необходимости полный сброс БД (УДАЛИТ данные):
# docker compose down -v

docker compose up --build -d
docker compose ps
docker compose logs -f app
```

Откройте в браузере: **http://localhost:8510**  
(или `http://IP_СЕРВЕРА:8510`, если на VPS открыт firewall).

### A.3. Что внутри `docker-compose.yml`

- **`db`**: Postgres 16, БД `herd_forecast`, пользователь `herd_user`, пароль `herd_password`
- **`app`**: Streamlit на порту **8501** внутри контейнера → **8510** снаружи
- **`POSTGRES_DSN`**: `postgresql+psycopg2://herd_user:herd_password@db:5432/herd_forecast` — только для сети Docker
- Том `.:/app` — код монтируется; после `git pull` достаточно перезапустить app:  
  `docker compose restart app`

### A.4. Первый сценарий пользователя

1. **Загрузка данных** — загрузите комплект файлов (отёлы, осеменения, запуск, выбытие, быки), как в старом tab3. Дождитесь статуса «готово» для подразделения.
2. **Прогноз** — выберите подразделение с **✓**, нажмите **«Рассчитать прогноз»**.
3. Расчёт может занять **20–60+ минут**. Следите за логом: `docker compose logs -f app`.
4. Результат: широкая таблица + скачивание Excel.

### A.5. Переменные окружения (Docker)

В `docker-compose.yml` в секции `app.environment` можно добавить:

| Переменная | По умолчанию | Смысл |
|------------|--------------|--------|
| `POSTGRES_DSN` | `@db:5432/...` | Не менять внутри compose |
| `ADMIN_KEY` | `supersecret123` | **Смените в проде** |
| `USE_FINAL_PIPELINE` | `1` | finál (оставить `1`) |
| `FORECAST_HORIZON_MONTHS` | `15` | Горизонт прогноза |
| `SHOW_TAB2_PARAMS` | `0` | Legacy-параметры скрыты |
| `SHOW_TAB3_LEGACY_FARM_FORECAST` | `0` | Legacy-агрегат хозяйства скрыт |
| `PIPELINE_FAST` | `1` | Быстрее XGB (для продакшена можно `0` — дольше, точнее grid) |
| `PIPELINE_WORK_ROOT` | `.pipeline_runtime` | Рабочие файлы finál |

Пример смены пароля админки:

```yaml
environment:
  - ADMIN_KEY=ваш_длинный_секрет
```

### A.6. Обновление версии на сервере

```bash
cd Forcast
git pull origin main
docker compose up --build -d app
# или: make re   # down + up --build
```

### A.7. VPS: доступ из интернета

1. Откройте порт **8510** в firewall / security group.
2. **Не публикуйте 15432** наружу без VPN и сильного пароля.
3. Для HTTPS поставьте reverse proxy (nginx/Caddy) с TLS на `8510` → `localhost:8510`.
4. Смените `ADMIN_KEY` и пароль Postgres в проде (правка `docker-compose.yml` + volume).

Команды makefile:

```bash
make re   # пересборка и запуск
make rev  # down -v (с wipe БД) и up --build
```

---

## Вариант B: Streamlit Community Cloud + внешний Postgres

Подходит, если нужен публичный URL `*.streamlit.app` **без своего VPS**.

**Важно:** Streamlit Cloud **не** запускает ваш `docker-compose` и **не** поднимает Postgres. Нужна **облачная БД** (Neon, Supabase, Railway Postgres, Amazon RDS, …).

### B.1. Подготовить PostgreSQL в облаке

1. Создайте проект Postgres (например Neon).
2. Выполните SQL из репозитория **`db/init.sql`** в новой БД (создаст raw-таблицы).
3. Получите connection string, приведите к формату SQLAlchemy, например:  
   `postgresql+psycopg2://USER:PASSWORD@HOST:5432/herd_forecast?sslmode=require`

### B.2. Залить код на GitHub

См. раздел [«Обновление GitHub»](#обновление-github) ниже. Репозиторий: `https://github.com/Eva-Evans/Forcast.git`

### B.3. Подключить Streamlit Cloud

1. Зайдите на [share.streamlit.io](https://share.streamlit.io) под GitHub-аккаунтом с доступом к репо.
2. **New app** → Repository `Eva-Evans/Forcast`, branch `main`, **Main file path** `app.py`.
3. **Advanced settings** → Python 3.11 (как в Dockerfile; 3.10+ обычно OK).
4. **Secrets** (TOML):

```toml
POSTGRES_DSN = "postgresql+psycopg2://USER:PASS@HOST:5432/herd_forecast?sslmode=require"
ADMIN_KEY = "ваш_секрет"
USE_FINAL_PIPELINE = "1"
FORECAST_HORIZON_MONTHS = "15"
```

5. Deploy. Дождитесь установки `requirements.txt` (есть **xgboost**, **scikit-learn** — сборка может занять несколько минут).

### B.4. Ограничения Streamlit Cloud для этого проекта

| Риск | Пояснение |
|------|-----------|
| **Таймаут сессии** | Долгий finál-прогон (30–60 мин) может оборваться лимитами Cloud. Надёжнее **Docker на VPS**. |
| **Диск** | `.pipeline_runtime` пишется на ephemeral FS — между перезапусками кэш finál пропадает (расчёт просто повторится). |
| **Память/CPU** | XGBoost на больших данных может упереться в лимит тарифа. |
| **Секреты** | Никогда не коммитьте DSN и пароли — только Secrets в UI. |

Если Cloud не тянет по времени — используйте **вариант A** на VPS.

---

## Вариант C: Локально без Docker (только разработка)

Postgres всё равно нужен (удобно поднять только `db` из compose).

```bash
cd Forcast
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Только база в Docker:
docker compose up -d db

export POSTGRES_DSN='postgresql+psycopg2://herd_user:herd_password@127.0.0.1:15432/herd_forecast'
export MPLCONFIGDIR=/tmp/mpl
streamlit run app.py --server.port=8510
```

**Не используйте** дефолтный DSN с хостом `db` вне Docker — будет ошибка «could not translate host name "db"».

---

## Обновление GitHub

Репозиторий: **https://github.com/Eva-Evans/Forcast.git**

### Что не попадает в Git (см. `.gitignore`)

- `.venv/`, `__pycache__/`
- `herd_data.db`, локальные логи, `.pipeline_runtime*`
- `.streamlit/secrets.toml`

### Типичный цикл после правок кода

```bash
cd Forcast
git status
git add app.py config.py core/ ui/ prognoz_vseh_parametrov.py requirements.txt DEPLOY.md README.md .gitignore .streamlit/
git commit -m "Описание изменений"
git push -u origin main
```

Если на Streamlit Cloud включён auto-deploy из `main`, приложение пересоберётся после push.

### Первый push в пустой remote

Если на GitHub уже есть README/license и истории нет локально:

```bash
git pull origin main --allow-unrelated-histories
# решите конфликты при необходимости
git push -u origin main
```

---

## Проверка после деплоя

1. Главная открывается, две вкладки: **Прогноз**, **Загрузка данных**.
2. Загрузка файлов без ошибки SQL (Postgres жив).
3. В списке подразделений после загрузки есть **✓**.
4. «Рассчитать прогноз» — spinner, в логах строки `Шаг: ячейка finál …`.
5. В конце — таблица и Excel.

Служебные скрипты (не обязательны на сервере):

```bash
POSTGRES_DSN='...' python scripts/verify_forecast_e2e.py   # полный прогон из herd_data.db (локально)
python scripts/verify_pipeline_ref_data.py                 # эталон на файлах из ../Прогноз_стada
```

---

## Частые проблемы

| Симптом | Решение |
|---------|---------|
| `could not translate host name "db"` | Локально задайте `POSTGRES_DSN` с `127.0.0.1:15432` или запускайте через `docker compose`. |
| `Conflict … container name "/herd-db"` | `docker compose down`, удалите старый контейнер или `docker rm -f herd-db herd-app`. |
| Подразделение «(нет данных)» | Загрузите полный комплект на вкладке «Загрузка данных». |
| Долго «крутится» без таблицы | Нормально для finál; смотрите `docker compose logs -f app`. |
| Ошибка на `ключ_коровы` | Обновите код с GitHub (исправлено в `tab3_to_final` / `normalize_events_df`). |

---

## English (short)

- **Recommended:** `docker compose up --build` → http://localhost:8510  
- **Streamlit Cloud:** needs external Postgres + Secrets (`POSTGRES_DSN`, `ADMIN_KEY`); long ML runs may hit timeouts — prefer Docker on a VPS.  
- **GitHub:** push `main`; Cloud redeploys automatically if linked.
