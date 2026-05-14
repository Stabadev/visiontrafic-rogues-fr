# VisionTrafic

Application Flask + worker YOLO pour detecter des vehicules depuis des webcams publiques.

## Lancement local

Creer un environnement Python puis installer les dependances :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Variables locales recommandees :

```bash
export INFOROUTE_OUT_DIR=./data
export INFOROUTE_DB_PATH=./data/events.sqlite3
export YOLO_CONFIG_DIR=./data/.ultralytics
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=-1
```

Lancer le web :

```bash
flask --app api run --host 127.0.0.1 --port 9107 --debug
```

Lancer le worker dans un autre terminal :

```bash
python run_yolo_cycle.py
```

## Lancement Docker

Creer le fichier d'environnement Docker :

```bash
cp .env.example .env
```

Construire et lancer les services :

```bash
docker compose up --build
```

Web seul :

```bash
docker compose up --build web
```

Worker seul :

```bash
docker compose up --build worker
```

L'application web est disponible sur :

```text
http://127.0.0.1:9107
```

## Services Docker

- `web` : lance Flask via Gunicorn sur le port `9107`.
- `worker` : lance `python run_yolo_cycle.py`.

Les deux services utilisent la meme image Docker et partagent le volume persistant :

```text
visiontrafic_data:/app/data
```

## Donnees runtime

Les images, le fichier SQLite et les fichiers d'etat du worker sont ecrits dans `/app/data` dans Docker.

En local, utiliser plutot :

```text
./data
```
