# pythonbackend

Backend Python minimal avec [FastAPI](https://fastapi.tiangolo.com/).

## Prérequis

- Python 3.10+

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer le serveur

```powershell
uvicorn app.main:app --reload
```

- API : http://127.0.0.1:8000
- Documentation interactive : http://127.0.0.1:8000/docs

## Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/health` | Vérification de santé |
| GET | `/api/items` | Liste des items |
| GET | `/api/items/{id}` | Détail d'un item |
| POST | `/api/items` | Créer un item (`{"name": "..."}`) |
| DELETE | `/api/items/{id}` | Supprimer un item |

Les items sont stockés en mémoire (réinitialisés à chaque redémarrage du serveur).
