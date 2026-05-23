from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="pythonbackend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_items: list[dict] = []
_next_id = 1


class ItemCreate(BaseModel):
    name: str


class ItemResponse(BaseModel):
    id: int
    name: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/items", response_model=list[ItemResponse])
def list_items():
    return _items


@app.get("/api/items/{item_id}", response_model=ItemResponse)
def get_item(item_id: int):
    for item in _items:
        if item["id"] == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")


@app.post("/api/items", response_model=ItemResponse, status_code=201)
def create_item(payload: ItemCreate):
    global _next_id
    item = {"id": _next_id, "name": payload.name}
    _next_id += 1
    _items.append(item)
    return item


@app.delete("/api/items/{item_id}", status_code=204)
def delete_item(item_id: int):
    for index, item in enumerate(_items):
        if item["id"] == item_id:
            _items.pop(index)
            return
    raise HTTPException(status_code=404, detail="Item not found")
