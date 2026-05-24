// Drop into: frontend/src/lib/sessionCache.ts
//
// IndexedDB cache for the last transcription session. Stores the original audio
// Blob + the raw notes (before melody_cleanup), so when the user changes their
// CleanupOptions preset we POST to /api/recleanup instead of re-uploading the
// audio and re-running basic-pitch.
//
// Quota: IndexedDB gives us hundreds of MB (vs. localStorage's ~5 MB), so we
// can store WAV blobs natively without base64 inflation.

const DB_NAME = "musicme";
const DB_VERSION = 1;
const STORE = "session";
const KEY = "current";

export type Note = {
  pitch: number;
  start: number;
  end: number;
  velocity: number;
  tied_to_next: boolean;
};

export type CleanupOptions = {
  preset?: "beginner" | "standard" | "expert";
  repeat_strategy?: "merge" | "tie" | "split";
  max_gap?: number;
};

export type CachedSession = {
  audio: Blob;
  rawNotes: Note[];
  cleanedNotes: Note[];
  options: CleanupOptions;
  createdAt: number;
};

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await openDb();
  return new Promise<T>((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const req = fn(t.objectStore(STORE));
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
    t.oncomplete = () => db.close();
  });
}

export const sessionCache = {
  save: (s: CachedSession) => tx("readwrite", (store) => store.put(s, KEY)),
  load: () => tx<CachedSession | undefined>("readonly", (store) => store.get(KEY)),
  clear: () => tx("readwrite", (store) => store.delete(KEY)),
};

// ---------------------------------------------------------------------------
// Usage example (in your transcribe page / recorder component):
// ---------------------------------------------------------------------------
//
// async function transcribe(audio: Blob, options: CleanupOptions) {
//   const form = new FormData();
//   form.append("file", audio, "humming.wav");
//   form.append("options", JSON.stringify(options));
//   const res = await fetch("/api/transcribe", { method: "POST", body: form });
//   const data = await res.json();
//   await sessionCache.save({
//     audio,
//     rawNotes: data.raw_notes,
//     cleanedNotes: data.notes,
//     options,
//     createdAt: Date.now(),
//   });
//   return data.notes as Note[];
// }
//
// async function changePreset(options: CleanupOptions) {
//   const cached = await sessionCache.load();
//   if (!cached) throw new Error("No cached session");
//   const res = await fetch("/api/recleanup", {
//     method: "POST",
//     headers: { "Content-Type": "application/json" },
//     body: JSON.stringify({ raw_notes: cached.rawNotes, options }),
//   });
//   const data = await res.json();
//   await sessionCache.save({ ...cached, cleanedNotes: data.notes, options });
//   return data.notes as Note[];
// }
