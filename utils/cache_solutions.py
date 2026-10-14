from pathlib import Path
import sqlite3
import io
import numpy as np
import hashlib

# Generate a stable hash key from (mu_glob, f)
def _make_key(mu_glob, f):
    if not isinstance(mu_glob, np.ndarray):
        raise TypeError("mu_glob must be a numpy ndarray")
    if mu_glob.shape != (6,):
        raise ValueError("mu_glob must have length 6")

    # Accept int or float for f
    if isinstance(f, (int, np.integer)):
        f = float(f)
    if not isinstance(f, float) and not np.issubdtype(type(f), np.floating):
        raise TypeError("f must be a float or an int")

    h = hashlib.sha256()
    h.update(mu_glob.astype(np.float64).tobytes())
    h.update(np.float64(f).tobytes())
    return h.hexdigest()

def _array_to_blob(arr: np.ndarray) -> bytes:
    """Serialize a numpy array to raw bytes using np.save."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()

def _blob_to_array(blob: bytes) -> np.ndarray:
    """Deserialize raw bytes produced by _array_to_blob back to a numpy array."""
    buf = io.BytesIO(blob)
    buf.seek(0)
    return np.load(buf, allow_pickle=False)

class CacheVectors:
    def __init__(self, name: str, folder: Path):
        # Ensure the folder exists
        db_folder = Path(folder)
        db_folder.mkdir(parents=True, exist_ok=True)

        # Database path (single SQLite file)
        self.path = db_folder / f"{name}.sqlite"

        # Ask before creating a new database file
        if not self.path.exists():
            print(f"Database {self.path} created")
            # Touch the file
            self.path.touch()

        # Open SQLite connection
        self.conn = sqlite3.connect(self.path)
        self._init_table()

    def _init_table(self):
        """Create table if it does not exist."""
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                key TEXT PRIMARY KEY,
                mu_glob BLOB NOT NULL,
                f REAL NOT NULL,
                vector BLOB NOT NULL
            )
            """
        )
        self.conn.commit()

    def close(self):
        """Close the underlying SQLite connection."""
        if getattr(self, "conn", None) is not None:
            self.conn.close()
            self.conn = None

    # You can also use: with CacheVecteurs("name") as cache:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __contains__(self, key):
        """Check if (mu_glob, f) is present in the cache."""
        mu_glob, f = key
        k = _make_key(mu_glob, f)
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM results WHERE key = ?", (k,))
        row = cur.fetchone()
        return row is not None

    def __getitem__(self, key):
        """Return the stored vector for (mu_glob, f)."""
        mu_glob, f = key
        k = _make_key(mu_glob, f)
        cur = self.conn.cursor()
        cur.execute("SELECT vector FROM results WHERE key = ?", (k,))
        row = cur.fetchone()
        if row is None:
            raise KeyError("Key not found in cache")
        vector_blob = row[0]
        return _blob_to_array(vector_blob)

    def __setitem__(self, key, value):
        """
        Store a vector for (mu_glob, f).

        The value must be a numpy array.
        """
        mu_glob, f = key
        if not isinstance(mu_glob, np.ndarray):
            raise TypeError("mu_glob must be a numpy ndarray")
        if mu_glob.shape != (6,):
            raise ValueError("mu_glob must have length 6")
        if not isinstance(value, np.ndarray):
            raise TypeError("Stored value must be a numpy ndarray")

        k = _make_key(mu_glob, f)

        # Normalize f as float for storage
        if isinstance(f, (int, np.integer)):
            f = float(f)

        mu_blob = _array_to_blob(mu_glob.astype(np.float64))
        vec_blob = _array_to_blob(value)

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO results (key, mu_glob, f, vector)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                mu_glob = excluded.mu_glob,
                f       = excluded.f,
                vector  = excluded.vector
            """,
            (k, mu_blob, float(f), vec_blob),
        )
        self.conn.commit()
