from pathlib import Path
import sqlite3

from utils.paths import SNAPSHOTS, MERGED_SNAPSHOTS


# -----------------------------
# Merge process
# -----------------------------
def merge_databases(files, db_folder: Path, output_name="merged.sqlite"):
    files = sorted(files)

    if not files:
        print("No databases to merge")
        return

    print(f"Found {len(files)} databases to merge")

    # Create / overwrite output DB
    output_path = db_folder / output_name
    if output_path.exists():
        print(f"Overwriting existing {output_path}")
        output_path.unlink()

    conn_out = sqlite3.connect(output_path)
    cur_out = conn_out.cursor()

    # Create table
    cur_out.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            key TEXT PRIMARY KEY,
            mu_glob BLOB NOT NULL,
            f REAL NOT NULL,
            vector BLOB NOT NULL
        )
        """
    )
    conn_out.commit()

    # Insert rows from each DB
    total_inserted = 0

    for dbfile in files:
        print(f"Processing {dbfile.name}...")
        conn_in = sqlite3.connect(dbfile)
        cur_in = conn_in.cursor()

        cur_in.execute("SELECT key, mu_glob, f, vector FROM results")
        rows = cur_in.fetchall()

        for k, mu_blob, f_val, vec_blob in rows:
            cur_out.execute(
                """
                INSERT INTO results (key, mu_glob, f, vector)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO NOTHING
                """,
                (
                    k,
                    sqlite3.Binary(mu_blob),
                    float(f_val),
                    sqlite3.Binary(vec_blob),
                ),
            )
            total_inserted += 1

        conn_in.close()
        conn_out.commit()

    conn_out.close()
    print(f"Merge complete. Total entries inserted = {total_inserted}")
    print(f"Merged database saved as: {output_path}")


if __name__ == "__main__":
    files = list(SNAPSHOTS.glob("solutions_*.sqlite"))
    merge_databases(files, MERGED_SNAPSHOTS.parent, output_name=MERGED_SNAPSHOTS.name)
