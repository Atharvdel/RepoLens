import sys
from pathlib import Path
from collections import Counter
import sqlalchemy as sa

BACKEND = Path(r"C:\Users\Atharv Sharma\Desktop\Work\RepoLens\backend")
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.db import SessionLocal
from app.indexing import ImportResolver, parse_imports
from app.models import Edge, File

REPO_ID = 2
SOURCE_PATH = "src/flask/app.py"
REPO_PATH = r"C:\Users\Atharv Sharma\Desktop\Work\flask"

with SessionLocal() as session:
    f = session.execute(
        sa.select(File).where(File.repository_id == REPO_ID, File.path == SOURCE_PATH)
    ).scalar_one_or_none()
    if f is None:
        print(f"NO file row for repo_id={REPO_ID} path={SOURCE_PATH!r}")
        sys.exit(0)
    print(f"file: {SOURCE_PATH}  (file_id={f.id})")

    files = session.execute(
        sa.select(File).where(File.repository_id == REPO_ID)
    ).scalars().all()
    id_to_path = {x.id: x.path for x in files}
    resolver = ImportResolver((x.id, x.path) for x in files)

    edges = session.execute(
        sa.select(Edge).where(
            Edge.repository_id == REPO_ID,
            Edge.source_id == f.id,
            Edge.source_type == "file",
            Edge.edge_type == "imports",
        )
    ).scalars().all()

    db_keys = Counter(
        (e.target_type, (e.target_label if e.target_type == "external"
                         else id_to_path.get(e.target_id, "<id " + str(e.target_id) + ">")))
        for e in edges
    )
    persist_n = len(edges)
    n_int_db = sum(1 for e in edges if e.target_type == "file")
    n_ext_db = sum(1 for e in edges if e.target_type == "external")
    print(f"persisted imports edges: {persist_n}  ({n_int_db} internal, {n_ext_db} external)")

    rows = []
    recon_ok = False
    try:
        abs_path = Path(REPO_PATH) / SOURCE_PATH
        for imp in parse_imports(abs_path):
            rid = resolver.resolve(SOURCE_PATH, imp)
            if rid is not None:
                rows.append((imp.level, "internal", id_to_path.get(rid, "<id " + str(rid) + ">")))
            else:
                rows.append((imp.level, "external", "." * imp.level + imp.target))
        recon_ok = True
    except Exception as exc:
        print(f"WARNING: source reconstruction failed ({exc!r})")
        for e in edges:
            if e.target_type == "external":
                lbl = e.target_label or ""
                rows.append((len(lbl) - len(lbl.lstrip(".")), "external", lbl))
            else:
                rows.append((-1, "internal", id_to_path.get(e.target_id, "<id " + str(e.target_id) + ">")))

    if recon_ok:
        rc_keys = Counter((k, t) for (_l, k, t) in rows)
        if persist_n and rc_keys == db_keys:
            print("cross-check: OK  (persisted edges match source reconstruction 1:1)")
        elif persist_n and rc_keys != db_keys:
            print("cross-check: MISMATCH  (persisted vs source-derived differ):")
            if db_keys - rc_keys:
                print("  only in DB    : " + repr(dict(db_keys - rc_keys)))
            if rc_keys - db_keys:
                print("  only in source: " + repr(dict(rc_keys - db_keys)))
        elif persist_n == 0:
            print("cross-check: DB has 0 edges; source reconstruction shows what *would* be indexed")

    print("-" * 70)
    rows.sort(key=lambda r: (r[0] if r[0] is not None else -1, r[2]))
    for lvl, kind, tgt in rows:
        print(f"{str(lvl):<6}{kind:<10}{tgt}")
    print("-" * 70)
    n_int = sum(1 for _l, k, _t in rows if k == "internal")
    n_ext = sum(1 for _l, k, _t in rows if k == "external")
    print(f"total: {len(rows)}  ({n_int} internal, {n_ext} external)")
