import sqlalchemy as sa
from app.db import SessionLocal
from app.models import File, Symbol

paths = ['src/flask/cli.py', 'src/flask/helpers.py', 'src/flask/blueprints.py']
with SessionLocal() as s:
    rows = s.execute(
        sa.select(File.path, Symbol.name, Symbol.kind, Symbol.line_start, Symbol.line_end, Symbol.docstring)
        .join(Symbol, Symbol.file_id == File.id)
        .where(File.repository_id == 2, File.path.in_(paths))
        .order_by(File.path, Symbol.line_start)
    ).all()

for p, n, k, ls, le, d in rows:
    doc = "yes" if d else "no"
    print(f"{p:<28} {k:<9} {ls:>5} {le:>5}  {doc:<4}  {n}")
print(f"total: {len(rows)} symbols")