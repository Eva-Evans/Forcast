from sqlalchemy import create_engine

from config import POSTGRES_DSN

_connect_args: dict = {}
if "sslmode=require" in POSTGRES_DSN or "supabase.co" in POSTGRES_DSN:
    _connect_args["sslmode"] = "require"

engine = create_engine(
    POSTGRES_DSN,
    echo=False,
    future=True,
    pool_pre_ping=True,
    connect_args=_connect_args,
)
