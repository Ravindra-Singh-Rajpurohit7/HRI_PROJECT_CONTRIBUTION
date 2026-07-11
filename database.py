from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Yeh add karo top pe
import os
from dotenv import load_dotenv
load_dotenv()

# Yeh badlo:
DATABASE_URL = os.environ.get("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Call once at startup to create any missing tables.
    Existing tables are NOT dropped or modified.
    """
    from models import Base as ModelsBase
    ModelsBase.metadata.create_all(bind=engine)
if __name__ == "__main__":
    print("Initializing Database...")
    init_db()