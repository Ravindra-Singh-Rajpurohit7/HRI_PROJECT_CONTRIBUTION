from database import engine, Base
import models

def init_db():
    print("Database tables create ho rahi hain...")
    Base.metadata.create_all(bind=engine)
    print("Tables successfully create ho gayi hain!")

if __name__ == "__main__":
    init_db()