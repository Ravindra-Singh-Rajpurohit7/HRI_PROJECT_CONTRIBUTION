from database import SessionLocal
from models import User


def create_user(user_name):

    db = SessionLocal()

    try:

        existing = (
            db.query(User)
            .filter(User.name == user_name)
            .first()
        )

        if existing:

            print(
                f"User already exists. ID={existing.user_id}"
            )

            return existing.user_id

        user = User(
            name=user_name,
            rapport_level=0,
            user_preference={}
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        print(
            f"Created user. ID={user.user_id}"
        )

        return user.user_id

    finally:
        db.close()


if __name__ == "__main__":

    name = input("Enter User Name: ")

    create_user(name)