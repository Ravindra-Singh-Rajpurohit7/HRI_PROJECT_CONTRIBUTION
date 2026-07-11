from sqlalchemy import Column, Integer, String, JSON, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from database import Base
import datetime


class User(Base):
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    rapport_level = Column(Integer, default=0)
    user_preference = Column(JSON, default={})

    disclosures = relationship("Disclosure", back_populates="user")
    conversation_contexts = relationship(
        "UserConversationContext",
        back_populates="user"
    )


class Disclosure(Base):
    __tablename__ = 'disclosures'

    disclosure_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    memory_id = Column(String, index=True)
    memory_topic = Column(String)
    disclosure_timestamp = Column(
        DateTime,
        default=datetime.datetime.utcnow
    )

    user = relationship("User", back_populates="disclosures")


class UserConversationContext(Base):
    """
    Stores what the USER shared in each session —
    their situation, feelings, topics — so Misty can
    remember and continue naturally in the next session.

    One row per session per user.
    """
    __tablename__ = 'user_conversation_contexts'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    session_id = Column(Integer, nullable=False)

    # LLM-generated summary of what the user shared this session
    user_summary = Column(Text, nullable=True)

    # Key topics the user mentioned (comma-separated)
    topics = Column(String, nullable=True)

    # Full last exchange (last user message + Misty reply) for
    # conversation continuity — "pick up from where we left off"
    last_user_message = Column(Text, nullable=True)
    last_bot_response = Column(Text, nullable=True)

    created_at = Column(
        DateTime,
        default=datetime.datetime.utcnow
    )

    user = relationship("User", back_populates="conversation_contexts")