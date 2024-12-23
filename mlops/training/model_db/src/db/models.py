from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.sql.functions import current_timestamp
from sqlalchemy.types import JSON
from src.db.database import Base

class Project(Base):

    __tablename__ = "projects"

    project_id = Column(
        String(255),
        primary_key=True,
        comment="主キー"
    )
    project_name = Column(
        String(255),
        nullable=False,
        unique=True
    )