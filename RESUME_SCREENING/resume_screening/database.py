from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:///candidates.db")
Base = declarative_base()

class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    score = Column(Float)
    skills = Column(String)

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()


def save_candidate(name, email, score, skills):
    candidate = Candidate(
        name=name,
        email=email,
        score=score,
        skills=", ".join(skills)
    )
    session.add(candidate)
    session.commit()


def get_all_candidates():
    return session.query(Candidate).order_by(Candidate.score.desc()).all()
def delete_all_candidates():
    session.query(Candidate).delete()
    session.commit()