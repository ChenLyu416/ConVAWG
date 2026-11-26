"""Storage module for persisting structured case data."""

import json
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from llm_structurer.schema import Case


logger = logging.getLogger(__name__)
Base = declarative_base()


class CaseDB(Base):
    """SQLAlchemy model for Case storage."""
    
    __tablename__ = 'cases'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String, unique=True, nullable=False, index=True)
    raw_text = Column(Text, nullable=False)
    source = Column(String)
    date_processed = Column(DateTime, default=datetime.utcnow)
    
    actors_json = Column(JSON)
    relationships_json = Column(JSON)
    events_json = Column(JSON)
    violence_types_json = Column(JSON)
    legal_outcomes_json = Column(JSON)
    summary = Column(Text)
    metadata_json = Column(JSON)


class CaseStorage:
    """Handles storage of structured cases to database and files."""

    def __init__(self, db_path: str = "outputs/cases.db"):
        """
        Initialize storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        self.logger = logging.getLogger(__name__)

    def save_case_to_db(self, case: Case):
        """
        Save a case to the database.

        Args:
            case: Case object to save
        """
        session: Session = self.SessionLocal()
        
        try:
            case_db = CaseDB(
                case_id=case.case_id,
                raw_text=case.raw_text,
                source=case.source,
                date_processed=case.date_processed,
                actors_json=[a.model_dump() for a in case.actors],
                relationships_json=[r.model_dump() for r in case.relationships],
                events_json=[e.model_dump() for e in case.events],
                violence_types_json=[str(v.value) for v in case.violence_types],
                legal_outcomes_json=case.legal_outcomes.model_dump() if case.legal_outcomes else None,
                summary=case.summary,
                metadata_json=case.metadata
            )
            
            session.merge(case_db)
            session.commit()
            self.logger.info(f"Saved case {case.case_id} to database")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to save case to database: {e}")
            raise
        finally:
            session.close()

    def save_case_to_json(self, case: Case, output_dir: str):
        """
        Save a case to JSON file.

        Args:
            case: Case object to save
            output_dir: Directory to save JSON file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / f"{case.case_id}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(case.model_dump(mode='json'), f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Saved case {case.case_id} to {file_path}")

    def save_case_to_jsonl(self, case: Case, output_file: str):
        """
        Append a case to JSONL file.

        Args:
            case: Case object to save
            output_file: Path to JSONL file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(case.model_dump(mode='json'), ensure_ascii=False, default=str) + '\n')
        
        self.logger.info(f"Appended case {case.case_id} to {output_file}")

    def load_case_from_db(self, case_id: str) -> Optional[Case]:
        """
        Load a case from the database.

        Args:
            case_id: ID of case to load

        Returns:
            Case object or None
        """
        session: Session = self.SessionLocal()
        
        try:
            case_db = session.query(CaseDB).filter(CaseDB.case_id == case_id).first()
            
            if not case_db:
                return None
            
            case_dict = {
                'case_id': case_db.case_id,
                'raw_text': case_db.raw_text,
                'source': case_db.source,
                'date_processed': case_db.date_processed,
                'actors': case_db.actors_json or [],
                'relationships': case_db.relationships_json or [],
                'events': case_db.events_json or [],
                'violence_types': case_db.violence_types_json or [],
                'legal_outcomes': case_db.legal_outcomes_json,
                'summary': case_db.summary,
                'metadata': case_db.metadata_json or {}
            }
            
            return Case(**case_dict)
            
        except Exception as e:
            self.logger.error(f"Failed to load case from database: {e}")
            return None
        finally:
            session.close()

    def export_all_to_jsonl(self, output_file: str):
        """
        Export all cases from database to JSONL file.

        Args:
            output_file: Path to output JSONL file
        """
        session: Session = self.SessionLocal()
        
        try:
            cases_db = session.query(CaseDB).all()
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for case_db in cases_db:
                    case_dict = {
                        'case_id': case_db.case_id,
                        'raw_text': case_db.raw_text,
                        'source': case_db.source,
                        'date_processed': case_db.date_processed.isoformat() if case_db.date_processed else None,
                        'actors': case_db.actors_json or [],
                        'relationships': case_db.relationships_json or [],
                        'events': case_db.events_json or [],
                        'violence_types': case_db.violence_types_json or [],
                        'legal_outcomes': case_db.legal_outcomes_json,
                        'summary': case_db.summary,
                        'metadata': case_db.metadata_json or {}
                    }
                    f.write(json.dumps(case_dict, ensure_ascii=False) + '\n')
            
            self.logger.info(f"Exported {len(cases_db)} cases to {output_file}")
            
        finally:
            session.close()
