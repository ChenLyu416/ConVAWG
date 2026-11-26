"""Pydantic schema definitions for VAWG case data structures."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class ActorRole(str, Enum):
    """Enumeration of actor roles in a case."""
    VICTIM = "victim"
    PERPETRATOR = "perpetrator"
    THIRD_PARTY = "third_party"
    WITNESS = "witness"
    LEGAL_REPRESENTATIVE = "legal_representative"
    UNKNOWN = "unknown"


class RelationshipType(str, Enum):
    """Types of relationships between actors."""
    INTIMATE_PARTNER = "intimate_partner"
    EX_PARTNER = "ex_partner"
    FAMILY = "family"
    ACQUAINTANCE = "acquaintance"
    STRANGER = "stranger"
    PROFESSIONAL = "professional"
    OTHER = "other"
    UNKNOWN = "unknown"


class ViolenceType(str, Enum):
    """Types of violence in VAWG cases."""
    PHYSICAL = "physical"
    SEXUAL = "sexual"
    PSYCHOLOGICAL = "psychological"
    ECONOMIC = "economic"
    STALKING = "stalking"
    HARASSMENT = "harassment"
    COERCIVE_CONTROL = "coercive_control"
    DIGITAL = "digital"
    HONOR_BASED = "honor_based"
    FORCED_MARRIAGE = "forced_marriage"
    FGM = "fgm"
    TRAFFICKING = "trafficking"
    OTHER = "other"


class EventType(str, Enum):
    """Types of events in case timeline."""
    INCIDENT = "incident"
    REPORT = "report"
    COMMUNICATION = "communication"
    CONTACT = "contact"
    ARREST = "arrest"
    CHARGE = "charge"
    INVESTIGATION = "investigation"
    INTERVIEW = "interview"
    COURT_HEARING = "court_hearing"
    VERDICT = "verdict"
    SENTENCING = "sentencing"
    PROTECTIVE_ORDER = "protective_order"
    APPEAL = "appeal"
    SERVICE_INTERVENTION = "service_intervention"
    MEDICAL_TREATMENT = "medical_treatment"
    OTHER = "other"


class Actor(BaseModel):
    """Represents a person involved in the case."""
    actor_id: str = Field(..., description="Unique identifier for the actor")
    name: Optional[str] = Field(None, description="Actor's name (may be anonymized)")
    role: ActorRole = Field(..., description="Role in the case")
    age: Optional[str] = Field(None, description="Age at time of incident (can be exact age or range like '30s', 'mid-20s')")
    gender: Optional[str] = Field(None, description="Gender identity")
    occupation: Optional[str] = Field(None, description="Occupation or employment status")
    ethnicity: Optional[str] = Field(None, description="Ethnicity or cultural background")
    marital_status: Optional[str] = Field(None, description="Marital or relationship status")
    demographics: Dict[str, Any] = Field(default_factory=dict, description="Additional demographic information")
    mentions: List[str] = Field(default_factory=list, description="Text spans mentioning this actor")

    class Config:
        use_enum_values = True


class Relationship(BaseModel):
    """Represents a relationship between two actors."""
    relationship_id: str = Field(..., description="Unique identifier for the relationship")
    actor_1_id: str = Field(..., description="First actor ID")
    actor_2_id: str = Field(..., description="Second actor ID")
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    description: Optional[str] = Field(None, description="Textual description of the relationship")
    temporal_info: Optional[str] = Field(None, description="Temporal information (e.g., 'former', 'current')")

    class Config:
        use_enum_values = True


class Event(BaseModel):
    """Represents a discrete event in the case timeline."""
    event_id: str = Field(..., description="Unique identifier for the event")
    event_type: EventType = Field(..., description="Type of event")
    description: str = Field(..., description="Description of the event")
    participants: List[str] = Field(default_factory=list, description="Actor IDs involved")
    temporal_order: Optional[int] = Field(None, description="Sequence order in timeline")
    date_time: Optional[str] = Field(None, description="Date/time if mentioned")
    location: Optional[str] = Field(None, description="Location if mentioned")
    text_span: Optional[str] = Field(None, description="Relevant text excerpt")

    class Config:
        use_enum_values = True


class Case(BaseModel):
    """Complete structured representation of a VAWG case."""
    case_id: str = Field(..., description="Unique case identifier")
    raw_text: str = Field(..., description="Original case narrative text")
    source: Optional[str] = Field(None, description="Source of the case document")
    date_processed: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")
    
    actors: List[Actor] = Field(default_factory=list, description="All actors in the case")
    relationships: List[Relationship] = Field(default_factory=list, description="Relationships between actors")
    events: List[Event] = Field(default_factory=list, description="Timeline of events")
    violence_types: List[ViolenceType] = Field(default_factory=list, description="Types of violence present")
    summary: Optional[str] = Field(None, description="Concise case summary")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        use_enum_values = True

    @field_validator('violence_types', mode='before')
    @classmethod
    def validate_violence_types(cls, v):
        """Ensure violence_types are valid ViolenceType values."""
        if isinstance(v, list):
            return [ViolenceType(item) if isinstance(item, str) else item for item in v]
        return v
