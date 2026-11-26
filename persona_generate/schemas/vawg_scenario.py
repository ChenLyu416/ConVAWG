"""
Pydantic schemas for VAWG scenario generation output.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class RelationshipType(str, Enum):
    """Types of relationships between perpetrator and victim."""
    intimate_partner = "intimate_partner"
    ex_partner = "ex_partner"
    family = "family"
    acquaintance = "acquaintance"
    stranger = "stranger"
    professional = "professional"
    other = "other"


class ViolenceType(str, Enum):
    """Types of violence against women and girls (based on CPS VAWG strategy)."""
    physical = "physical"
    sexual = "sexual"
    rape = "rape"
    sexual_assault = "sexual_assault"
    assault_by_penetration = "assault_by_penetration"
    child_sexual_abuse = "child_sexual_abuse"
    psychological = "psychological"
    emotional = "emotional"
    economic = "economic"
    financial = "financial"
    coercive_control = "coercive_control"
    stalking = "stalking"
    harassment = "harassment"
    cyberstalking = "cyberstalking"
    digital = "digital"
    child_abuse_non_sexual = "child_abuse_non_sexual"
    neglect = "neglect"
    honor_based = "honor_based"
    forced_marriage = "forced_marriage"
    fgm = "fgm"
    trafficking = "trafficking"
    exploitation = "exploitation"
    prostitution_related = "prostitution_related"
    obscene_publications = "obscene_publications"
    other = "other"


class ActorProfile(BaseModel):
    """Base profile for any actor in the scenario."""
    name: Optional[str] = Field(None, description="Name or identifier")
    age: Optional[int] = Field(None, description="Age in years", ge=0, le=120)
    gender: Optional[str] = Field(None, description="Gender identity")
    occupation: Optional[str] = Field(None, description="Occupation or employment status")
    ethnicity: Optional[str] = Field(None, description="Ethnicity or nationality")
    demographics: Dict[str, Any] = Field(default_factory=dict, description="Additional demographic information")


class PerpetratorProfile(ActorProfile):
    """Profile of the perpetrator."""
    relationship_to_victim: Optional[RelationshipType] = Field(
        None, 
        description="Type of relationship with victim"
    )
    motivations: List[str] = Field(
        default_factory=list,
        description="Motivations or risk factors (e.g., 'jealousy', 'control', 'financial gain')"
    )
    history: List[str] = Field(
        default_factory=list,
        description="History of violence, criminal record, or relevant background"
    )
    characteristics: List[str] = Field(
        default_factory=list,
        description="Personality traits or behavioral characteristics"
    )


class ThirdPartyActor(ActorProfile):
    """Profile of a third-party actor (witness, helper, bystander, etc.)."""
    role: str = Field(..., description="Role in the scenario (e.g., 'witness', 'family_member', 'service_provider')")
    relationship_to_victim: Optional[str] = Field(None, description="Relationship to victim")
    actions: List[str] = Field(
        default_factory=list,
        description="Actions taken or not taken in the scenario"
    )


class VictimProfile(ActorProfile):
    """Profile of the victim."""
    vulnerabilities: List[str] = Field(
        default_factory=list,
        description="Vulnerabilities or risk factors (e.g., 'financial dependence', 'isolation', 'disability')"
    )
    support_seeking: List[str] = Field(
        default_factory=list,
        description="History of seeking support or reporting (e.g., 'contacted police', 'visited GP')"
    )


class VAWGScenario(BaseModel):
    """Complete VAWG scenario generated from persona and retrieved cases."""
    persona_id: str = Field(..., description="ID of the input persona")
    persona_text: str = Field(..., description="Original persona description text")
    
    victim_profile: VictimProfile = Field(..., description="Generated victim profile")
    perpetrator_profile: PerpetratorProfile = Field(..., description="Generated perpetrator profile")
    
    crime_types: List[ViolenceType] = Field(
        ..., 
        description="Types of violence in the scenario",
        min_items=1
    )
    
    time_duration: Optional[str] = Field(
        None,
        description="Duration of the abusive relationship or violence period (e.g., '6 months', '2 years', '3 weeks')"
    )
    
    third_party_actors: List[ThirdPartyActor] = Field(
        default_factory=list,
        description="Third-party actors involved (witnesses, helpers, bystanders, etc.)"
    )
    
    scenario_description: str = Field(
        ...,
        description="Narrative description of the VAWG scenario",
        min_length=50
    )
    
    retrieved_case_ids: List[str] = Field(
        default_factory=list,
        description="IDs of KB cases used as few-shot examples"
    )
    
    generation_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about generation (model, timestamp, etc.)"
    )


class RetrievedCase(BaseModel):
    """Represents a retrieved case from the knowledge base."""
    case_id: str = Field(..., description="Unique case identifier")
    victim_summary: str = Field(..., description="Summary of victim profile")
    perpetrator_summary: str = Field(..., description="Summary of perpetrator profile")
    violence_types: List[str] = Field(default_factory=list, description="Types of violence in this case")
    context: str = Field(..., description="Case context and description")
    similarity_score: float = Field(..., description="Similarity score to query persona", ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata including full case data, actors, relationships, events"
    )
