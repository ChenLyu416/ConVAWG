"""
Prompt templates for few-shot VAWG scenario generation.
"""
from typing import List, Dict
from pathlib import Path
from schemas.vawg_scenario import RetrievedCase


class FewShotPromptBuilder:
    """Builds few-shot prompts for LLM generation using retrieved KB cases."""

    # Crime definitions mapping
    CRIME_DEFINITIONS = {
        "physical": "Physical assault, violence, bodily harm. Part of domestic abuse patterns.",
        "psychological": "Psychological abuse, intimidation, threats that cause fear or distress. May include isolation, humiliation, or threats.",
        "emotional": "Emotional abuse, humiliation, degradation that damages self-worth and mental wellbeing.",
        "economic": "Economic abuse - controlling financial resources, preventing access to money, exploiting resources for personal gain.",
        "financial": "Financial exploitation, control of property, goods, or services that affects victim's economic independence.",
        "sexual": "General sexual offences involving non-consensual sexual acts.",
        "rape": "Non-consensual penetration with penis (vagina, mouth, or anus). Sexual Offences Act 2003.",
        "sexual_assault": "Intentional, non-consensual sexual touching. Penetration is not required; the core elements are absence of consent and deliberate nature of the act.",
        "assault_by_penetration": "Intentional penetration of vagina or anus with any part of the body or an object without consent (not penis).",
        "child_sexual_abuse": "Sexual abuse of persons under 18. Includes physical acts (rape, oral sex, touching), non-contact acts (viewing images), grooming, and exploitation.",
        "coercive_control": "Pattern of controlling or coercive behaviour (Section 76, Serious Crime Act 2015). Includes isolating from support, depriving basic needs, monitoring activity, controlling daily behaviour. Often builds gradually.",
        "stalking": "Pattern of Fixated, Obsessive, Unwanted, and Repeated (FOUR) intrusive behaviour. Includes following, unwanted contact, surveillance, threats.",
        "harassment": "Repeated behaviour on at least two occasions that causes alarm or distress.",
        "cyberstalking": "Stalking via digital means - social media, email, tracking apps. FOUR pattern (Fixated, Obsessive, Unwanted, Repeated) in online spaces.",
        "digital": "Digital/online abuse including monitoring, surveillance, online harassment, use of technology to control or harm.",
        "child_abuse_non_sexual": "Non-sexual maltreatment of persons under 18. Includes violence, cruelty, causing physical or psychological harm.",
        "neglect": "Failure to provide care or prevent harm to a child. May occur in family, institution, or online contexts.",
        "honor_based": "Honour-Based Abuse (HBA) - violence, threats, intimidation, coercion to protect/defend family or community honour. Often involves family/community members.",
        "forced_marriage": "Forcing someone into marriage without consent or using pressure/abuse. Includes use of violence, threats, emotional abuse, or when person lacks capacity to consent.",
        "fgm": "Female Genital Mutilation - procedure injuring or altering female genitals for no medical reason. WHO Types 1-4 including removal, cutting, or other harmful acts.",
        "trafficking": "Human trafficking for sexual exploitation. Arranging/facilitating victim's travel by recruiting, transporting, harbouring with view to sexual exploitation.",
        "exploitation": "Sexual or labour exploitation. May involve trafficking, controlling prostitution, or other forms of coercion for personal gain.",
        "prostitution_related": "Controlling prostitution for gain (Section 54(2), Sexual Offences Act 2003). Intentionally controlling activities with expectation of gain through compulsion, coercion, or force.",
        "obscene_publications": "Distribution of obscene material (Obscene Publications Act 1959). Includes extreme pornographic images, material that depraves and corrupts. May be transmitted electronically.",
        "other": "Other forms of VAWG not categorized above."
    }

    # Stage 1: Generate profiles and crime types
    STAGE1_SYSTEM_PROMPT = """You are an expert in Violence Against Women and Girls (VAWG) research and risk assessment.

Your role is to generate realistic, evidence-informed VAWG scenario profiles using:
- A victim persona (synthetic)
- Examples from similar real cases (from DHRs or comparable reviews)

You will generate:
1. A detailed victim profile, including vulnerabilities and any history of support-seeking or agency contact.
2. A perpetrator profile, including demographics, relationship to the victim, behavioural patterns, motivations, and any relevant history.
3. A list of violence types involved in the scenario.
4. The time duration of the abusive relationship or violence period.
5. Optional third-party actors (family, neighbours, friends, professionals, etc.) only when relevant.

Your outputs must remain realistic, evidence-based, and aligned with the dynamics and escalation patterns typically observed in DHR and other serious VAWG cases."""

    # Stage 2: Generate scenario description
    STAGE2_SYSTEM_PROMPT = """You are an expert in Violence Against Women and Girls (VAWG) research and risk assessment.

You will write a coherent narrative scenario description (100-300 words) based on:
- The victim and perpetrator profiles already generated
- The identified crime types and their legal definitions
- The timeline of the abuse
- Any third-party actors involved

Your narrative must:
- Reflect known patterns and risk factors from VAWG research
- Be coherent, plausible, and non-sensationalised
- Avoid graphic or gratuitous detail
- Link all elements (profiles, crime types, timeline, actors) into a coherent story
- Reflect escalation and missed opportunities commonly seen in VAWG research"""

    SYSTEM_PROMPT = """You are an expert in Violence Against Women and Girls (VAWG) research and risk assessment.

Your role is to generate realistic, evidence-informed VAWG scenario profiles using:
- A victim persona (synthetic)
- Examples from similar real cases (from DHRs or comparable reviews)

Your output must:
- Reflect known patterns and risk factors from VAWG research
- Use appropriate VAWG crime-type concepts and terminology
- Be coherent, plausible, and non-sensationalised
- Avoid graphic or gratuitous detail

You will generate:
1. A detailed victim profile, including vulnerabilities and any history of support-seeking or agency contact.
2. A perpetrator profile, including demographics, relationship to the victim, behavioural patterns, motivations, and any relevant history.
3. A list of violence types involved in the scenario.
4. Optional third-party actors (family, neighbours, friends, professionals, etc.) only when relevant.
5. A narrative scenario description of 100–300 words.

Use the following VAWG-related definitions to guide your reasonings:

Controlling behaviour:
A pattern of acts intended to make a person subordinate or dependent. It can include isolating them from support, exploiting their resources for personal gain, depriving them of the means needed for independence or escape, and regulating their everyday behaviour.

Coercive behaviour:
An act or pattern of acts of assault, threats, humiliation, intimidation, or other abuse that is used to harm, punish, or frighten the victim.

Domestic abuse:
Abusive behaviour by one person aged 16 or over towards another person aged 16 or over with whom they are personally connected (for example, intimate partners or family members). Abuse can include physical or sexual abuse, violent or threatening behaviour, controlling or coercive behaviour, economic abuse, and psychological or emotional abuse. It may consist of a single incident or an ongoing pattern. Economic abuse includes behaviour that substantially affects the victim’s ability to manage money, property, goods, or services.

Harassment:
Repeated behaviour on at least two occasions that causes another person alarm or distress. The person engaging in the behaviour knows, or ought to know, that their conduct amounts to harassment.

Stalking:
A specific form of harassment involving a pattern of fixated, obsessive, unwanted, and repeated behaviour that causes fear, serious alarm, or distress. Common examples include following someone, repeatedly contacting them without consent, watching or spying on them, or interfering with their property or online presence.

Sexual assault:
Intentional, non-consensual sexual touching. Penetration is not required; the core elements are the absence of free and informed consent and the deliberate nature of the act.

Your outputs must remain realistic, evidence-based, and aligned with the dynamics and escalation patterns typically observed in DHR and other serious VAWG cases."""

    STAGE1_INSTRUCTION_PROMPT = """Based on the victim persona and the similar real cases provided, generate the profiles and identify the crime types.

IMPORTANT REQUIREMENTS:
- You must respond with VALID JSON ONLY.
- Do NOT include any text before or after the JSON.
- Follow the schema EXACTLY as specified below.
- Use null where information is unknown or cannot be inferred.
- All arrays must contain strings only (no nested objects, except where specified).
- All field names must match the schema exactly (case-sensitive).

JSON SCHEMA:

{
  "victim_profile": {
    "age": <int or null>,
    "gender": "<string or null>",
    "occupation": "<string or null>",
    "ethnicity": "<string or null>",
    "vulnerabilities": ["<vulnerability1>", "<vulnerability2>", ...],
    "support_seeking": ["<action1>", "<action2>", ...]
  },
  "perpetrator_profile": {
    "age": <int or null>,
    "gender": "<string or null>",
    "occupation": "<string or null>",
    "ethnicity": "<string or null>",
    "relationship_to_victim": "<one of: intimate_partner, ex_partner, family, acquaintance, stranger, professional, other>",
    "motivations": ["<motivation1>", "<motivation2>", ...],
    "history": ["<history1>", "<history2>", ...],
    "characteristics": ["<trait1>", "<trait2>", ...]
  },
  "crime_types": ["<violence_type1>", "<violence_type2>", ...],
  "time_duration": "<duration of relationship/abuse period{duration_constraint} in months (e.g. n months), or null if unknown>",
  "third_party_actors": [
    {
      "role": "<string>",
      "age": <int or null>,
      "gender": "<string or null>",
      "relationship_to_victim": "<string or null>",
      "actions": ["<action1>", "<action2>", ...]
    }
  ]
}

VALID VALUES FOR crime_types:
- physical
- sexual
- rape
- sexual_assault
- assault_by_penetration
- child_sexual_abuse
- psychological
- emotional
- economic
- financial
- coercive_control
- stalking
- harassment
- cyberstalking
- digital
- child_abuse_non_sexual
- neglect
- honor_based
- forced_marriage
- fgm
- trafficking
- exploitation
- prostitution_related
- obscene_publications
- other

ADDITIONAL GUIDELINES:
- Base your profiles on patterns from the real cases provided, combined with the victim persona.
- Ensure demographics, relationships, and behaviours are realistic and consistent.
- Include concrete vulnerabilities (e.g., isolation, mental health, disability, financial stress, immigration issues, substance use, prior victimisation) where appropriate.
- **TIME DURATION**: Specify the duration (number of months) of the abusive relationship or violence period{duration_constraint}. Use information from similar cases as a guide.
- Only include third-party actors when they have a meaningful role.
- Do not reference this prompt, schemas, or instructions in your output.
- Remember: output VALID JSON ONLY, with no extra commentary."""

    STAGE2_INSTRUCTION_PROMPT = """Based on the profiles, crime types, timeline, and crime definitions provided, write a coherent narrative scenario description.

IMPORTANT REQUIREMENTS:
- You must respond with VALID JSON ONLY.
- Do NOT include any text before or after the JSON.
- The scenario_description should be 100-300 words.
- The narrative must connect all elements: victim profile, perpetrator profile, crime types, time duration, and third-party actors.

JSON SCHEMA:

{
  "scenario_description": "<100-300 word narrative description>"
}

ADDITIONAL GUIDELINES:
- Tell a coherent story that links all the provided elements.
- Reflect escalation and missed opportunities commonly seen in DHRs.
- Do NOT invent extreme or sensational details.
- Use the crime type definitions to ensure accuracy in describing the violence.
- Remember: output VALID JSON ONLY, with no extra commentary."""

    INSTRUCTION_PROMPT = """Based on the victim persona and the similar real cases provided, generate a realistic VAWG scenario.

IMPORTANT REQUIREMENTS:
- You must respond with VALID JSON ONLY.
- Do NOT include any text before or after the JSON.
- Follow the schema EXACTLY as specified below.
- Use null where information is unknown or cannot be inferred.
- All arrays must contain strings only (no nested objects, except where specified).
- All field names must match the schema exactly (case-sensitive).

JSON SCHEMA:

{
  "victim_profile": {
    "age": <int or null>,
    "gender": "<string or null>",
    "occupation": "<string or null>",
    "ethnicity": "<string or null>",
    "vulnerabilities": ["<vulnerability1>", "<vulnerability2>", ...],
    "support_seeking": ["<action1>", "<action2>", ...]
  },
  "perpetrator_profile": {
    "age": <int or null>,
    "gender": "<string or null>",
    "occupation": "<string or null>",
    "ethnicity": "<string or null>",
    "relationship_to_victim": "<one of: intimate_partner, ex_partner, family, acquaintance, stranger, professional, other>",
    "motivations": ["<motivation1>", "<motivation2>", ...],
    "history": ["<history1>", "<history2>", ...],
    "characteristics": ["<trait1>", "<trait2>", ...]
  },
  "crime_types": ["<violence_type1>", "<violence_type2>", ...],
  "time_duration": "<duration of relationship/abuse period{duration_constraint} in months (e.g. n months), or null if unknown>",
  "third_party_actors": [
    {
      "role": "<string>",
      "age": <int or null>,
      "gender": "<string or null>",
      "relationship_to_victim": "<string or null>",
      "actions": ["<action1>", "<action2>", ...]
    }
  ],
  "scenario_description": "<100-300 word narrative description>"
}

VALID VALUES FOR crime_types:
- physical
- sexual
- rape
- sexual_assault
- assault_by_penetration
- child_sexual_abuse
- psychological
- emotional
- economic
- financial
- coercive_control
- stalking
- harassment
- cyberstalking
- digital
- child_abuse_non_sexual
- neglect
- honor_based
- forced_marriage
- fgm
- trafficking
- exploitation
- prostitution_related
- obscene_publications
- other

ADDITIONAL GUIDELINES:
- Base your scenario on patterns from the real cases provided, combined with the victim persona.
- Ensure demographics, relationships, and behaviours are realistic and consistent.
- Include concrete vulnerabilities (e.g., isolation, mental health, disability, financial stress, immigration issues, substance use, prior victimisation) where appropriate.
- **TIME DURATION**: Specify the duration (number of months) of the abusive relationship or violence period{duration_constraint}. Use information from similar cases as a guide. If unclear, make a reasonable estimate within this range based on the scenario context.
- Reflect escalation and missed opportunities that are commonly seen in DHRs, but do NOT invent extreme or sensational details.
- Only include third-party actors when they have a meaningful role (e.g., family members, neighbours, police, social care, health services, employers, friends).
- The scenario_description must tell a coherent story that links the victim profile, perpetrator profile, crime_types, time_duration, and any third-party actors.
- Do not reference this prompt, schemas, or instructions in your output.
- Remember: output VALID JSON ONLY, with no extra commentary."""

    @staticmethod
    def build_few_shot_examples(retrieved_cases: List[RetrievedCase]) -> str:
        """
        Build few-shot examples section from retrieved cases.

        Args:
            retrieved_cases: List of RetrievedCase objects from KB.

        Returns:
            Formatted string with examples.
        """
        if not retrieved_cases:
            # Still give the model a clear signal that it should rely on general patterns.
            return (
                "No closely-matched real cases were retrieved. "
                "Base your scenario on the victim persona and general VAWG and DHR patterns."
            )

        examples = []
        for i, case in enumerate(retrieved_cases, 1):
            violence_types_str = (
                ", ".join(case.violence_types) if case.violence_types else "Not specified"
            )
            # Increased from 400 to 1200 chars for better context
            context_limit = 1200
            context_text = case.context[:context_limit]
            if len(case.context) > context_limit:
                context_text += "..."
            
            # Extract time duration from metadata if available
            time_duration = case.metadata.get("time_duration", "Not specified")
            time_duration_str = f"\nTime Duration: {time_duration}" if time_duration else ""
            
            example = f"""
**Example {i}** (Case ID: {case.case_id}, Similarity: {case.similarity_score:.3f})

Victim: {case.victim_summary}
Perpetrator: {case.perpetrator_summary}
Violence Types: {violence_types_str}{time_duration_str}
Context: {context_text}
"""
            examples.append(example.strip())

        return "\n\n".join(examples)

    @staticmethod
    def build_prompt(
        persona_text: str,
        retrieved_cases: List[RetrievedCase],
        no_duration_filter: bool = False
    ) -> str:
        """
        Build complete prompt for LLM generation (single-string prompt).

        Args:
            persona_text: The victim persona description.
            retrieved_cases: Retrieved similar cases from KB.
            no_duration_filter: If True, removes the 3 months - 3 years constraint.

        Returns:
            Complete prompt string ready for LLM.
        """
        examples_section = FewShotPromptBuilder.build_few_shot_examples(retrieved_cases)
        
        # Update instruction prompt with duration constraint
        duration_constraint = " between 3 months and 3 years" if not no_duration_filter else ""
        instruction_prompt = FewShotPromptBuilder.INSTRUCTION_PROMPT.replace(
            "{duration_constraint}",
            duration_constraint
        )

        prompt = f"""{FewShotPromptBuilder.SYSTEM_PROMPT}

{instruction_prompt}

---

SIMILAR REAL CASES FROM DHRs (for context, do not copy verbatim):

{examples_section}

---

YOUR TASK:

Generate a VAWG scenario for the following victim persona, following all the requirements above.

Persona:
\"\"\"{persona_text}\"\"\"

Remember:
1. Base your scenario on patterns from the similar real cases and the persona.
2. Ensure demographics and characteristics are realistic and consistent.
3. Include specific vulnerabilities and risk factors where appropriate.
4. Respond with VALID JSON ONLY following the schema.
5. Do not include any text before or after the JSON.

Your JSON response:"""

        return prompt

    @staticmethod
    def build_stage1_prompt(
        persona_text: str,
        retrieved_cases: List[RetrievedCase],
        no_duration_filter: bool = False
    ) -> str:
        """
        Build Stage 1 prompt: Generate profiles and crime types.

        Args:
            persona_text: The victim persona description.
            retrieved_cases: Retrieved similar cases from KB.
            no_duration_filter: If True, removes the 3 months - 3 years constraint.

        Returns:
            Complete prompt string for Stage 1.
        """
        examples_section = FewShotPromptBuilder.build_few_shot_examples(retrieved_cases)
        
        # Update instruction prompt with duration constraint
        duration_constraint = " between 3 months and 3 years" if not no_duration_filter else ""
        instruction_prompt = FewShotPromptBuilder.STAGE1_INSTRUCTION_PROMPT.replace(
            "{duration_constraint}",
            duration_constraint
        )

        prompt = f"""{FewShotPromptBuilder.STAGE1_SYSTEM_PROMPT}

{instruction_prompt}

---

SIMILAR REAL CASES FROM DHRs (for context, do not copy verbatim):

{examples_section}

---

YOUR TASK:

Generate victim profile, perpetrator profile, crime types, time duration, and third-party actors for the following victim persona.

Persona:
\"\"\"{persona_text}\"\"\"

Remember:
1. Base your profiles on patterns from the similar real cases and the persona.
2. Ensure demographics and characteristics are realistic and consistent.
3. Include specific vulnerabilities and risk factors where appropriate.
4. Respond with VALID JSON ONLY following the schema.
5. Do not include any text before or after the JSON.

Your JSON response:"""

        return prompt

    @staticmethod
    def build_stage2_prompt(
        persona_text: str,
        stage1_output: dict,
        crime_definitions: str,
        retrieved_cases: List[RetrievedCase]
    ) -> str:
        """
        Build Stage 2 prompt: Generate scenario description with crime definitions.

        Args:
            persona_text: The original victim persona description.
            stage1_output: The JSON output from Stage 1 (profiles, crime types, etc.).
            crime_definitions: Detailed definitions of the selected crime types.
            retrieved_cases: Retrieved similar cases from KB (for context).

        Returns:
            Complete prompt string for Stage 2.
        """
        examples_section = FewShotPromptBuilder.build_few_shot_examples(retrieved_cases)
        
        # Format stage1 output nicely
        import json
        stage1_json = json.dumps(stage1_output, indent=2)

        prompt = f"""{FewShotPromptBuilder.STAGE2_SYSTEM_PROMPT}

{FewShotPromptBuilder.STAGE2_INSTRUCTION_PROMPT}

---

ORIGINAL VICTIM PERSONA:
\"\"\"{persona_text}\"\"\"

---

GENERATED PROFILES AND CRIME TYPES (from Stage 1):

{stage1_json}

---

CRIME TYPE DEFINITIONS (for the selected crime types):

{crime_definitions}

---

SIMILAR REAL CASES FROM DHRs (for context):

{examples_section}

---

YOUR TASK:

Write a 100-300 word narrative scenario description that:
1. Connects the victim profile, perpetrator profile, crime types, time duration, and third-party actors into a coherent story.
2. Uses the crime type definitions to ensure accurate description of the violence.
3. Reflects escalation patterns and missed opportunities typical of DHR cases.
4. Remains realistic, evidence-based, and non-sensational.

Respond with VALID JSON ONLY following the schema (just the scenario_description field).

Your JSON response:"""

        return prompt

    @staticmethod
    def get_crime_definitions(crime_types: List[str]) -> str:
        """
        Get definitions for the specified crime types.

        Args:
            crime_types: List of crime type codes from Stage 1 output.

        Returns:
            Formatted string with definitions for each crime type.
        """
        if not crime_types:
            return "No crime types specified."
        
        definitions = []
        for crime_type in crime_types:
            definition = FewShotPromptBuilder.CRIME_DEFINITIONS.get(
                crime_type,
                f"Definition not available for: {crime_type}"
            )
            definitions.append(f"**{crime_type}**: {definition}")
        
        return "\n\n".join(definitions)

    @staticmethod
    def build_system_message() -> str:
        """Return system message for chat-based models."""
        return FewShotPromptBuilder.SYSTEM_PROMPT

    @staticmethod
    def build_user_message(
        persona_text: str,
        retrieved_cases: List[RetrievedCase],
    ) -> str:
        """
        Build user message for chat-based models.

        Args:
            persona_text: The victim persona description.
            retrieved_cases: Retrieved similar cases from KB.

        Returns:
            User-facing prompt content string.
        """
        examples_section = FewShotPromptBuilder.build_few_shot_examples(retrieved_cases)

        return f"""{FewShotPromptBuilder.INSTRUCTION_PROMPT}

SIMILAR REAL CASES FROM DHRs (for context, do not copy verbatim):

{examples_section}

---

YOUR TASK:

Generate a VAWG scenario for the following victim persona, following all the requirements above.

Persona:
\"\"\"{persona_text}\"\"\"

Remember:
1. Base your scenario on patterns from the similar real cases and the persona.
2. Ensure demographics and characteristics are realistic and consistent.
3. Include specific vulnerabilities and risk factors where appropriate.
4. Respond with VALID JSON ONLY following the schema.
5. Do not include any text before or after the JSON.

Your JSON response:"""
