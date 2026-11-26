"""
Post-processing script for generated VAWG scenarios.

Performs two main tasks:
1. Ensures consistency between scenario description and structured data (personas, crime types)
2. Adds realistic names to personas and updates scenario description accordingly
"""
import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Set
from datetime import datetime
import sys

# Add RAG_kb_construct to path for imports (same as vawg_scenario_generator.py)
sys.path.insert(0, str(Path(__file__).parent.parent / "RAG_kb_construct"))

from llm_structurer.llm_backends import HuggingFaceBackend, OpenAIBackend
from llm_structurer.config import Config
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Diverse name pools organized by ethnicity and gender
DIVERSE_NAMES = {
    "white_british": {
        "female": ["Olivia", "Amelia", "Isla", "Ava", "Mia", "Isabella", "Sophia", "Grace", "Lily", "Freya", "Ella", "Rosie", "Evie", "Florence", "Poppy", "Willow", "Elsie", "Imogen", "Phoebe", "Sophie"],
        "male": ["Oliver", "George", "Harry", "Noah", "Jack", "Leo", "Oscar", "Charlie", "Alfie", "Freddie", "Archie", "Henry", "Thomas", "Finley", "Theo", "Arthur", "Logan", "Lucas", "Jacob", "Joshua"]
    },
    "asian": {
        "female": ["Priya", "Aisha", "Fatima", "Zara", "Amina", "Sara", "Layla", "Maya", "Aaliyah", "Yasmin", "Nisha", "Riya", "Anaya", "Isha", "Meera", "Kavya", "Diya", "Aria", "Mira", "Zoya"],
        "male": ["Mohammed", "Ali", "Omar", "Yusuf", "Adam", "Ibrahim", "Hassan", "Tariq", "Imran", "Zayn", "Amir", "Rahul", "Arjun", "Rohan", "Aarav", "Aryan", "Kabir", "Vihaan", "Ishaan", "Rayan"]
    },
    "black_african": {
        "female": ["Aisha", "Amara", "Nia", "Zuri", "Asha", "Imani", "Kaya", "Lila", "Nala", "Zara", "Amina", "Safiya", "Adanna", "Chioma", "Folake", "Ngozi", "Thandiwe", "Naledi", "Ayanna", "Kemi"],
        "male": ["Jamal", "Kofi", "Kwame", "Malik", "Tariq", "Idris", "Rashid", "Adebayo", "Chukwudi", "Emeka", "Olu", "Tunde", "Kwesi", "Mandla", "Thabo", "Sipho", "Bandile", "Dumisani", "Jabulani", "Tshepo"]
    },
    "black_caribbean": {
        "female": ["Aaliyah", "Amara", "Nia", "Zara", "Layla", "Maya", "Jasmine", "Kya", "Tiana", "Alicia", "Keisha", "Tamara", "Simone", "Rochelle", "Latoya", "Shanice", "Mariah", "Chantelle", "Nadine", "Zoe"],
        "male": ["Andre", "Marcus", "Dwayne", "Tyrone", "Jamal", "Malik", "Xavier", "Isaiah", "Elijah", "Jaden", "Devon", "Jerome", "Rashawn", "Darian", "Terrell", "Kamari", "Dante", "Marquis", "Khalil", "Quinton"]
    },
    "chinese": {
        "female": ["Mei", "Ling", "Wei", "Yan", "Xiu", "Li", "Jing", "Yue", "Hui", "Qing", "Xia", "Min", "Fang", "Ning", "Yu", "Shu", "Wen", "Hong", "Ping", "Rong"],
        "male": ["Wei", "Chen", "Lei", "Jun", "Ming", "Yang", "Feng", "Jian", "Yong", "Hui", "Tao", "Peng", "Bin", "Cheng", "Gang", "Long", "Hai", "Qiang", "Bo", "Kun"]
    },
    "south_asian": {
        "female": ["Priya", "Deepa", "Anita", "Sunita", "Kavita", "Neha", "Pooja", "Sneha", "Radha", "Anjali", "Lakshmi", "Divya", "Rekha", "Geeta", "Nisha", "Meena", "Shilpa", "Rani", "Sita", "Usha"],
        "male": ["Rajesh", "Suresh", "Mahesh", "Ramesh", "Dinesh", "Anil", "Vijay", "Ajay", "Sanjay", "Manoj", "Rakesh", "Ashok", "Deepak", "Naresh", "Pankaj", "Ravi", "Naveen", "Sandeep", "Gopal", "Mohan"]
    },
    "middle_eastern": {
        "female": ["Fatima", "Layla", "Zara", "Amina", "Sara", "Noor", "Maryam", "Hana", "Rania", "Dina", "Salma", "Leila", "Yasmin", "Jamila", "Zainab", "Amal", "Soraya", "Nadia", "Rabia", "Samira"],
        "male": ["Mohammed", "Ahmed", "Ali", "Omar", "Hassan", "Ibrahim", "Yousef", "Khalid", "Rashid", "Tariq", "Samir", "Karim", "Jamal", "Fahad", "Mansour", "Walid", "Adel", "Faisal", "Nabil", "Rami"]
    },
    "eastern_european": {
        "female": ["Anastasia", "Natasha", "Elena", "Olga", "Irina", "Svetlana", "Tatiana", "Marina", "Yulia", "Katya", "Daria", "Alina", "Sofia", "Maria", "Anna", "Ekaterina", "Valentina", "Larisa", "Oksana", "Vera"],
        "male": ["Dimitri", "Alexei", "Ivan", "Mikhail", "Sergei", "Vladimir", "Nikolai", "Andrei", "Boris", "Pavel", "Viktor", "Yuri", "Maxim", "Roman", "Anton", "Konstantin", "Leonid", "Stanislav", "Grigori", "Oleg"]
    },
    "hispanic": {
        "female": ["Sofia", "Isabella", "Camila", "Valentina", "Maria", "Lucia", "Gabriela", "Daniela", "Victoria", "Martina", "Elena", "Carmen", "Rosa", "Adriana", "Beatriz", "Carolina", "Fernanda", "Alejandra", "Catalina", "Diana"],
        "male": ["Santiago", "Mateo", "Sebastian", "Diego", "Miguel", "Carlos", "Juan", "Luis", "Pedro", "Ricardo", "Rafael", "Javier", "Fernando", "Manuel", "Alejandro", "Antonio", "Eduardo", "Jorge", "Pablo", "Raul"]
    },
    "default": {
        "female": ["Anna", "Emma", "Lucy", "Hannah", "Rachel", "Laura", "Sarah", "Rebecca", "Jennifer", "Michelle", "Nicole", "Christina", "Stephanie", "Amanda", "Jessica", "Ashley", "Melissa", "Samantha", "Elizabeth", "Victoria"],
        "male": ["David", "James", "Michael", "Robert", "Christopher", "Daniel", "Matthew", "Andrew", "Mark", "Paul", "Steven", "Kevin", "Brian", "Jonathan", "Ryan", "Jason", "Justin", "Nathan", "Eric", "Aaron"]
    }
}


class ScenarioPostProcessor:
    """Post-processes generated VAWG scenarios for consistency and name assignment."""
    
    def __init__(self, model_config_path: Path, scenarios_dir: Path, output_dir: Path = None):
        """
        Initialize post-processor.
        
        Args:
            model_config_path: Path to LLM config (e.g., gpt52_openai.yaml)
            scenarios_dir: Directory containing generated scenario JSON files
            output_dir: Directory to save processed scenarios (if None, modifies in place)
        """
        self.scenarios_dir = Path(scenarios_dir)
        self.output_dir = Path(output_dir) if output_dir else self.scenarios_dir
        self.config = Config(str(model_config_path))
        
        # Initialize LLM backend based on config
        backend_type = self.config.get('llm.backend', 'huggingface')
        logger.info(f"Loading model: {self.config.model_name} (backend: {backend_type})")
        
        if backend_type == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(
                    "OpenAI backend requires OPENAI_API_KEY environment variable. "
                    "Set it with: export OPENAI_API_KEY='your-api-key'"
                )
            self.backend = OpenAIBackend(
                model_name=self.config.model_name,
                api_key=api_key,
                base_url=self.config.get('llm.base_url')
            )
        elif backend_type == 'huggingface':
            self.backend = HuggingFaceBackend(
                model_name=self.config.model_name,
                load_in_8bit=self.config.get('llm.load_in_8bit', True),
                device_map=self.config.get('llm.device_map', 'auto')
            )
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}. Use 'openai' or 'huggingface'")
        
        # Track used names to avoid repetition
        self.used_names: Set[str] = set()
        # Track available names per category (deep copy to avoid modifying global)
        import copy
        self.available_names = copy.deepcopy(DIVERSE_NAMES)
        self.stats = {
            "total": 0,
            "consistency_fixes": 0,
            "names_added": 0,
            "failed": 0
        }
    
    def load_scenario(self, filepath: Path) -> Dict[str, Any]:
        """Load scenario from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_scenario(self, filepath: Path, scenario: Dict[str, Any]) -> None:
        """Save scenario to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(scenario, f, indent=2, ensure_ascii=False)
    
    def map_ethnicity_to_category(self, ethnicity: str) -> str:
        """
        Map ethnicity string to name category.
        
        Args:
            ethnicity: Ethnicity string from profile
            
        Returns:
            Name category key for DIVERSE_NAMES
        """
        if not ethnicity or ethnicity.lower() == "unknown":
            return "default"
        
        ethnicity_lower = ethnicity.lower()
        
        # Mapping rules
        if any(term in ethnicity_lower for term in ["white", "british", "english", "scottish", "welsh", "irish", "european"]):
            if any(term in ethnicity_lower for term in ["eastern", "polish", "romanian", "bulgarian", "russian", "ukrainian"]):
                return "eastern_european"
            return "white_british"
        elif any(term in ethnicity_lower for term in ["asian", "indian", "pakistani", "bangladeshi", "sri lankan"]):
            if any(term in ethnicity_lower for term in ["chinese", "taiwanese", "hong kong"]):
                return "chinese"
            elif any(term in ethnicity_lower for term in ["middle east", "arab", "iranian", "iraqi", "syrian", "lebanese", "turkish"]):
                return "middle_eastern"
            return "south_asian"
        elif any(term in ethnicity_lower for term in ["black", "african"]):
            if any(term in ethnicity_lower for term in ["caribbean", "jamaican", "barbados"]):
                return "black_caribbean"
            return "black_african"
        elif any(term in ethnicity_lower for term in ["hispanic", "latina", "spanish", "mexican", "colombian", "brazilian"]):
            return "hispanic"
        elif any(term in ethnicity_lower for term in ["middle east", "arab", "persian"]):
            return "middle_eastern"
        else:
            return "default"
    
    def select_diverse_name(self, ethnicity: str, gender: str) -> str:
        """
        Select a diverse name from hardcoded pools based on ethnicity and gender.
        
        Args:
            ethnicity: Ethnicity string
            gender: Gender string (male/female)
            
        Returns:
            Selected name
        """
        # Map ethnicity to category
        category = self.map_ethnicity_to_category(ethnicity)
        
        # Normalize gender
        gender_key = "female" if gender and "female" in gender.lower() else "male"
        
        # Get available names for this category and gender
        if category in self.available_names and gender_key in self.available_names[category]:
            name_pool = self.available_names[category][gender_key]
        else:
            # Fallback to default
            name_pool = self.available_names["default"][gender_key]
        
        # Filter out already used names
        available = [name for name in name_pool if name not in self.used_names]
        
        # If all names used, allow reuse but try to use different category
        if not available:
            # Try other categories
            for cat in self.available_names.keys():
                if cat != category and gender_key in self.available_names[cat]:
                    potential = [n for n in self.available_names[cat][gender_key] if n not in self.used_names]
                    if potential:
                        available = potential
                        break
            
            # If still none available, just use the original pool
            if not available:
                available = name_pool
        
        # Select random name
        name = random.choice(available)
        self.used_names.add(name)
        
        return name
    
    def generate_contextual_names(self, scenario: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate contextually appropriate names from hardcoded diverse name pools.
        
        Args:
            scenario: Full scenario dict with ethnicity and demographic info
        
        Returns:
            Dict mapping role to first name (e.g., {'victim': 'Priya', 'perpetrator': 'Rahul'})
        """
        victim = scenario['victim_profile']
        perp = scenario['perpetrator_profile']
        third_parties = scenario.get('third_party_actors', [])
        
        # Generate names from hardcoded diverse pools
        names_dict = {}
        
        # Victim name
        names_dict['victim'] = self.select_diverse_name(
            ethnicity=victim.get('ethnicity', ''),
            gender=victim.get('gender', 'female')
        )
        
        # Perpetrator name
        names_dict['perpetrator'] = self.select_diverse_name(
            ethnicity=perp.get('ethnicity', ''),
            gender=perp.get('gender', 'male')
        )
        
        # Third party names
        for i, actor in enumerate(third_parties):
            names_dict[f'third_party_{i}'] = self.select_diverse_name(
                ethnicity=actor.get('ethnicity', ''),
                gender=actor.get('gender', 'unknown')
            )
        
        logger.info(f"  Generated names: Victim={names_dict['victim']}, Perpetrator={names_dict['perpetrator']}")
        
        return names_dict
    
    def check_consistency(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check consistency between scenario_description and structured data.
        
        Returns:
            Dict with 'consistent' (bool) and 'issues' (list of strings)
        """
        issues = []
        description = scenario.get('scenario_description', '')
        
        # Check if crime types mentioned in description
        crime_types = scenario.get('crime_types', [])
        crime_keywords = {
            'physical': ['hit', 'punch', 'kick', 'beat', 'assault', 'violent', 'struck', 'injured', 'physical'],
            'psychological': ['threaten', 'intimidat', 'humiliat', 'belittle', 'manipulat', 'psychological'],
            'emotional': ['emotional', 'criticiz', 'demean', 'undermin', 'shame', 'guilt', 'anxiet'],
            'coercive_control': ['control', 'isolat', 'monitor', 'restrict', 'dominat', 'coercive'],
            'stalking': ['follow', 'stalk', 'watch', 'surveil', 'track'],
            'harassment': ['harass', 'unwanted', 'persistent', 'repeated', 'calls', 'texts', 'messages'],
            'economic': ['money', 'financial', 'account', 'income', 'employ', 'spending'],
            'sexual': ['sexual', 'rape', 'assault', 'consent']
        }
        
        for crime_type in crime_types:
            if crime_type in crime_keywords:
                keywords = crime_keywords[crime_type]
                if not any(kw in description.lower() for kw in keywords):
                    issues.append(f"Crime type '{crime_type}' not reflected in description")
        
        # Check if victim characteristics match
        victim = scenario.get('victim_profile', {})
        if victim.get('age') and str(victim['age']) not in description:
            if not any(str(age) in description for age in range(victim['age']-5, victim['age']+6)):
                issues.append("Victim age not mentioned in description")
        
        # Check if relationship type mentioned
        perp = scenario.get('perpetrator_profile', {})
        relationship = perp.get('relationship_to_victim', '')
        relationship_keywords = {
            'intimate_partner': ['partner', 'boyfriend', 'girlfriend', 'husband', 'wife'],
            'ex_partner': ['ex-', 'former', 'previous'],
            'family': ['father', 'mother', 'brother', 'sister', 'family']
        }
        
        if relationship in relationship_keywords:
            keywords = relationship_keywords[relationship]
            if not any(kw in description.lower() for kw in keywords):
                issues.append(f"Relationship type '{relationship}' not clear in description")
        
        return {
            'consistent': len(issues) == 0,
            'issues': issues
        }
    
    def fix_consistency(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to fix consistency issues between description and structured data.
        
        Args:
            scenario: Scenario dict with potential inconsistencies
        
        Returns:
            Updated scenario with consistent description
        """
        prompt = f"""You are an expert editor ensuring consistency in VAWG (Violence Against Women and Girls) scenario data.

The following scenario has a description that may not fully reflect the structured data. Your task is to rewrite ONLY the scenario_description to make it consistent with all the structured information provided.

STRUCTURED DATA:
- Victim: Age {scenario['victim_profile'].get('age', 'unknown')}, Gender: {scenario['victim_profile'].get('gender', 'unknown')}, Occupation: {scenario['victim_profile'].get('occupation', 'unknown')}
- Vulnerabilities: {', '.join(scenario['victim_profile'].get('vulnerabilities', []))}
- Perpetrator: Age {scenario['perpetrator_profile'].get('age', 'unknown')}, Gender: {scenario['perpetrator_profile'].get('gender', 'unknown')}, Relationship: {scenario['perpetrator_profile'].get('relationship_to_victim', 'unknown')}
- Crime Types: {', '.join(scenario.get('crime_types', []))}
- Third Party Actors: {len(scenario.get('third_party_actors', []))} actors involved

CURRENT DESCRIPTION:
{scenario['scenario_description']}

INSTRUCTIONS:
1. Rewrite the description to clearly reflect ALL crime types listed
2. Ensure victim and perpetrator demographics are accurate
3. Include the relationship type between victim and perpetrator
4. Mention key vulnerabilities where relevant
5. Keep the description 150-300 words
6. Maintain a realistic, non-sensationalized tone
7. Output ONLY the new description text, nothing else

NEW DESCRIPTION:"""
        
        try:
            new_description = self.backend.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            ).strip()
            
            # Clean up any markdown or extra formatting
            new_description = new_description.replace('**', '').replace('*', '')
            new_description = new_description.split('\n')[0] if '\n\n' in new_description else new_description
            
            scenario['scenario_description'] = new_description
            return scenario
            
        except Exception as e:
            logger.error(f"Error fixing consistency: {e}")
            return scenario
    
    def extract_names_from_description(self, description: str) -> List[str]:
        """Extract proper names from scenario description."""
        # Simple heuristic: capitalized words that aren't sentence starts
        import re
        words = description.split()
        names = []
        for i, word in enumerate(words):
            # Skip first word and words after punctuation
            if i > 0 and words[i-1][-1] not in '.!?':
                # Check if capitalized and not common words
                if word[0].isupper() and word.lower() not in [
                    'the', 'she', 'he', 'her', 'his', 'they', 'their', 'a', 'an'
                ]:
                    clean_word = re.sub(r'[^\w\s]', '', word)
                    if clean_word and len(clean_word) > 2:
                        names.append(clean_word)
        return list(set(names))
    
    def add_names_and_rewrite(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add names to personas if missing and rewrite description with names.
        
        Args:
            scenario: Scenario dict potentially missing names
        
        Returns:
            Updated scenario with names added and description rewritten
        """
        names_added = False
        victim = scenario['victim_profile']
        perp = scenario['perpetrator_profile']
        
        # Check if any names are missing
        needs_names = (
            not victim.get('name') or 
            not perp.get('name') or
            any(not actor.get('name') for actor in scenario.get('third_party_actors', []))
        )
        
        if needs_names:
            # Generate all names contextually at once
            names_dict = self.generate_contextual_names(scenario)
            
            # Assign names
            if not victim.get('name'):
                victim['name'] = names_dict.get('victim', 'Anna')
                names_added = True
            
            if not perp.get('name'):
                perp['name'] = names_dict.get('perpetrator', 'David')
                names_added = True
            
            # Assign third party names
            for i, actor in enumerate(scenario.get('third_party_actors', [])):
                if not actor.get('name'):
                    actor['name'] = names_dict.get(f'third_party_{i}', f'Person{i+1}')
                    names_added = True
            
            # Rewrite description with new names
            if names_added:
                scenario = self.rewrite_with_names(
                    scenario, 
                    victim['name'], 
                    perp['name']
                )
        
        return scenario, names_added
    
    def rewrite_with_names(
        self,
        scenario: Dict[str, Any],
        victim_name: str,
        perp_name: str
    ) -> Dict[str, Any]:
        """
        Rewrite scenario description using assigned names.
        
        Args:
            scenario: Scenario dict
            victim_name: Name assigned to victim
            perp_name: Name assigned to perpetrator
        
        Returns:
            Scenario with rewritten description
        """
        victim = scenario['victim_profile']
        perp = scenario['perpetrator_profile']
        
        # Build third party actors description
        third_party_desc = ""
        if scenario.get('third_party_actors'):
            actors_list = []
            for actor in scenario['third_party_actors']:
                actor_name = actor.get('name', 'a third party')
                actors_list.append(f"{actor_name} ({actor['role']})")
            third_party_desc = f"\nThird parties involved: {', '.join(actors_list)}"
        
        prompt = f"""You are a professional writer creating a realistic narrative for a VAWG (Violence Against Women and Girls) scenario.

PERSONAS:
Victim: {victim_name} (first name only)
- Age: {victim.get('age', 'unknown')}, Gender: {victim.get('gender', 'unknown')}
- Occupation: {victim.get('occupation', 'unknown')}
- Ethnicity: {victim.get('ethnicity', 'unknown')}
- Vulnerabilities: {', '.join(victim.get('vulnerabilities', []))}

Perpetrator: {perp_name} (first name only)
- Age: {perp.get('age', 'unknown')}, Gender: {perp.get('gender', 'unknown')}
- Occupation: {perp.get('occupation', 'unknown')}
- Ethnicity: {perp.get('ethnicity', 'unknown')}
- Relationship to victim: {perp.get('relationship_to_victim', 'unknown')}
- Characteristics: {', '.join(perp.get('characteristics', []))}
{third_party_desc}

CRIME TYPES: {', '.join(scenario.get('crime_types', []))}

ORIGINAL DESCRIPTION (for reference):
{scenario['scenario_description']}

INSTRUCTIONS:
Write a coherent 150-300 word narrative that:
1. Uses the names provided ({victim_name} and {perp_name})
2. Reflects all the structured data accurately
3. Describes the relationship and dynamics between victim and perpetrator
4. Shows how the crime types manifested in their relationship
5. Includes relevant vulnerabilities and their impact
6. Mentions third parties if present and their roles
7. Maintains a realistic, evidence-based tone (not sensationalized)
8. Follows typical patterns seen in Domestic Homicide Reviews

Output ONLY the narrative text, nothing else.

NARRATIVE:"""
        
        try:
            new_description = self.backend.generate(
                prompt=prompt,
                max_tokens=600,
                temperature=0.4
            ).strip()
            
            # Clean up formatting
            new_description = new_description.replace('**', '').replace('*', '')
            
            scenario['scenario_description'] = new_description
            return scenario
            
        except Exception as e:
            logger.error(f"Error rewriting with names: {e}")
            return scenario
    
    def process_scenario(self, filepath: Path) -> bool:
        """
        Process a single scenario file.
        
        Args:
            filepath: Path to scenario JSON file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing {filepath.name}...")
            
            # Load scenario
            scenario = self.load_scenario(filepath)
            modified = False
            
            # Step 1: Check consistency
            consistency_check = self.check_consistency(scenario)
            if not consistency_check['consistent']:
                logger.info(f"  Fixing {len(consistency_check['issues'])} consistency issues")
                scenario = self.fix_consistency(scenario)
                modified = True
                self.stats['consistency_fixes'] += 1
            
            # Step 2: Add names if missing and rewrite
            scenario, names_added = self.add_names_and_rewrite(scenario)
            if names_added:
                logger.info(f"  Added names and rewrote description")
                modified = True
                self.stats['names_added'] += 1
            
            # Save if modified
            if modified:
                # Determine output path
                output_path = self.output_dir / filepath.name
                
                # Create backup only if modifying in place
                if self.output_dir == self.scenarios_dir:
                    backup_path = filepath.with_suffix('.json.backup')
                    if not backup_path.exists():
                        import shutil
                        shutil.copy2(filepath, backup_path)
                
                # Save updated scenario
                self.save_scenario(output_path, scenario)
                logger.info(f"  ✓ Saved updated scenario to {output_path.name}")
            else:
                # Copy unchanged file if using separate output dir
                if self.output_dir != self.scenarios_dir:
                    import shutil
                    shutil.copy2(filepath, self.output_dir / filepath.name)
                logger.info(f"  ✓ No changes needed")
            
            self.stats['total'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error processing {filepath.name}: {e}")
            self.stats['failed'] += 1
            return False
    
    def process_all(self) -> Dict[str, Any]:
        """
        Process all scenario files in the directory.
        
        Returns:
            Statistics dictionary
        """
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        scenario_files = sorted(self.scenarios_dir.glob("*.json"))
        scenario_files = [f for f in scenario_files if f.name != "generation_summary.json"]
        
        logger.info(f"Found {len(scenario_files)} scenarios to process")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 80)
        
        for i, filepath in enumerate(scenario_files, 1):
            logger.info(f"\n[{i}/{len(scenario_files)}] {filepath.name}")
            self.process_scenario(filepath)
            
            # Log progress every 10 files
            if i % 10 == 0:
                logger.info(f"\nProgress: {i}/{len(scenario_files)} completed")
                logger.info(f"Consistency fixes: {self.stats['consistency_fixes']}")
                logger.info(f"Names added: {self.stats['names_added']}")
                logger.info(f"Failed: {self.stats['failed']}")
        
        logger.info("\n" + "=" * 80)
        logger.info("POST-PROCESSING COMPLETE")
        logger.info(f"Total scenarios: {self.stats['total']}")
        logger.info(f"Consistency fixes: {self.stats['consistency_fixes']}")
        logger.info(f"Names added: {self.stats['names_added']}")
        logger.info(f"Failed: {self.stats['failed']}")
        
        return self.stats


def main():
    """Main entry point."""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Post-process VAWG scenarios with consistency fixes and name assignment')
    parser.add_argument(
        '--model-config',
        type=str,
        default='gpt52_openai.yaml',
        help='Model config file name (default: gpt52_openai.yaml)'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input directory containing scenario JSON files to process'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for processed scenarios'
    )
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent
    model_config = base_dir / "RAG_kb_construct" / "configs" / args.model_config
    scenarios_dir = Path(args.input) if args.input else base_dir / "persona_generate" / "outputs" / "vawg_scenarios"
    output_dir = Path(args.output) if args.output else base_dir / "persona_generate" / "outputs" / "vawg_scenarios_processed"
    
    # Verify paths
    if not model_config.exists():
        logger.error(f"Model config not found: {model_config}")
        sys.exit(1)
    
    if not scenarios_dir.exists():
        logger.error(f"Scenarios directory not found: {scenarios_dir}")
        sys.exit(1)
    
    logger.info(f"Input: {scenarios_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Initialize processor
    processor = ScenarioPostProcessor(
        model_config_path=model_config,
        scenarios_dir=scenarios_dir,
        output_dir=output_dir
    )
    
    # Process all scenarios
    stats = processor.process_all()
    
    # Save processing report
    report = {
        "timestamp": datetime.now().isoformat(),
        "input_directory": str(scenarios_dir),
        "output_directory": str(output_dir),
        "statistics": stats
    }
    
    report_file = output_dir / "postprocessing_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nReport saved to {report_file}")


if __name__ == "__main__":
    main()
