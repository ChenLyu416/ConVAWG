"""
LLM-based VAWG scenario generator with JSON parsing and reliability.
Uses a configurable LLM backend with a 6-layer JSON reliability approach.
"""
import json
import re
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

from pydantic import ValidationError

# Use existing llm_structurer infrastructure
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "RAG_kb_construct"))

from llm_structurer.llm_backends import GGUFBackend, HuggingFaceBackend
try:
    from llm_structurer.llm_backends import VLLMBackend
except ImportError:
    VLLMBackend = None
from llm_structurer.config import Config
from schemas.vawg_scenario import VAWGScenario, VictimProfile, PerpetratorProfile
from prompt_templates import FewShotPromptBuilder


logger = logging.getLogger(__name__)


class VAWGScenarioGenerator:
    """Generates VAWG scenarios using LLM with robust JSON parsing."""
    
    def __init__(
        self,
        config_path: Path,
        output_dir: Path,
        max_retries: int = 2,
        no_duration_filter: bool = False
    ):
        """
        Initialize generator with LLM backend (GGUF, HuggingFace, OpenAI, or vLLM).
        
        Args:
            config_path: Path to model config YAML (e.g., gpt52_openai.yaml)
            output_dir: Directory to save raw LLM responses
            max_retries: Maximum retry attempts for JSON parsing
            no_duration_filter: If True, removes the 3 months - 3 years constraint on time_duration
        """
        self.config = Config(str(config_path))
        self.no_duration_filter = no_duration_filter
        
        # Initialize appropriate backend based on config
        backend_type = self.config.llm_backend.lower()
        if backend_type == "gguf":
            self.backend = GGUFBackend(
                model_name=self.config.model_name,
                gguf_file=self.config.gguf_file,
                n_gpu_layers=self.config.n_gpu_layers,
                n_ctx=self.config.n_ctx
            )
        elif backend_type == "huggingface":
            self.backend = HuggingFaceBackend(
                model_name=self.config.model_name,
                load_in_8bit=self.config.get('llm.load_in_8bit', True),
                device_map=self.config.get('llm.device_map', 'auto')
            )
        elif backend_type == "openai":
            from llm_structurer.llm_backends import OpenAIBackend

            api_key = self.config.get('llm.api_key')
            if not api_key:
                api_key = self.config.get_openai_api_key()

            self.backend = OpenAIBackend(
                model_name=self.config.model_name,
                api_key=api_key,
                base_url=self.config.get('llm.base_url')
            )
        elif backend_type == "vllm":
            if VLLMBackend is None:
                raise ImportError("vLLM backend is not available. Install vllm package or use a different backend.")
            self.backend = VLLMBackend(
                model_name=self.config.model_name,
                base_url=self.config.get('llm.base_url', None),
                api_key=self.config.get('llm.api_key', None),
            )
        else:
            raise ValueError(f"Unsupported backend: {backend_type}. Supported: gguf, huggingface, openai, vllm")
        
        self.output_dir = Path(output_dir)
        self.raw_responses_dir = self.output_dir / "raw_responses"
        self.raw_responses_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        
        logger.info(f"Initialized VAWGScenarioGenerator with config: {config_path}")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Backend: {self.config.llm_backend}")
    
    def generate(
        self,
        persona_id: str,
        persona_text: str,
        retrieved_cases: list
    ) -> Optional[VAWGScenario]:
        """
        Generate VAWG scenario for a persona using two-stage generation:
        Stage 1: Generate profiles and crime types
        Stage 2: Generate scenario description with crime definitions
        
        Args:
            persona_id: Unique identifier for the persona
            persona_text: Description of the victim persona
            retrieved_cases: List of RetrievedCase objects
        
        Returns:
            VAWGScenario object or None if generation fails
        """
        logger.info(f"Generating scenario for persona {persona_id} (two-stage approach)")
        
        # STAGE 1: Generate profiles and crime types
        stage1_output = self._generate_stage1(persona_id, persona_text, retrieved_cases)
        if stage1_output is None:
            logger.error(f"Stage 1 failed for persona {persona_id}")
            return None
        
        # STAGE 2: Generate scenario description with crime definitions
        stage2_output = self._generate_stage2(
            persona_id, persona_text, stage1_output, retrieved_cases
        )
        if stage2_output is None:
            logger.error(f"Stage 2 failed for persona {persona_id}")
            return None
        
        # Combine Stage 1 and Stage 2 outputs into final scenario
        try:
            scenario = self._create_final_scenario(
                persona_id=persona_id,
                persona_text=persona_text,
                stage1_output=stage1_output,
                stage2_output=stage2_output,
                retrieved_cases=retrieved_cases
            )
            
            if scenario is not None:
                logger.info(f"Successfully generated two-stage scenario for {persona_id}")
                return scenario
        except Exception as e:
            logger.error(f"Error creating final scenario: {e}")
            return None
        
        return None
    
    def _generate_stage1(
        self,
        persona_id: str,
        persona_text: str,
        retrieved_cases: list
    ) -> Optional[Dict[str, Any]]:
        """
        Stage 1: Generate profiles and crime types.
        
        Returns:
            Parsed JSON with profiles, crime_types, time_duration, third_party_actors
        """
        logger.info(f"Stage 1: Generating profiles and crime types for {persona_id}")
        
        # Build Stage 1 prompt
        prompt = FewShotPromptBuilder.build_stage1_prompt(
            persona_text, retrieved_cases, self.no_duration_filter
        )
        
        # Try generation with retries
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Stage 1 Attempt {attempt}/{self.max_retries}")
            
            try:
                # Generate from LLM
                raw_response = self._call_llm(prompt)
                
                # Save raw response
                self._save_raw_response(
                    f"{persona_id}_stage1", attempt, prompt, raw_response
                )
                
                # Parse to JSON
                parsed_json = self._extract_json(raw_response)
                
                if parsed_json is None:
                    logger.warning(f"Stage 1: Failed to extract JSON on attempt {attempt}")
                    continue
                
                # Validate required fields
                required_fields = ["victim_profile", "perpetrator_profile", "crime_types"]
                if all(field in parsed_json for field in required_fields):
                    logger.info("Stage 1: Successfully generated profiles and crime types")
                    return parsed_json
                else:
                    logger.warning(f"Stage 1: Missing required fields on attempt {attempt}")
                    continue
                
            except Exception as e:
                logger.error(f"Stage 1 error on attempt {attempt}: {e}")
                continue
        
        logger.error(f"Stage 1: Failed after {self.max_retries} attempts")
        return None
    
    def _generate_stage2(
        self,
        persona_id: str,
        persona_text: str,
        stage1_output: Dict[str, Any],
        retrieved_cases: list
    ) -> Optional[Dict[str, Any]]:
        """
        Stage 2: Generate scenario description with crime definitions.
        
        Returns:
            Parsed JSON with scenario_description
        """
        logger.info(f"Stage 2: Generating scenario description for {persona_id}")
        
        # Extract crime types and get definitions
        crime_types = stage1_output.get("crime_types", [])
        crime_definitions = FewShotPromptBuilder.get_crime_definitions(crime_types)
        
        # Build Stage 2 prompt
        prompt = FewShotPromptBuilder.build_stage2_prompt(
            persona_text, stage1_output, crime_definitions, retrieved_cases
        )
        
        # Try generation with retries
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Stage 2 Attempt {attempt}/{self.max_retries}")
            
            try:
                # Generate from LLM
                raw_response = self._call_llm(prompt)
                
                # Save raw response
                self._save_raw_response(
                    f"{persona_id}_stage2", attempt, prompt, raw_response
                )
                
                # Parse to JSON
                parsed_json = self._extract_json(raw_response)
                
                if parsed_json is None:
                    logger.warning(f"Stage 2: Failed to extract JSON on attempt {attempt}")
                    continue
                
                # Validate scenario_description field
                if "scenario_description" in parsed_json:
                    logger.info("Stage 2: Successfully generated scenario description")
                    return parsed_json
                else:
                    logger.warning(f"Stage 2: Missing scenario_description on attempt {attempt}")
                    continue
                
            except Exception as e:
                logger.error(f"Stage 2 error on attempt {attempt}: {e}")
                continue
        
        logger.error(f"Stage 2: Failed after {self.max_retries} attempts")
        return None
    
    def _create_final_scenario(
        self,
        persona_id: str,
        persona_text: str,
        stage1_output: Dict[str, Any],
        stage2_output: Dict[str, Any],
        retrieved_cases: list
    ) -> Optional[VAWGScenario]:
        """
        Combine Stage 1 and Stage 2 outputs into final VAWGScenario object.
        """
        try:
            # Extract victim profile
            victim_data = stage1_output.get("victim_profile", {})
            victim_profile = VictimProfile(**victim_data)
            
            # Extract perpetrator profile
            perp_data = stage1_output.get("perpetrator_profile", {})
            perpetrator_profile = PerpetratorProfile(**perp_data)
            
            # Extract crime types
            crime_types = stage1_output.get("crime_types", [])
            
            # Extract time duration (coerce to str in case model returns an int)
            time_duration = stage1_output.get("time_duration")
            if time_duration is not None and not isinstance(time_duration, str):
                time_duration = str(time_duration)
            
            # Extract third-party actors
            third_party_actors = stage1_output.get("third_party_actors", [])
            
            # Extract scenario description from Stage 2
            scenario_description = stage2_output.get("scenario_description", "")
            
            # Build case IDs list
            retrieved_case_ids = [case.case_id for case in retrieved_cases]
            
            # Create VAWGScenario
            scenario = VAWGScenario(
                persona_id=persona_id,
                persona_text=persona_text,
                victim_profile=victim_profile,
                perpetrator_profile=perpetrator_profile,
                crime_types=crime_types,
                time_duration=time_duration,
                third_party_actors=third_party_actors,
                scenario_description=scenario_description,
                retrieved_case_ids=retrieved_case_ids,
                generation_metadata={
                    "model": self.config.model_name,
                    "backend": self.config.llm_backend,
                    "timestamp": datetime.now().isoformat(),
                    "two_stage_generation": True
                }
            )
            
            return scenario
            
        except ValidationError as e:
            logger.error(f"Pydantic validation error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating final scenario: {e}")
            return None
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM backend with prompt."""
        try:
            # Use backend's generate method
            # Note: GGUFBackend may not have direct json_mode, but we use prompt engineering
            response = self.backend.generate(
                prompt=prompt,
                max_tokens=2048,  # Enough for detailed JSON
                temperature=self.config.temperature
            )
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _extract_json(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from LLM response.
        Uses balanced-brace tracking to isolate the first complete JSON object,
        so extra text or multiple objects after the closing brace are ignored.

        Layer 2 of 6-layer JSON reliability approach.
        """
        # Remove markdown code blocks if present
        cleaned = re.sub(r'```json\s*|\s*```', '', raw_response)

        # Find the first '{'
        start = cleaned.find('{')
        if start == -1:
            logger.warning("No JSON object found in response")
            return None

        # Walk forward tracking brace depth, skipping string contents
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(cleaned[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    json_str = cleaned[start:i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parse error: {e}")
                        return None

        logger.warning("No complete JSON object found in response")
        return None
    
    def _create_scenario(
        self,
        persona_id: str,
        persona_text: str,
        parsed_json: Dict[str, Any],
        retrieved_cases: list
    ) -> Optional[VAWGScenario]:
        """
        Create VAWGScenario object from parsed JSON with validation.
        
        Layer 3-4 of 6-layer approach: Pydantic validation + error handling.
        """
        try:
            # Extract victim profile
            victim_data = parsed_json.get("victim_profile", {})
            victim_profile = VictimProfile(**victim_data)
            
            # Extract perpetrator profile
            perp_data = parsed_json.get("perpetrator_profile", {})
            perpetrator_profile = PerpetratorProfile(**perp_data)
            
            # Extract crime types
            crime_types = parsed_json.get("crime_types", [])
            
            # Extract time duration (optional)
            time_duration = parsed_json.get("time_duration")
            
            # Extract third-party actors (optional)
            third_party_actors = parsed_json.get("third_party_actors", [])
            
            # Extract scenario description
            scenario_description = parsed_json.get("scenario_description", "")
            
            # Build case IDs list
            retrieved_case_ids = [case.case_id for case in retrieved_cases]
            
            # Create VAWGScenario
            scenario = VAWGScenario(
                persona_id=persona_id,
                persona_text=persona_text,
                victim_profile=victim_profile,
                perpetrator_profile=perpetrator_profile,
                crime_types=crime_types,
                time_duration=time_duration,
                third_party_actors=third_party_actors,
                scenario_description=scenario_description,
                retrieved_case_ids=retrieved_case_ids,
                generation_metadata={
                    "model": self.config.model_name,
                    "backend": self.config.llm_backend,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return scenario
            
        except ValidationError as e:
            logger.error(f"Pydantic validation error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating scenario: {e}")
            return None
    
    def _save_raw_response(
        self,
        persona_id: str,
        attempt: int,
        prompt: str,
        response: str
    ) -> None:
        """
        Save raw LLM response to file for debugging.
        
        Layer 5 of 6-layer approach: Raw response logging.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{persona_id}_attempt{attempt}_{timestamp}.txt"
        filepath = self.raw_responses_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("PROMPT\n")
                f.write("=" * 80 + "\n")
                f.write(prompt)
                f.write("\n\n")
                f.write("=" * 80 + "\n")
                f.write("RESPONSE\n")
                f.write("=" * 80 + "\n")
                f.write(response)
            
            logger.debug(f"Saved raw response to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save raw response: {e}")
