"""
Main script to generate VAWG scenarios for all personas in VAWG_sampled_200_personas.csv
"""
import argparse
import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "RAG_kb_construct"))

from retrieval import KBCaseRetriever
from vawg_scenario_generator import VAWGScenarioGenerator


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vawg_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VAWGPipeline:
    """Main pipeline for processing personas and generating VAWG scenarios."""
    
    def __init__(
        self,
        personas_csv: Path,
        persona_index_dir: Path,
        kb_homicide_dir: Path,
        model_config: Path,
        output_dir: Path,
        k_cases: int = 10,
        zero_shot: bool = False
    ):
        """
        Initialize pipeline with all required paths.
        
        Args:
            personas_csv: Path to VAWG_sampled_200_personas.csv
            persona_index_dir: Path to persona_index/ with FAISS index
            kb_homicide_dir: Path to KB_homicide/ with case JSON files
            model_config: Path to a backend config YAML (e.g. gpt52_openai.yaml)
            output_dir: Directory to save generated scenarios
            k_cases: Number of similar cases to retrieve per persona
            zero_shot: If True, skip RAG retrieval and generate from persona only
        """
        self.personas_csv = Path(personas_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.k_cases = k_cases
        self.zero_shot = zero_shot
        
        # Initialize retriever (skip if zero_shot mode)
        if not self.zero_shot:
            logger.info("Initializing case retriever...")
            self.retriever = KBCaseRetriever(
                persona_index_dir=persona_index_dir,
                kb_homicide_dir=kb_homicide_dir
            )
        else:
            logger.info("Zero-shot mode enabled - skipping retriever initialization")
            self.retriever = None
        
        # Initialize generator
        logger.info("Initializing scenario generator...")
        self.generator = VAWGScenarioGenerator(
            config_path=model_config,
            output_dir=output_dir
        )
        
        # Stats tracking
        self.stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "errors": []
        }
    
    def load_personas(self) -> List[Dict[str, Any]]:
        """Load personas from CSV file."""
        logger.info(f"Loading personas from {self.personas_csv}")
        
        personas = []
        with open(self.personas_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                personas.append({
                    "persona_id": row["persona_id"],
                    "text": row["text"],
                    "score": float(row["score"])
                })
        
        logger.info(f"Loaded {len(personas)} personas")
        return personas
    
    def process_persona(self, persona: Dict[str, Any]) -> bool:
        """
        Process a single persona: retrieve -> generate -> save.
        
        Args:
            persona: Dict with persona_id, text, score
        
        Returns:
            True if successful, False otherwise
        """
        persona_id = persona["persona_id"]
        persona_text = persona["text"]
        
        logger.info(f"Processing persona {persona_id}: {persona_text[:100]}...")
        
        try:
            # Step 1: Retrieve similar cases (skip if zero_shot)
            if self.zero_shot:
                logger.info("Zero-shot mode - skipping retrieval")
                retrieved_cases = []
            else:
                logger.info(f"Retrieving {self.k_cases} similar cases...")
                retrieved_cases = self.retriever.retrieve_similar_cases(
                    persona_text=persona_text,
                    k=self.k_cases,
                    role_filter="victim"
                )
                
                if not retrieved_cases:
                    logger.warning(f"No cases retrieved for {persona_id}")
                    return False
                
                logger.info(f"Retrieved {len(retrieved_cases)} cases")
            
            # Step 2: Generate scenario
            logger.info("Generating VAWG scenario...")
            scenario = self.generator.generate(
                persona_id=persona_id,
                persona_text=persona_text,
                retrieved_cases=retrieved_cases
            )
            
            if scenario is None:
                logger.error(f"Failed to generate scenario for {persona_id}")
                return False
            
            # Step 3: Save to JSON
            output_file = self.output_dir / f"{persona_id}.json"
            self._save_scenario(scenario, output_file)
            
            logger.info(f"Successfully processed {persona_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {persona_id}: {e}", exc_info=True)
            self.stats["errors"].append({
                "persona_id": persona_id,
                "error": str(e)
            })
            return False
    
    def _save_scenario(self, scenario: Any, output_file: Path) -> None:
        """Save VAWGScenario to JSON file."""
        try:
            # Convert Pydantic model to dict
            scenario_dict = scenario.model_dump()
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(scenario_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved scenario to {output_file}")
        except Exception as e:
            logger.error(f"Error saving scenario to {output_file}: {e}")
    def parse_args() -> argparse.Namespace:
        """Parse CLI arguments for pipeline configuration."""

        parser = argparse.ArgumentParser(
            description="Generate VAWG scenarios with selectable LLM backends"
        )
        parser.add_argument(
            "--model",
            choices=["gpt52", "deepseek_v4_flash", "gemini_flash_lite", "qwen35_35b", "llama3_8b"],
            default="gpt52",
            help="Preset model to use (default: gpt52)"
        )
        parser.add_argument(
            "--model-config",
            type=str,
            help="Path to a custom LLM config YAML. Overrides --model when provided."
        )
        parser.add_argument(
            "--k-cases",
            type=int,
            default=5,
            help="Number of similar KB cases to retrieve per persona (default: 5)"
        )
        return parser.parse_args()


        raise
    
        args = parse_args()
    def run(self) -> Dict[str, Any]:
        """
        Run the full pipeline on all personas.
        
        Returns:
            Summary statistics
        """
        logger.info("=" * 80)
        logger.info("Starting VAWG Scenario Generation Pipeline")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Load personas
        personas = self.load_personas()
        
        # Check which personas are already completed
        completed_personas = set()
        if self.output_dir.exists():
            for json_file in self.output_dir.glob("*.json"):
                if json_file.name != "generation_summary.json":
                    completed_personas.add(json_file.stem)
        
        logger.info(f"Found {len(completed_personas)} already completed personas")
        logger.info(f"Total personas to process: {len(personas)}")
        logger.info(f"Remaining personas: {len(personas) - len(completed_personas)}")
        
        self.stats["total"] = len(personas)
        
        # Process each persona
        for i, persona in enumerate(personas, 1):
            # Skip if already completed
            if str(persona['persona_id']) in completed_personas:
                logger.info(f"Skipping persona {persona['persona_id']} (already completed)")
                self.stats["successful"] += 1
                continue
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Progress: {i}/{len(personas)} ({i/len(personas)*100:.1f}%)")
            logger.info(f"{'='*80}")
            
            success = self.process_persona(persona)
            
            if success:
                self.stats["successful"] += 1
            else:
                self.stats["failed"] += 1
            
            # Log progress every 10 personas
            if i % 10 == 0:
                self._log_progress(i, len(personas))
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total personas: {self.stats['total']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Success rate: {self.stats['successful']/self.stats['total']*100:.1f}%")
        logger.info(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"Avg time per persona: {duration/self.stats['total']:.1f} seconds")
        
        # Save summary
        self._save_summary(start_time, end_time, duration)
        
        return self.stats
    
    def _log_progress(self, current: int, total: int) -> None:
        """Log progress checkpoint."""
        logger.info("\n" + "-" * 80)
        logger.info(f"CHECKPOINT: {current}/{total} personas processed")
        logger.info(f"Success: {self.stats['successful']}, Failed: {self.stats['failed']}")
        logger.info(f"Success rate so far: {self.stats['successful']/current*100:.1f}%")
        logger.info("-" * 80 + "\n")
    
    def _save_summary(self, start_time: datetime, end_time: datetime, duration: float) -> None:
        """Save summary statistics to JSON."""
        summary = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "duration_minutes": duration / 60,
            "total_personas": self.stats["total"],
            "successful": self.stats["successful"],
            "failed": self.stats["failed"],
            "success_rate": self.stats["successful"] / self.stats["total"] * 100,
            "avg_time_per_persona": duration / self.stats["total"],
            "errors": self.stats["errors"]
        }
        
        summary_file = self.output_dir / "generation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved summary to {summary_file}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Generate VAWG scenarios with configurable LLM backends"
    )
    parser.add_argument(
        "--model",
        choices=["gpt52", "deepseek_v4_flash", "gemini_flash_lite", "qwen35_35b", "llama3_8b"],
        default="gpt52",
        help="Preset model alias to use (default: gpt52)"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        help="Custom YAML config path. Overrides --model when provided."
    )
    parser.add_argument(
        "--k-cases",
        type=int,
        default=5,
        help="Number of similar KB cases to retrieve per persona (default: 5)"
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Skip RAG retrieval and generate scenarios from persona only (zero-shot mode)"
    )
    parser.add_argument(
        "--personas-csv",
        type=str,
        default=None,
        help="Custom personas CSV path (default: persona_generate/VAWG_sampled_200_personas.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for generated scenarios (default: persona_generate/outputs/vawg_scenarios)"
    )
    parser.add_argument(
        "--persona-index-dir",
        type=str,
        default=None,
        help="Custom persona index directory (default: persona_generate/outputs/persona_index_full)"
    )
    parser.add_argument(
        "--kb-homicide-dir",
        type=str,
        default=None,
        help="Custom KB_homicide directory (default: RAG_kb_construct/KB_homicide)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    base_dir = Path(__file__).parent.parent
    personas_csv = (
        Path(args.personas_csv)
        if args.personas_csv
        else base_dir / "persona_generate" / "VAWG_sampled_200_personas.csv"
    )
    persona_index_dir = (
        Path(args.persona_index_dir)
        if args.persona_index_dir
        else base_dir / "persona_generate" / "outputs" / "persona_index_full"
    )
    kb_homicide_dir = (
        Path(args.kb_homicide_dir)
        if args.kb_homicide_dir
        else base_dir / "RAG_kb_construct" / "KB_homicide"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else base_dir / "persona_generate" / "outputs" / "vawg_scenarios"
    )

    preset_configs = {
        "gpt52": base_dir / "RAG_kb_construct" / "configs" / "gpt52_openai.yaml",
        "deepseek_v4_flash": base_dir / "RAG_kb_construct" / "configs" / "deepseek_v4_flash_openai.yaml",
        "gemini_flash_lite": base_dir / "RAG_kb_construct" / "configs" / "gemini_flash_lite_openai.yaml",
        "qwen35_35b": base_dir / "RAG_kb_construct" / "configs" / "qwen35_35b_vllm.yaml",
        "llama3_8b": base_dir / "RAG_kb_construct" / "configs" / "llama3_8b_vllm.yaml",
    }

    if args.model_config:
        model_config = Path(args.model_config)
    else:
        model_config = preset_configs[args.model]

    # Verify paths exist (skip index/KB verification if zero-shot)
    paths_to_verify = [
        (personas_csv, "Personas CSV"),
        (model_config, "Model Config")
    ]
    
    if not args.zero_shot:
        paths_to_verify.extend([
            (persona_index_dir, "Persona Index"),
            (kb_homicide_dir, "KB Homicide")
        ])
    
    for path, name in paths_to_verify:
        if not path.exists():
            logger.error(f"{name} not found: {path}")
            sys.exit(1)

    logger.info("All required paths verified")
    logger.info(f"Using model config: {model_config}")
    logger.info(f"Mode: {'Zero-shot (no RAG)' if args.zero_shot else f'RAG-based (k={args.k_cases})'}")

    pipeline = VAWGPipeline(
        personas_csv=personas_csv,
        persona_index_dir=persona_index_dir,
        kb_homicide_dir=kb_homicide_dir,
        model_config=model_config,
        output_dir=output_dir,
        k_cases=args.k_cases,
        zero_shot=args.zero_shot
    )

    stats = pipeline.run()

    success_rate = stats["successful"] / stats["total"] * 100
    if stats["successful"] > 0:
        logger.info(f"Pipeline completed: {stats['successful']}/{stats['total']} succeeded ({success_rate:.1f}%)")
        sys.exit(0)
    else:
        logger.warning(f"Pipeline completed with 0 successful scenarios (success rate: {success_rate:.1f}%)")
        sys.exit(1)


if __name__ == "__main__":
    main()
