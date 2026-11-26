#!/usr/bin/env python3
"""
VAWG Dataset Generation Pipeline - Batch Processing Script

This script orchestrates the entire VAWG dataset generation pipeline across 8 stages:
1. Persona & Scenarios Generation
2. Event Graph Extraction
3. Event Graph Expansion
4. Chat Script Generation
5. Roleplay Dialogue Generation
6. CAA-Steered Toxic Rewrite (Levelled)
7. Convert Dialogues to Minimal Format (with PDF generation)
8. LLM-as-Judge Dialogue Evaluation

All outputs are organized under a timestamped folder in Generation_Results/.
Each stage checks if outputs already exist before running to enable resumption.
"""

import argparse
import sys
import yaml
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, Any, List, Optional, Tuple
import logging
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates the full VAWG dataset generation pipeline."""
    
    def __init__(self, config_path: Path, base_dir: Path):
        """
        Initialize pipeline runner.
        
        Args:
            config_path: Path to YAML configuration file
            base_dir: Base directory of the repository
        """
        self.config_path = config_path
        self.base_dir = base_dir
        self.config = self._load_config()
        
        # Create timestamped run directory — include PID so parallel pipeline
        # instances launched at the same second don't share the same run_dir
        # (shared run_dir causes .staged_input_sN symlink collisions).
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = base_dir / "Generation_Results" / f"run_{timestamp}_{os.getpid()}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy configuration file to run directory
        config_backup = self.run_dir / "pipeline_config.yaml"
        shutil.copy(config_path, config_backup)
        logger.info(f"Configuration backed up to: {config_backup}")
        
        # Setup logging to file in run directory
        log_file = self.run_dir / "pipeline_run.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info(f"Pipeline run directory: {self.run_dir}")
        logger.info(f"Configuration file: {config_path}")
        
        # Track stage completion
        self.stage_status = {
            1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False, 8: False
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    def _resolve_path(self, path_str: str) -> Path:
        """
        Resolve a path from config, handling both absolute and relative paths.
        
        If path starts with 'Generation_Results/', treat it as workspace-relative.
        Otherwise, treat it as run-directory relative.
        
        Args:
            path_str: Path string from config
            
        Returns:
            Resolved absolute Path object
        """
        if path_str.startswith('Generation_Results/'):
            # This is a workspace-relative path (from previous runs)
            return self.base_dir / path_str
        else:
            # This is relative to the current run directory
            return self.run_dir / path_str

    def _resolve_repo_or_abs_path(self, path_str: str) -> Path:
        """
        Resolve a user-provided path as absolute or repository-relative.

        Args:
            path_str: Path string from config

        Returns:
            Resolved absolute Path object
        """
        path = Path(path_str)
        if path.is_absolute():
            return path
        return self.base_dir / path_str
    
    def _stage_has_output(self, output_dir: Path) -> bool:
        """
        Check if a stage output directory exists and has content.
        
        Args:
            output_dir: Path to stage output directory
            
        Returns:
            True if directory exists and contains files, False otherwise
        """
        if not output_dir.exists():
            return False
        
        # Check for any JSON files (standard output format)
        json_files = list(output_dir.glob("*.json"))
        if json_files:
            logger.info(f"Found {len(json_files)} existing output files in {output_dir}")
            return True
        
        # Check for any subdirectories with JSON files
        for subdir in output_dir.iterdir():
            if subdir.is_dir():
                json_files = list(subdir.glob("*.json"))
                if json_files:
                    logger.info(f"Found {len(json_files)} existing output files in {subdir}")
                    return True
        
        return False
    
    def _find_latest_run_dir(self, base_dir: Path) -> Path:
        """
        Find the most recent run_* subdirectory in a base directory.
        Some stages (like Stage 4) create timestamped run_* subdirectories.
        
        Args:
            base_dir: Base directory containing run_* subdirectories
            
        Returns:
            Path to most recent run directory, or base_dir if none found
        """
        if not base_dir.exists():
            return base_dir
        
        # Look for run_* subdirectories
        run_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
        
        if not run_dirs:
            return base_dir
        
        # Sort by name (timestamp format ensures newest is last)
        run_dirs.sort()
        latest = run_dirs[-1]
        logger.info(f"Using latest run directory: {latest}")
        return latest
    
    def _run_command(self, cmd: list, stage_name: str, cwd: Path = None, env: dict = None) -> bool:
        """
        Execute a shell command and log output.
        
        Args:
            cmd: Command as list of strings
            stage_name: Name of the stage (for logging)
            cwd: Working directory for command execution
            env: Environment variables (defaults to os.environ if None)
            
        Returns:
            True if command succeeded, False otherwise
        """
        logger.info(f"Executing: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=cwd or self.base_dir,
                env=env
            )
            
            # Stream output in real-time
            for line in process.stdout:
                logger.info(f"[{stage_name}] {line.rstrip()}")
            
            process.wait()
            
            if process.returncode == 0:
                logger.info(f"{stage_name} completed successfully")
                return True
            else:
                logger.error(f"{stage_name} failed with return code {process.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing {stage_name}: {e}")
            return False
    
    # ---------------------------------------------------------------------------
    # Residual-item detection helpers
    # ---------------------------------------------------------------------------

    def _get_residual_input_files(
        self,
        input_dir: Path,
        output_dir: Path,
        input_glob: str = "*.json",
        output_name_fn: Optional[Callable[[str], str]] = None,
        exclude_filenames: Optional[set] = None,
    ) -> List[Path]:
        """
        Return input files that have no corresponding output file.

        For each file matching ``input_glob`` in ``input_dir``, the expected
        output filename is ``output_dir / output_name_fn(input_file.name)``.
        Files listed in ``exclude_filenames`` are ignored (e.g. report JSONs).
        ``output_name_fn`` defaults to identity (same filename as input).
        """
        if not input_dir.exists():
            return []
        out_fn = output_name_fn or (lambda n: n)
        exclude = exclude_filenames or set()
        residuals: List[Path] = []
        for in_file in sorted(input_dir.glob(input_glob)):
            if in_file.name in exclude:
                continue
            if not (output_dir / out_fn(in_file.name)).exists():
                residuals.append(in_file)
        return residuals

    def _create_staged_input(self, residual_files: List[Path], stage_tag: str) -> Path:
        """
        Create (or refresh) a staging directory under ``run_dir`` containing
        symlinks to only the given residual input files.

        Passing this directory to a sub-script means it processes only the
        files that have not yet been handled, without modifying the script.
        """
        staged = self.run_dir / f".staged_input_{stage_tag}"
        staged.mkdir(parents=True, exist_ok=True)
        for f in residual_files:
            link = staged / f.name
            if not link.exists():
                link.symlink_to(f.resolve())
        return staged

    # ---------------------------------------------------------------------------

    def run_stage_1_persona_generation(self) -> bool:
        """
        Stage 1: VAWG Persona & Scenarios Generation
        
        Returns:
            True if stage completed successfully, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STAGE 1: VAWG Persona & Scenarios Generation")
        logger.info("=" * 80)
        
        stage_config = self.config.get('stage_1_persona_generation', {})
        output_dir = self._resolve_path(stage_config.get('output_dir', 'persona_n_scenarios'))
        
        # Check if stage is enabled
        if not self.config.get('global', {}).get('enable_stage_1', True):
            logger.info("Stage 1 is disabled in configuration. Skipping.")
            # Check if output exists from a previous run
            if self._stage_has_output(output_dir):
                logger.info("Stage 1 output found from previous run. Marking as completed.")
                self.stage_status[1] = True
            else:
                logger.warning("Stage 1 is disabled but no output exists. Subsequent stages may fail.")
                self.stage_status[1] = True  # Mark as completed to allow pipeline to continue
            return True
        
        # Check if output already exists — with residual detection.
        # A partial Stage 1 run (e.g. < 10/10 personas succeeded) still satisfies
        # _stage_has_output, so we also compare against the expected CSV row count.
        if self._stage_has_output(output_dir):
            expected_count = None
            personas_csv_path = stage_config.get('personas_csv')
            if personas_csv_path:
                try:
                    import csv as _csv
                    csv_full = self._resolve_repo_or_abs_path(personas_csv_path)
                    with open(csv_full, newline='') as _f:
                        expected_count = sum(1 for _ in _csv.DictReader(_f))
                except Exception as _e:
                    logger.warning(f"Could not count personas CSV rows: {_e}")

            actual_count = len([
                f for f in output_dir.glob("*.json")
                if f.name != "generation_summary.json"
            ])

            if expected_count is not None and actual_count < expected_count:
                logger.info(
                    f"Stage 1: {actual_count}/{expected_count} persona(s) complete "
                    f"— resuming for {expected_count - actual_count} remaining."
                )
                # Fall through to re-invoke the generation script; it will skip
                # already-generated persona files internally.
            else:
                logger.info("Stage 1 output already exists. Skipping.")
                self.stage_status[1] = True
                return True

        # Build command
        script_path = self.base_dir / "persona_generate" / "run_vawg_generation.py"
        cmd = [sys.executable, str(script_path)]

        # Optional custom personas CSV for Stage 1
        personas_csv = stage_config.get('personas_csv')
        if personas_csv:
            cmd.extend(["--personas-csv", str(self._resolve_repo_or_abs_path(personas_csv))])

        # Pass stage output directory directly to generation script
        cmd.extend(["--output-dir", str(output_dir)])

        # Optional custom retrieval/index paths
        persona_index_dir = stage_config.get('persona_index_dir')
        if persona_index_dir:
            cmd.extend(["--persona-index-dir", str(self._resolve_repo_or_abs_path(persona_index_dir))])

        kb_homicide_dir = stage_config.get('kb_homicide_dir')
        if kb_homicide_dir:
            cmd.extend(["--kb-homicide-dir", str(self._resolve_repo_or_abs_path(kb_homicide_dir))])
        
        # Add model configuration
        # NOTE: stage 1 runs with cwd=persona_generate/, so model_config paths must be absolute.
        if stage_config.get('model_config'):
            resolved_mc = str(self._resolve_repo_or_abs_path(stage_config['model_config']))
            cmd.extend(["--model-config", resolved_mc])
        elif stage_config.get('model'):
            cmd.extend(["--model", stage_config['model']])

        # Add k-cases parameter
        k_retrieval = stage_config.get('k_retrieval', 5)
        cmd.extend(["--k-cases", str(k_retrieval)])
        
        # Add zero-shot flag
        if stage_config.get('zero_shot', False):
            cmd.append("--zero-shot")
        
        # Execute command
        # Inject API key and GPU assignment into subprocess environment if provided in config
        import os
        env = os.environ.copy()
        if stage_config.get('api_key'):
            env['OPENAI_API_KEY'] = stage_config['api_key']
        if stage_config.get('gpus'):
            env['CUDA_VISIBLE_DEVICES'] = str(stage_config['gpus'])

        success = self._run_command(cmd, "Stage 1", cwd=self.base_dir / "persona_generate", env=env)
        
        if success:
            # Backward-compatible fallback copy from legacy default output
            default_output = self.base_dir / "persona_generate" / "outputs" / "vawg_scenarios"
            if default_output.exists() and default_output != output_dir and not self._stage_has_output(output_dir):
                # Copy files to our run directory
                output_dir.mkdir(parents=True, exist_ok=True)
                for file in default_output.glob("*.json"):
                    shutil.copy(file, output_dir / file.name)
                logger.info(f"Copied {len(list(output_dir.glob('*.json')))} files to {output_dir}")

            # Optional post-processing (consistency fixes + name assignment)
            # Runs in-place by default unless postprocess_output_dir is provided.
            if stage_config.get('enable_postprocess', False):
                postprocess_script = self.base_dir / "persona_generate" / "postprocess_scenarios.py"
                postprocess_output_dir_str = stage_config.get('postprocess_output_dir')
                postprocess_output_dir = (
                    self._resolve_path(postprocess_output_dir_str)
                    if postprocess_output_dir_str
                    else output_dir
                )

                post_cmd = [
                    sys.executable,
                    str(postprocess_script),
                    "--input", str(output_dir),
                    "--output", str(postprocess_output_dir),
                ]

                if stage_config.get('postprocess_model_config'):
                    post_cmd.extend(["--model-config", stage_config['postprocess_model_config']])

                logger.info("Running Stage 1 post-processing...")
                post_success = self._run_command(
                    post_cmd,
                    "Stage 1 Post-process",
                    cwd=self.base_dir / "persona_generate",
                    env=env
                )

                if not post_success:
                    logger.error("Stage 1 post-processing failed")
                    return False

                logger.info(f"Stage 1 post-processing completed: {postprocess_output_dir}")
            
            self.stage_status[1] = True
        
        return success
    
    def run_stage_2_event_graph(self) -> bool:
        """
        Stage 2: Event Graph Extraction from Summaries
        
        Returns:
            True if stage completed successfully, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STAGE 2: Event Graph Extraction")
        logger.info("=" * 80)
        
        stage_config = self.config.get('stage_2_event_graph', {})
        input_dir = self._resolve_path(stage_config.get('input_dir', 'persona_n_scenarios'))
        output_dir = self._resolve_path(stage_config.get('output_dir', 'extracted_event_graphs'))
        
        # Check if stage is enabled
        if not self.config.get('global', {}).get('enable_stage_2', True):
            logger.info("Stage 2 is disabled in configuration. Skipping.")
            # Check if output exists from a previous run
            if self._stage_has_output(output_dir):
                logger.info("Stage 2 output found from previous run. Marking as completed.")
                self.stage_status[2] = True
            else:
                self.stage_status[2] = True  # Mark as completed to allow pipeline to continue
            return True
        
        # Check if input data exists (either from previous stage or previous run)
        if not self.stage_status[1]:
            logger.warning("Stage 1 was not run in this execution. Checking for existing input data...")
        
        # Verify input directory exists
        if not input_dir.exists() or not self._stage_has_output(input_dir):
            logger.error(f"Input directory not found or empty: {input_dir}")
            return False
        
        # Detect residual inputs (output uses the same filename as input)
        _S2_REPORTS = {"event_graph_report.json", "postprocessing_report.json", "generation_report.json", "generation_summary.json"}
        residuals_s2 = self._get_residual_input_files(
            input_dir, output_dir, input_glob="*.json", exclude_filenames=_S2_REPORTS
        )
        if self._stage_has_output(output_dir) and not residuals_s2:
            logger.info("Stage 2: all inputs already processed. Skipping.")
            self.stage_status[2] = True
            return True
        if residuals_s2:
            logger.info("Stage 2: %d unprocessed input(s) detected; running for residuals.", len(residuals_s2))
            effective_input_s2 = self._create_staged_input(residuals_s2, "s2")
        else:
            effective_input_s2 = input_dir

        # Build command
        script_path = self.base_dir / "event_generation" / "event_graph_from_summary" / "build_event_graph.py"
        cmd = [sys.executable, str(script_path)]

        # Add input directory (staged if partial, full if first run)
        cmd.extend(["--input", str(effective_input_s2)])
        
        # Add output directory
        cmd.extend(["--output", str(output_dir)])
        
        # Add model configuration
        if stage_config.get('model'):
            cmd.extend(["--model", stage_config['model']])
        
        if stage_config.get('model_config'):
            cmd.extend(["--config", stage_config['model_config']])
        
        if stage_config.get('api_key'):
            cmd.extend(["--api-key", stage_config['api_key']])

        if stage_config.get('tensor_parallel_size') is not None:
            cmd.extend(["--tensor-parallel-size", str(stage_config['tensor_parallel_size'])])

        # Execute command
        import os
        env = os.environ.copy()
        if stage_config.get('gpus'):
            env['CUDA_VISIBLE_DEVICES'] = str(stage_config['gpus'])
        success = self._run_command(cmd, "Stage 2", env=env)
        
        if success:
            self.stage_status[2] = True
        
        return success
    
    def run_stage_3_graph_expansion(self) -> bool:
        """
        Stage 3: Event Graph Expansion
        
        Returns:
            True if stage completed successfully, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STAGE 3: Event Graph Expansion")
        logger.info("=" * 80)
        
        stage_config = self.config.get('stage_3_graph_expansion', {})
        input_dir = self._resolve_path(stage_config.get('input_dir', 'extracted_event_graphs'))
        output_dir = self._resolve_path(stage_config.get('output_dir', 'expanded_event_graphs'))
        
        # Check if stage is enabled
        if not self.config.get('global', {}).get('enable_stage_3', True):
            logger.info("Stage 3 is disabled in configuration. Skipping.")
            if self._stage_has_output(output_dir):
                logger.info("Stage 3 output found from previous run. Marking as completed.")
                self.stage_status[3] = True
            else:
                self.stage_status[3] = True
            return True
        
        # Check if input data exists
        if not self.stage_status[2]:
            logger.warning("Stage 2 was not run in this execution. Checking for existing input data...")
        
        # Verify input directory exists
        if not input_dir.exists() or not self._stage_has_output(input_dir):
            logger.error(f"Input directory not found or empty: {input_dir}")
            return False
        
        # Detect residual inputs (output uses the same filename as input)
        _S3_REPORTS = {"expansion_report.json"}
        residuals_s3 = self._get_residual_input_files(
            input_dir, output_dir, input_glob="*.json", exclude_filenames=_S3_REPORTS
        )
        if self._stage_has_output(output_dir) and not residuals_s3:
            logger.info("Stage 3: all inputs already processed. Skipping.")
            self.stage_status[3] = True
            return True
        if residuals_s3:
            logger.info("Stage 3: %d unprocessed input(s) detected; running for residuals.", len(residuals_s3))
            effective_input_s3 = self._create_staged_input(residuals_s3, "s3")
        else:
            effective_input_s3 = input_dir

        # Build command
        script_path = self.base_dir / "event_generation" / "graph_expansion" / "expand_event_graph.py"
        cmd = [sys.executable, str(script_path)]

        # Add input directory (staged if partial, full if first run)
        cmd.extend(["--input-dir", str(effective_input_s3)])
        
        # Add output directory
        cmd.extend(["--output-dir", str(output_dir)])
        
        # Add model configuration
        if stage_config.get('model_config'):
            cmd.extend(["--model-config", stage_config['model_config']])
        elif stage_config.get('model'):
            cmd.extend(["--model", stage_config['model']])

        # Execute command
        import os
        env = os.environ.copy()
        if stage_config.get('gpus'):
            env['CUDA_VISIBLE_DEVICES'] = str(stage_config['gpus'])
        success = self._run_command(cmd, "Stage 3", env=env)
        
        if success:
            self.stage_status[3] = True
        
        return success
    
    def run_stage_4_chat_scripts(self) -> bool:
        """
        Stage 4: Chat Script Generation
        
        Returns:
            True if stage completed successfully, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STAGE 4: Chat Script Generation")
        logger.info("=" * 80)
        
        stage_config = self.config.get('stage_4_chat_scripts', {})
        input_dir = self._resolve_path(stage_config.get('input_dir', 'expanded_event_graphs'))
        output_dir = self._resolve_path(stage_config.get('output_dir', 'chat_scripts'))
        
        # Check if stage is enabled
        if not self.config.get('global', {}).get('enable_stage_4', True):
            logger.info("Stage 4 is disabled in configuration. Skipping.")
            if self._stage_has_output(output_dir):
                logger.info("Stage 4 output found from previous run. Marking as completed.")
                self.stage_status[4] = True
            else:
                self.stage_status[4] = True
            return True
        
        # Check if input data exists
        if not self.stage_status[3]:
            logger.warning("Stage 3 was not run in this execution. Checking for existing input data...")
        
        # Verify input directory exists
        if not input_dir.exists() or not self._stage_has_output(input_dir):
            logger.error(f"Input directory not found or empty: {input_dir}")
            return False
        
        # Detect completion: stage 4 writes to a run_XXXXXXX subdir — compare counts
        _s4_latest = self._find_latest_run_dir(output_dir)
        _s4_input_count = len([f for f in input_dir.glob("*.json") if f.is_file()])
        _s4_output_count = len([f for f in _s4_latest.glob("*.json") if f.is_file()]) if _s4_latest.exists() else 0
        if _s4_output_count >= _s4_input_count > 0:
            logger.info("Stage 4: all %d inputs already processed. Skipping.", _s4_input_count)
            self.stage_status[4] = True
            return True
        if 0 < _s4_output_count < _s4_input_count:
            logger.info(
                "Stage 4: %d/%d inputs processed; re-running to complete remaining %d.",
                _s4_output_count, _s4_input_count, _s4_input_count - _s4_output_count,
            )

        # Build command
        script_path = self.base_dir / "event_generation" / "graph_to_chat" / "generate_chat_scripts.py"
        cmd = [sys.executable, str(script_path)]
        
        # Add input directory
        cmd.extend(["--input-dir", str(input_dir)])
        
        # Add output directory
        cmd.extend(["--output-dir", str(output_dir)])
        
        # Add model configuration
        if stage_config.get('model_config'):
            cmd.extend(["--model-config", stage_config['model_config']])
        elif stage_config.get('model'):
            cmd.extend(["--model", stage_config['model']])
        
        # Add LLM usage flag
        if not stage_config.get('use_llm', True):
            cmd.append("--no-llm")

        # Execute command
        import os
        env = os.environ.copy()
        if stage_config.get('gpus'):
            env['CUDA_VISIBLE_DEVICES'] = str(stage_config['gpus'])
        success = self._run_command(cmd, "Stage 4", env=env)
        
        if success:
            self.stage_status[4] = True
        
        return success
    
    def run_stage_5_roleplay_dialogue(self) -> bool:
        """
        Stage 5: Roleplay Dialogue Generation
        
        Returns:
            True if stage completed successfully, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STAGE 5: Roleplay Dialogue Generation")
        logger.info("=" * 80)
        
        stage_config = self.config.get('stage_5_roleplay_dialogue', {})
        input_dir = self._resolve_path(stage_config.get('input_dir', 'chat_scripts'))
        output_dir = self._resolve_path(stage_config.get('output_dir', 'generated_dialogues'))
        
        # Check if stage is enabled
        if not self.config.get('global', {}).get('enable_stage_5', True):
            logger.info("Stage 5 is disabled in configuration. Skipping.")
            if self._stage_has_output(output_dir):
                logger.info("Stage 5 output found from previous run. Marking as completed.")
                self.stage_status[5] = True
            else:
                self.stage_status[5] = True
            return True
        
        # Check if input data exists
        if not self.stage_status[4]:
            logger.warning("Stage 4 was not run in this execution. Checking for existing input data...")
        
        # Verify input directory exists
        if not input_dir.exists() or not self._stage_has_output(input_dir):
            logger.error(f"Input directory not found or empty: {input_dir}")
            return False
        
        # Detect completion: stage 5 writes to a run_XXXXXXX subdir — compare counts
        _s5_latest = self._find_latest_run_dir(output_dir)
        _s5_input_count = len([f for f in input_dir.glob("*.json") if f.is_file()])
        _s5_output_count = len([f for f in _s5_latest.glob("*.json") if f.is_file()]) if _s5_latest.exists() else 0
        if _s5_output_count >= _s5_input_count > 0:
            logger.info("Stage 5: all %d inputs already processed. Skipping.", _s5_input_count)
            self.stage_status[5] = True
            return True
        if 0 < _s5_output_count < _s5_input_count:
            logger.info(
                "Stage 5: %d/%d inputs processed; re-running to complete remaining %d.",
                _s5_output_count, _s5_input_count, _s5_input_count - _s5_output_count,
            )

        # Build command
        script_path = self.base_dir / "roleplay_dialogue" / "generate_roleplay_dialogues.py"
        cmd = [sys.executable, str(script_path)]
        
        # Add input directory
        cmd.extend(["--input-dir", str(input_dir)])
        
        # Add output directory
        cmd.extend(["--output-dir", str(output_dir)])
        
        # Add model configuration
        if stage_config.get('model_config'):
            cmd.extend(["--model-config", stage_config['model_config']])
        elif stage_config.get('model'):
            cmd.extend(["--model", stage_config['model']])
        
        # Add persona directory
        persona_dir = stage_config.get('persona_dir')
        if persona_dir:
            cmd.extend(["--persona-dir", str(self._resolve_path(persona_dir))])
        else:
            # Use stage 1 output by default
            stage1_config = self.config.get('stage_1_persona_generation', {})
            persona_dir = self._resolve_path(stage1_config.get('output_dir', 'persona_n_scenarios'))
            cmd.extend(["--persona-dir", str(persona_dir)])
        
        # Add turn limits
        cmd.extend(["--min-turns", str(stage_config.get('min_turns', 2))])
        cmd.extend(["--max-turns", str(stage_config.get('max_turns', 10))])
        
        # Add adaptive length flag
        if not stage_config.get('adaptive_length', True):
            cmd.append("--no-adaptive-length")
        
        # Add GPU configuration
        if stage_config.get('gpus'):
            cmd.extend(["--gpus", stage_config['gpus']])
        
        if stage_config.get('tensor_parallel_size'):
            cmd.extend(["--tensor-parallel-size", str(stage_config['tensor_parallel_size'])])
        
        # Add post-processing flags
        if not stage_config.get('enable_enforcer', True):
            cmd.append("--no-enforcer")
        
        # Add prompt template parameters
        if stage_config.get('prompt_template'):
            cmd.extend(["--prompt-template", stage_config['prompt_template']])
        
        if stage_config.get('template_name'):
            cmd.extend(["--template-name", stage_config['template_name']])

        if stage_config.get('bare_prompt', False):
            cmd.append("--bare-prompt")
        
        # Execute command
        success = self._run_command(cmd, "Stage 5")
        
        if success:
            self.stage_status[5] = True
        
        return success
    
    def run_stage_6_toxic_injection(self) -> bool:
        """
        Stage 6: CAA-Steered Toxic Rewrite (Levelled)
        
        Uses Contrastive Activation Addition (CAA) to rewrite perpetrator
        messages at escalation-matched steering levels:
            Escalation 2 → Level 1  mild
            Escalation 3 → Level 2  moderate
            Escalation 4 → Level 3  strong
        
        Returns:
            True if stage completed successfully, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STAGE 6: CAA-Steered Toxic Rewrite (Levelled)")
        logger.info("=" * 80)
        
        stage_config = self.config.get('stage_6_toxic_injection', {})
        input_dir = self._resolve_path(stage_config.get('input_dir', 'generated_dialogues'))
        output_dir = self._resolve_path(stage_config.get('output_dir', 'toxic_injected_dialogues'))
        
        # Check if stage is enabled
        if not self.config.get('global', {}).get('enable_stage_6', True):
            logger.info("Stage 6 is disabled in configuration. Skipping.")
            if self._stage_has_output(output_dir):
                logger.info("Stage 6 output found from previous run. Marking as completed.")
                self.stage_status[6] = True
            else:
                self.stage_status[6] = True
            return True
        
        # Check if input data exists
        if not self.stage_status[5]:
            logger.warning("Stage 5 was not run in this execution. Checking for existing input data...")
        
        # Verify input directory exists
        if not input_dir.exists() or not self._stage_has_output(input_dir):
            logger.error(f"Input directory not found or empty: {input_dir}")
            return False
        
        # Detect completion: compare flat output file count against input count
        _s6_input_count = len([f for f in input_dir.glob("*.json") if f.is_file()])
        _s6_output_count = len([f for f in output_dir.glob("*.json") if f.is_file()]) if output_dir.exists() else 0
        if _s6_output_count >= _s6_input_count > 0:
            logger.info("Stage 6: all %d inputs already processed. Skipping.", _s6_input_count)
            self.stage_status[6] = True
            return True
        if 0 < _s6_output_count < _s6_input_count:
            logger.info(
                "Stage 6: %d/%d inputs processed; re-running to complete remaining %d.",
                _s6_output_count, _s6_input_count, _s6_input_count - _s6_output_count,
            )

        # CAA steering directory
        steer_dir = self.base_dir / "toxic_steering"
        
        # Build command — run as module from toxic_steering/ directory
        cmd = [sys.executable, "-m", "levelled_rewrite.run_levelled_rewrite"]
        
        # Add input directory (absolute so it works from steer_dir cwd)
        cmd.extend(["--input", str(input_dir)])
        
        # Add output directory
        cmd.extend(["--output", str(output_dir)])
        
        # Steering assets (not shipped with the public release; see
        # toxic_steering/README.md). Paths are relative to toxic_steering/.
        missing = [k for k in ('model', 'calibration', 'vector_dir') if not stage_config.get(k)]
        if missing:
            logger.error(
                "Stage 6 needs the calibrated steering assets; missing config keys: %s. "
                "See toxic_steering/README.md, or set global.enable_stage_6: false.",
                ", ".join(missing),
            )
            return False

        cmd.extend(["--model", stage_config['model']])
        cmd.extend(["--calibration", str(stage_config['calibration'])])
        cmd.extend(["--vector-dir", str(stage_config['vector_dir'])])
        if stage_config.get('prompts'):
            cmd.extend(["--prompts", str(stage_config['prompts'])])

        # Sampling parameters – per-level injection ratios
        injection_ratio = stage_config.get('injection_ratio', {"2": 0.05, "3": 0.10, "4": 0.20})
        if isinstance(injection_ratio, dict):
            import json as _json
            cmd.extend(["--injection-ratio", _json.dumps({str(k): v for k, v in injection_ratio.items()})])
        else:
            # Single float — will be broadcast to all levels in the rewrite script
            cmd.extend(["--injection-ratio", str(injection_ratio)])
        
        max_retries = stage_config.get('max_retries', 10)
        cmd.extend(["--max-retries", str(max_retries)])
        
        seed = stage_config.get('random_seed', 42)
        cmd.extend(["--seed", str(seed)])
        
        min_per_conv = stage_config.get('min_messages_per_conversation', 1)
        cmd.extend(["--min-per-conversation", str(min_per_conv)])
        
        # Escalation levels to process
        esc_levels = stage_config.get('escalation_levels', [2, 3, 4])
        cmd.extend(["--escalation-levels"] + [str(l) for l in esc_levels])

        # Optional escalation→CAA-level mapping override
        level_map = stage_config.get('level_map', None)
        if level_map is not None:
            import json as _json2
            cmd.extend(["--level-map", _json2.dumps({str(k): v for k, v in level_map.items()})])

        # Optional CAA-level→alpha override (on top of calibration file)
        alpha_map = stage_config.get('alpha_map', None)
        if alpha_map is not None:
            import json as _json3
            cmd.extend(["--alpha-map", _json3.dumps({str(k): v for k, v in alpha_map.items()})])

        # Batched generation and parallel labeling
        batch_size = stage_config.get('batch_size', 8)
        cmd.extend(["--batch-size", str(batch_size)])

        label_workers = stage_config.get('label_workers', 4)
        cmd.extend(["--label-workers", str(label_workers)])
        
        # Labeling options
        if stage_config.get('skip_labeling', False):
            cmd.append("--skip-labeling")
        else:
            label_backend = stage_config.get('label_backend', 'openai')
            cmd.extend(["--label-backend", label_backend])

            gpt_temp = stage_config.get('gpt_temperature', 0.3)
            cmd.extend(["--gpt-temperature", str(gpt_temp)])

            if label_backend == 'openai':
                gpt_model = stage_config.get('gpt_model', 'gpt-5.2')
                cmd.extend(["--gpt-model", gpt_model])
                if stage_config.get('gpt_api_base'):
                    cmd.extend(["--gpt-api-base", stage_config['gpt_api_base']])
            else:
                # vLLM backend (any local model served by vLLM)
                qwen_model = stage_config.get('qwen_model', 'Qwen/Qwen3.5-35B-A3B')
                cmd.extend(["--qwen-model", qwen_model])
                if stage_config.get('vllm_dtype'):
                    cmd.extend(["--vllm-dtype", stage_config['vllm_dtype']])
                if stage_config.get('vllm_tensor_parallel_size'):
                    cmd.extend([
                        "--vllm-tensor-parallel-size",
                        str(stage_config['vllm_tensor_parallel_size']),
                    ])
                if stage_config.get('vllm_gpu_memory_utilization'):
                    cmd.extend([
                        "--vllm-gpu-memory-utilization",
                        str(stage_config['vllm_gpu_memory_utilization']),
                    ])
                if stage_config.get('vllm_max_model_len'):
                    cmd.extend([
                        "--vllm-max-model-len",
                        str(stage_config['vllm_max_model_len']),
                    ])
        
        # Set CUDA_VISIBLE_DEVICES if gpus specified
        import os
        env = os.environ.copy()
        if stage_config.get('gpus'):
            env['CUDA_VISIBLE_DEVICES'] = stage_config['gpus']

        # Avoid intermittent HF Xet download crashes on cluster filesystems
        env.setdefault('HF_HUB_DISABLE_XET', '1')

        # Optional Hugging Face token for authenticated Hub downloads
        if stage_config.get('hf_token'):
            env['HF_TOKEN'] = stage_config['hf_token']
            env['HUGGINGFACE_HUB_TOKEN'] = stage_config['hf_token']
        
        # Execute command from toxic_steering/ directory
        logger.info(f"Working directory: {steer_dir}")
        success = self._run_command(cmd, "Stage 6", cwd=steer_dir, env=env)
        
        if success:
            self.stage_status[6] = True
        
        return success
    
    def run_stage_7_convert_dialogues(self) -> bool:
        """
        Stage 7: Convert Dialogues to Minimal Format with PDF Generation
        
        Returns:
            True if stage completed successfully, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STAGE 7: Convert Dialogues to Minimal Format")
        logger.info("=" * 80)
        
        stage_config = self.config.get('stage_7_convert_dialogues', {})
        input_dir = self._resolve_path(stage_config.get('input_dir', 'toxic_injected_dialogues'))
        output_dir = self._resolve_path(stage_config.get('output_dir', 'converted_dialogues'))
        
        # Check if stage is enabled
        if not self.config.get('global', {}).get('enable_stage_7', True):
            logger.info("Stage 7 is disabled in configuration. Skipping.")
            # Check if output exists from a previous run
            if self._stage_has_output(output_dir):
                logger.info("Stage 7 output found from previous run. Marking as completed.")
                self.stage_status[7] = True
            else:
                self.stage_status[7] = True  # Mark as completed to allow pipeline to continue
            return True
        
        # Check if input data exists
        if not self.stage_status[6]:
            logger.warning("Stage 6 was not run in this execution. Checking for existing input data...")
        
        # Verify input directory exists
        if not input_dir.exists() or not self._stage_has_output(input_dir):
            logger.error(f"Input directory not found or empty: {input_dir}")
            return False
        
        # Detect residual inputs (output uses the same filename as input)
        residuals_s7 = self._get_residual_input_files(input_dir, output_dir)
        if self._stage_has_output(output_dir) and not residuals_s7:
            logger.info("Stage 7: all inputs already processed. Skipping.")
            self.stage_status[7] = True
            return True
        if residuals_s7:
            logger.info("Stage 7: %d unprocessed input(s) detected; running for residuals.", len(residuals_s7))
            effective_input_s7 = self._create_staged_input(residuals_s7, "s7")
        else:
            effective_input_s7 = input_dir

        # Build command
        script_path = self.base_dir / "convert_dialogues_to_minimal.py"
        cmd = [sys.executable, str(script_path)]

        # Add input directory (staged if partial, full if first run)
        cmd.extend(["-i", str(effective_input_s7)])
        
        # Add output directory
        cmd.extend(["-o", str(output_dir)])
        
        # Add recursive flag if specified
        if stage_config.get('recursive', False):
            cmd.append("-r")
        
        # Add observe-toxic flag if specified
        if stage_config.get('observe_toxic', False):
            cmd.append("--observe-toxic")
        
        # Add no-pdf flag if PDF generation is disabled
        if not stage_config.get('generate_pdf', True):
            cmd.append("--no-pdf")
        
        # Add gibberish review flag
        if stage_config.get('review_gibberish', True):
            cmd.append("--review-gibberish")
            gpt_model = stage_config.get('gpt_model', 'gpt-5.2')
            cmd.extend(["--gpt-model", gpt_model])
            gpt_api_base = stage_config.get('gpt_api_base')
            if gpt_api_base:
                cmd.extend(["--gpt-api-base", gpt_api_base])
        
        # Execute command
        success = self._run_command(cmd, "Stage 7")
        
        if success:
            self.stage_status[7] = True
        
        return success
    
    def _build_judge_cmd(
        self,
        stage_config: dict,
        input_dir: Path,
        output_dir: Path,
    ) -> Tuple[list, dict]:
        """
        Build the run_judge.py command and environment dict for a single judge config.

        Returns:
            (cmd, env) tuple ready for subprocess.Popen.
        """
        script_path = self.base_dir / "LLM_judge" / "run_judge.py"
        cmd = [sys.executable, str(script_path)]
        cmd.extend(["--input", str(input_dir)])
        cmd.extend(["--output-dir", str(output_dir)])

        judge_config = stage_config.get('judge_config')
        if judge_config:
            config_path = Path(judge_config)
            if not config_path.is_absolute():
                config_path = self.base_dir / config_path
            cmd.extend(["--config", str(config_path)])
        else:
            cmd.extend(["--config", str(self.base_dir / "LLM_judge" / "configs" / "judge_config.yaml")])

        if stage_config.get('model'):
            cmd.extend(["--model", stage_config['model']])
        if stage_config.get('api_key'):
            cmd.extend(["--api-key", stage_config['api_key']])
        if stage_config.get('backend'):
            cmd.extend(["--backend", stage_config['backend']])
        if stage_config.get('base_url'):
            cmd.extend(["--base-url", stage_config['base_url']])
        if stage_config.get('gpu_memory_utilization') is not None:
            cmd.extend(["--gpu-memory-utilization", str(stage_config['gpu_memory_utilization'])])
        if stage_config.get('tensor_parallel_size') is not None:
            cmd.extend(["--tensor-parallel-size", str(stage_config['tensor_parallel_size'])])
        if stage_config.get('max_num_seqs') is not None:
            cmd.extend(["--max-num-seqs", str(stage_config['max_num_seqs'])])

        env = os.environ.copy()
        backend = stage_config.get('backend', 'openai').lower()
        if stage_config.get('api_key'):
            if backend == 'deepseek':
                env['DEEPSEEK_API_KEY'] = stage_config['api_key']
            elif backend == 'gemini':
                env['GEMINI_API_KEY'] = stage_config['api_key']
            else:
                env['OPENAI_API_KEY'] = stage_config['api_key']
        if stage_config.get('gpus'):
            env['CUDA_VISIBLE_DEVICES'] = str(stage_config['gpus'])

        return cmd, env

    def _run_parallel_judges(self, judge_configs: list, input_dir: Path) -> bool:
        """
        Run multiple LLM judge configurations in parallel (one subprocess per judge).

        Each entry in judge_configs must have its own ``output_dir``.
        All judges share the same ``input_dir`` unless overridden per-entry.

        Config example (pipeline YAML)::

            stage_8_judges:
              - label: "deepseek"
                backend: "deepseek"
                model: "deepseek-v4-flash"
                output_dir: "Generation_Results/8_judge_results/run_deepseek"
              - label: "gemini"
                backend: "gemini"
                model: "gemini-3.1-flash-lite"
                output_dir: "Generation_Results/8_judge_results/run_gemini"

        Returns:
            True only if ALL judges complete successfully.
        """
        logger.info("Launching %d judge(s) in parallel …", len(judge_configs))

        def _run_one(cfg: dict) -> Tuple[str, bool]:
            label = cfg.get('label', cfg.get('backend', 'judge'))
            out_dir = self._resolve_path(cfg['output_dir'])
            in_dir  = self._resolve_path(cfg['input_dir']) if cfg.get('input_dir') else input_dir
            cmd, env = self._build_judge_cmd(cfg, in_dir, out_dir)

            logger.info("[%s] Starting: %s", label, ' '.join(cmd))
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=self.base_dir,
                    env=env,
                )
                for line in process.stdout:
                    logger.info("[%s] %s", label, line.rstrip())
                process.wait()
                ok = process.returncode == 0
                if ok:
                    logger.info("[%s] Completed successfully.", label)
                else:
                    logger.error("[%s] Failed (exit %d).", label, process.returncode)
                return label, ok
            except Exception as exc:
                logger.error("[%s] Error: %s", label, exc)
                return label, False

        results: Dict[str, bool] = {}
        with ThreadPoolExecutor(max_workers=len(judge_configs)) as pool:
            futures = {pool.submit(_run_one, cfg): cfg for cfg in judge_configs}
            for future in as_completed(futures):
                label, ok = future.result()
                results[label] = ok

        all_ok = all(results.values())
        for label, ok in results.items():
            status = "OK" if ok else "FAILED"
            logger.info("  Judge %-20s → %s", label, status)
        return all_ok

    def run_stage_8_llm_judge(self) -> bool:
        """
        Stage 8: LLM-as-Judge Dialogue Evaluation

        Single-judge mode  — use ``stage_8_llm_judge`` in the pipeline config.
        Parallel-judge mode — use ``stage_8_judges`` (list) instead; all judges
        run simultaneously, each writing to its own ``output_dir``.

        Returns:
            True if stage completed successfully, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STAGE 8: LLM-as-Judge Dialogue Evaluation")
        logger.info("=" * 80)

        # Check if stage is enabled
        if not self.config.get('global', {}).get('enable_stage_8', True):
            logger.info("Stage 8 is disabled in configuration. Skipping.")
            self.stage_status[8] = True
            return True

        # Warn if Stage 7 input is missing
        if not self.stage_status[7]:
            logger.warning("Stage 7 was not run in this execution. Checking for existing input data...")

        # Resolve the shared input directory (used for both single and parallel modes)
        stage_config = self.config.get('stage_8_llm_judge', {})
        default_input = self.config.get('stage_7_convert_dialogues', {}).get('output_dir', 'converted_dialogues')
        input_dir = self._resolve_path(stage_config.get('input_dir', default_input))

        if not input_dir.exists() or not self._stage_has_output(input_dir):
            logger.error(f"Input directory not found or empty: {input_dir}")
            return False

        # ── Parallel mode ────────────────────────────────────────────────────
        parallel_configs: Optional[List[dict]] = self.config.get('stage_8_judges')
        if parallel_configs:
            logger.info("Parallel judge mode: %d judge(s) configured.", len(parallel_configs))
            success = self._run_parallel_judges(parallel_configs, input_dir)
            if success:
                self.stage_status[8] = True
            return success

        # ── Single-judge mode ─────────────────────────────────────────────────
        output_dir = self._resolve_path(stage_config.get('output_dir', 'judge_results'))

        # Detect residual inputs: {stem}_dialogues.json → {stem}_dialogues_judge_results.json
        residuals_s8 = self._get_residual_input_files(
            input_dir, output_dir,
            input_glob="*_dialogues.json",
            output_name_fn=lambda n: Path(n).stem + "_judge_results.json",
        )
        _s8_total = len(list(input_dir.glob("*_dialogues.json")))
        if not residuals_s8 and (output_dir / "batch_summary.json").exists():
            logger.info("Stage 8: all %d inputs already evaluated. Skipping.", _s8_total)
            self.stage_status[8] = True
            return True
        if residuals_s8:
            logger.info(
                "Stage 8: %d/%d dialogue file(s) not yet evaluated; proceeding.",
                len(residuals_s8), _s8_total,
            )
            stale = output_dir / "batch_summary.json"
            if stale.exists():
                stale.unlink()
                logger.info("Removed stale batch_summary.json for regeneration.")

        logger.info(f"Judge input:  {input_dir}")
        logger.info(f"Judge output: {output_dir}")

        cmd, env = self._build_judge_cmd(stage_config, input_dir, output_dir)
        success = self._run_command(cmd, "Stage 8", env=env)

        if success:
            self.stage_status[8] = True

        return success

    
    def run_pipeline(self) -> bool:
        """
        Run the complete pipeline across all enabled stages.
        
        Returns:
            True if all stages completed successfully, False otherwise
        """
        logger.info("=" * 80)
        logger.info("VAWG DATASET GENERATION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Run directory: {self.run_dir}")
        logger.info("=" * 80)
        
        stages = [
            (1, "Persona & Scenarios Generation", self.run_stage_1_persona_generation),
            (2, "Event Graph Extraction", self.run_stage_2_event_graph),
            (3, "Event Graph Expansion", self.run_stage_3_graph_expansion),
            (4, "Chat Script Generation", self.run_stage_4_chat_scripts),
            (5, "Roleplay Dialogue Generation", self.run_stage_5_roleplay_dialogue),
            (6, "CAA-Steered Toxic Rewrite", self.run_stage_6_toxic_injection),
            (7, "Convert Dialogues to Minimal Format", self.run_stage_7_convert_dialogues),
            (8, "LLM-as-Judge Dialogue Evaluation", self.run_stage_8_llm_judge),
        ]
        
        failed_stages = []
        
        for stage_num, stage_name, stage_func in stages:
            try:
                success = stage_func()
                if not success:
                    failed_stages.append((stage_num, stage_name))
                    logger.error(f"Stage {stage_num} ({stage_name}) failed")
                    
                    # Ask if user wants to continue
                    logger.warning("Stage failed. Stopping pipeline.")
                    break
            except Exception as e:
                logger.error(f"Unexpected error in Stage {stage_num} ({stage_name}): {e}")
                failed_stages.append((stage_num, stage_name))
                break
        
        # Summary
        logger.info("=" * 80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Run directory: {self.run_dir}")
        
        for stage_num, completed in self.stage_status.items():
            status = "✓ COMPLETED" if completed else "✗ FAILED/SKIPPED"
            logger.info(f"  Stage {stage_num}: {status}")
        
        if failed_stages:
            logger.error(f"Pipeline completed with {len(failed_stages)} failed stage(s)")
            for stage_num, stage_name in failed_stages:
                logger.error(f"  - Stage {stage_num}: {stage_name}")
            return False
        else:
            logger.info("✓ Pipeline completed successfully!")
            logger.info(f"All outputs are available in: {self.run_dir}")
            return True


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the complete VAWG dataset generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config file
  python run_batch_pipeline.py
  
  # Run with custom config file
  python run_batch_pipeline.py --config my_config.yaml
  
Configuration:
  The pipeline uses a YAML configuration file (default: pipeline_config.yaml)
  to control all aspects of the generation process. See pipeline_config.yaml
  for detailed documentation of all options.
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline_config.yaml",
        help="Path to pipeline configuration file (default: pipeline_config.yaml)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get base directory (repo root)
    base_dir = Path(__file__).parent.resolve()
    
    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please create a pipeline_config.yaml file. See pipeline_config.yaml for template.")
        sys.exit(1)
    
    # Create and run pipeline
    runner = PipelineRunner(config_path, base_dir)
    success = runner.run_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
