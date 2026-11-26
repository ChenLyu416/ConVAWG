"""Command-line interface for the LLM structurer."""

import click
import logging
from pathlib import Path
from tqdm import tqdm

from llm_structurer.config import Config
from llm_structurer.ingest import DocumentIngestor
from llm_structurer.pipeline import ExtractionPipeline
from llm_structurer.storage import CaseStorage


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """LLM-driven VAWG case structurer CLI."""
    pass


@cli.command()
@click.option('--input-dir', required=True, help='Directory containing case documents')
@click.option('--output', required=True, help='Output JSONL file path')
def ingest(input_dir, output):
    """Ingest raw case documents."""
    logger.info(f"Ingesting documents from {input_dir}")
    
    ingestor = DocumentIngestor()
    documents = ingestor.ingest_directory(input_dir, pattern="**/*")
    ingestor.save_ingested(documents, output)
    
    logger.info(f"Ingestion complete. Saved {len(documents)} documents to {output}")


@cli.command()
@click.option('--input-dir', help='Directory containing case documents')
@click.option('--input-file', help='JSONL file with ingested cases')
@click.option('--output-dir', required=True, help='Output directory for structured cases')
@click.option('--config', default='configs/qwen_14b_gguf.yaml', help='Configuration file path')
@click.option('--shard-index', type=int, help='Shard index for distributed processing')
@click.option('--num-shards', type=int, help='Total number of shards')
def extract(input_dir, input_file, output_dir, config, shard_index, num_shards):
    """Extract structured information from cases using LLM."""
    logger.info("Initializing extraction pipeline")
    
    cfg = Config(config)
    pipeline = ExtractionPipeline(cfg)
    storage = CaseStorage(cfg.db_path)
    ingestor = DocumentIngestor()
    
    if input_file:
        logger.info(f"Loading cases from {input_file}")
        documents = ingestor.load_ingested(input_file)
    elif input_dir:
        logger.info(f"Ingesting cases from {input_dir}")
        documents = ingestor.ingest_directory(input_dir, pattern="**/*")
    else:
        raise click.UsageError("Either --input-dir or --input-file must be specified")
    
    if shard_index is not None and num_shards is not None:
        logger.info(f"Processing shard {shard_index}/{num_shards}")
        shard_size = len(documents) // num_shards
        start_idx = shard_index * shard_size
        end_idx = start_idx + shard_size if shard_index < num_shards - 1 else len(documents)
        documents = documents[start_idx:end_idx]
        logger.info(f"Processing {len(documents)} documents in this shard")
    
    logger.info(f"Processing {len(documents)} cases")
    
    output_jsonl = Path(output_dir) / "cases.jsonl"
    
    for doc in tqdm(documents, desc="Extracting cases"):
        try:
            case = pipeline.process_case(doc)
            
            if pipeline.validate_case(case):
                if 'json' in cfg.output_formats:
                    storage.save_case_to_json(case, output_dir)
                
                if 'jsonl' in cfg.output_formats:
                    storage.save_case_to_jsonl(case, str(output_jsonl))
                
                if cfg.enable_storage:
                    storage.save_case_to_db(case)
            else:
                logger.warning(f"Case {doc.get('case_id')} failed validation")
                
        except Exception as e:
            logger.error(f"Failed to process case {doc.get('case_id')}: {e}", exc_info=True)
    
    logger.info("Extraction complete")


@cli.command()
@click.option('--input-dir', help='Directory containing JSON case files')
@click.option('--output', required=True, help='Output JSONL file path')
@click.option('--format', default='jsonl', help='Output format (jsonl)')
@click.option('--config', default='configs/qwen_14b_gguf.yaml', help='Configuration file path')
def export(input_dir, output, format, config):
    """Export structured cases to specified format."""
    logger.info("Exporting cases")
    
    cfg = Config(config)
    
    if input_dir:
        logger.info(f"Exporting from directory: {input_dir}")
        
        import json
        from llm_structurer.schema import Case
        
        cases = []
        for json_file in Path(input_dir).glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    case_dict = json.load(f)
                    cases.append(case_dict)
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
        
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, 'w', encoding='utf-8') as f:
            for case_dict in cases:
                f.write(json.dumps(case_dict, ensure_ascii=False, default=str) + '\n')
        
        logger.info(f"Exported {len(cases)} cases to {output}")
        
    else:
        logger.info("Exporting from database")
        storage = CaseStorage(cfg.db_path)
        storage.export_all_to_jsonl(output)
    
    logger.info("Export complete")


if __name__ == '__main__':
    cli()
