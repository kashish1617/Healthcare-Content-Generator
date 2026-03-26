"""CLI for Healthcare Content Generator."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from src.content_generator import create_generator
from src import prompts


CONTENT_TYPE_CHOICES = list(prompts.CONTENT_TYPES.keys())


def main():
    parser = argparse.ArgumentParser(
        description="Healthcare Content Generator: transform topics into professional outputs (Patient Summaries, clinical notes, education handouts) using GenAI, Vector DB, and prompt engineering.",
    )
    parser.add_argument(
        "topic",
        nargs="*",
        default=None,
        help="Topic or case description (e.g., '58-year-old male, chest pain x 2 hours, HTN, DM'). If omitted, a default example is used.",
    )
    parser.add_argument(
        "-t",
        "--type",
        choices=CONTENT_TYPE_CHOICES,
        default="patient_summary",
        metavar="TYPE",
        help=f"Content type to generate. Choices: {CONTENT_TYPE_CHOICES}. Default: patient_summary.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature (0–1). Default: 0.3.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens for generation. Default: 2048.",
    )
    args = parser.parse_args()

    topic = " ".join(args.topic).strip() if args.topic else "55-year-old female, shortness of breath, history of asthma."
    if not topic:
        topic = "55-year-old female, shortness of breath, history of asthma."

    settings = get_settings()
    if not settings.openai_api_key:
        print("Error: Set OPENAI_API_KEY in .env")
        sys.exit(1)

    config = {
        "openai_api_key": settings.openai_api_key,
        "openai_api_base": settings.openai_api_base,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "chroma_persist_dir": settings.chroma_persist_dir,
    }
    gen = create_generator(config)
    label = args.type.replace("_", " ").title()
    print(f"Generating {label} for topic: {topic[:80]}{'...' if len(topic) > 80 else ''}")
    print("-" * 60)
    print(gen(topic=topic, content_type=args.type, temperature=args.temperature, max_tokens=args.max_tokens))
    print("-" * 60)


if __name__ == "__main__":
    main()
