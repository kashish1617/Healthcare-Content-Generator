"""Healthcare content generation orchestration."""
from src.vector_store import get_vector_store, retrieve_context
from src.llm_client import get_client, generate
from src import prompts


def create_generator(config: dict):
    """Create a content generator with vector store and LLM client."""
    persist_dir = config.get("chroma_persist_dir", "./data/chroma_db")
    embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
    collection, embedder = get_vector_store(persist_dir, embedding_model)

    api_key = config.get("openai_api_key")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required. Set it in .env or pass openai_api_key in config.")
    client = get_client(
        api_key=api_key,
        base_url=config.get("openai_api_base"),
    )
    model = config.get("llm_model", "gpt-4o-mini")

    def generate_content(
        topic: str,
        content_type: str = "patient_summary",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        top_k_context: int = 5,
    ) -> str:
        builder = prompts.CONTENT_TYPES.get(content_type)
        if not builder:
            raise ValueError(
                f"Unknown content_type: {content_type}. "
                f"Choose from: {list(prompts.CONTENT_TYPES.keys())}"
            )
        context = retrieve_context(collection, embedder, topic, top_k=top_k_context)
        system_prompt, user_prompt = builder(context, topic)
        return generate(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return generate_content
