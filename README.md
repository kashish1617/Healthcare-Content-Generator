
# DAI- 42 Healthcare Content Generator

# <h2>Division</h2>

A **GenAI-powered content generation tool** for healthcare professionals. It turns basic topics into high-quality, professional outputs—such as **Patient Summaries**, clinical notes, and patient education handouts—using **Prompt Engineering**, a **Vector DB** for terminology/guidelines, and an **LLM API**.

## Features

- **Patient Summaries**: Structured summaries with Chief Complaint, HPI, PMH, Medications, Allergies, Vitals, Findings, Assessment, and Plan.
- **Clinical notes**: SOAP/progress note style with consistent headings and standard abbreviations.
- **Education handouts**: Patient-friendly handouts with clear structure and when-to-seek-care guidance.
- **Consistent terminology**: Vector DB (Chroma) stores healthcare terminology and formatting rules; **RAG** retrieves relevant guidelines so the LLM output stays aligned with industry standards.
- **Configurable**: Temperature, max tokens, and content type selectable via API or UI.

## Tech Stack

| Component   | Technology        |
|------------|-------------------|
| Language   | Python 3.10+      |
| Vector DB  | ChromaDB          |
| Embeddings | sentence-transformers (e.g. all-MiniLM-L6-v2) |
| LLM        | OpenAI API (or compatible) |
| Backend API| FastAPI            |
| UI         | Streamlit          |

## Project Structure

```
healthcare-content-generator/
├── api/
│   └── main.py              # FastAPI app
├── data/
│   └── healthcare_guidelines.json   # Terminology & formatting rules (seeded into Vector DB)
├── src/
│   ├── vector_store.py      # ChromaDB + embeddings, RAG retrieval
│   ├── llm_client.py        # OpenAI client and generate()
│   ├── prompts.py           # Prompt templates (Patient Summary, clinical note, handout)
│   └── content_generator.py # Orchestrator: vector store + prompts + LLM
├── config.py                # Settings from .env
├── app_streamlit.py         # Streamlit UI
├── cli.py                   # CLI for quick generation
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Clone / navigate to the project**
   ```bash
   cd healthcare-content-generator
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # source venv/bin/activate  # Linux/macOS
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   - Copy `.env.example` to `.env`.
   - Set your **OpenAI API key**:
     ```
     OPENAI_API_KEY=sk-your-key-here
     ```
   - Optional: `OPENAI_API_BASE` (e.g. for Azure), `LLM_MODEL`, `EMBEDDING_MODEL`, `CHROMA_PERSIST_DIR`.

## Usage

### Streamlit UI

```bash
streamlit run app_streamlit.py
```

- Choose **Content type** (Patient Summary, Clinical note, Education handout).
- Enter a **topic** or case description (e.g. “58-year-old male, chest pain x 2 hours, HTN, DM”).
- Click **Generate content**. The app uses the Vector DB for context and the LLM for generation.

### FastAPI backend

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- **Health**: `GET /health`
- **Content types**: `GET /content-types`
- **Generate**: `POST /generate` with JSON body:
  ```json
  {
    "topic": "55-year-old female, SOB, history of asthma",
    "content_type": "patient_summary",
    "temperature": 0.3,
    "max_tokens": 2048,
    "top_k_context": 5
  }
  ```
  Optional `top_k_context` (default 5) controls how many guideline chunks are retrieved from the Vector DB for RAG.

### CLI

```bash
python cli.py [topic] [-t TYPE] [--temperature 0.3] [--max-tokens 2048]
```

Examples:
```bash
# Patient Summary (default)
python cli.py 55-year-old female shortness of breath history of asthma

# Clinical note
python cli.py "58-year-old male, chest pain x 2 hours" -t clinical_note

# Education handout
python cli.py "post-op knee replacement care" -t education_handout
```

Use `python cli.py --help` for all options. Generates content and prints it to the console.

## How It Works

1. **Vector DB (Chroma)**  
   On first run, `healthcare_guidelines.json` is loaded and embedded with **sentence-transformers**. Documents (terminology, formatting rules, summary section names) are stored in Chroma.

2. **RAG at generation time**  
   For each request, the user’s **topic** is used as a query. The top-k relevant guideline chunks are retrieved and passed as **context** into the prompt.

3. **Prompt engineering**  
   `prompts.py` defines:
   - A **system** prompt (role: medical writer, tone, structure, use of context).
   - **User** prompts that inject the retrieved context and the user’s topic.
   Different templates exist for **patient_summary**, **clinical_note**, and **education_handout**.

4. **LLM**  
   The OpenAI (or compatible) API is called with the composed messages; the model’s reply is returned as the generated content.

## Customization

- **Terminology and rules**: Edit `data/healthcare_guidelines.json` (terminology, formatting_rules, patient_summary_template). Restart the app; to reseed the Vector DB, delete the `data/chroma_db` directory and run again.
- **Prompts**: Adjust `src/prompts.py` (section order, wording, new content types).
- **Model**: Set `LLM_MODEL` in `.env` (e.g. `gpt-4o`, `gpt-4o-mini`).

## License

Use for educational and internal healthcare workflows. Ensure compliance with your organization’s policies and data handling requirements when processing real patient information.
>>>>>>> d34e27f (Initial commit project)
