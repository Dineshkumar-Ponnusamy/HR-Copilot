# HR Policy Copilot

An intelligent HR policy assistant that uses Retrieval-Augmented Generation (RAG) to provide role-based answers from your organization's policy documents. Built with Streamlit, Groq API, and FAISS for fast, accurate policy lookups.

## Features

- **Role-based Access Control**: Different permission levels for Employees and Managers
- **Intelligent Document Search**: FAISS-powered vector search with fallback to brute-force similarity
- **Multi-format Support**: Processes PDF and TXT policy documents
- **Query Logging**: Comprehensive CSV logging of all queries for audit trails
- **Confidence Thresholding**: Configurable similarity thresholds to ensure answer quality
- **Local Document Processing**: All documents processed and indexed locally for privacy

## Architecture

The system uses a RAG (Retrieval-Augmented Generation) approach:

1. **Document Ingestion**: PDF/TXT policies are chunked and embedded using SentenceTransformers
2. **Vector Indexing**: FAISS index enables fast similarity search (with numpy fallback)
3. **Role-based Retrieval**: Queries filtered by user role permissions before retrieval
4. **LLM Enhancement**: Groq API provides natural language responses based on retrieved context
5. **Logging**: All queries logged with metadata for compliance and analytics

## Prerequisites

- Python 3.8+
- Groq API key (get one at [groq.com](https://groq.com))
- Policy documents in PDF or TXT format

## Setup

### 1) Environment Setup

```bash
cd hr-copilot

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
```

### 2) Install Dependencies

```bash
pip install -r requirements.txt
```

### 3) (Optional) Generate Sample Policies

If you don't have policy documents yet, generate sample PDFs:

```bash
python generate_sample_policies.py
```

This creates sample policies in the `data_policies/` directory:

- Code of Conduct
- Leave Policy
- Travel Expense Policy

### 4) Build Vector Index

Process your policy documents into searchable vectors:

```bash
python build_index.py
```

This creates:

- `index/embeddings.npy`: Document embeddings
- `index/texts.json`: Document chunks
- `index/meta.json`: Metadata (source, page numbers)
- `index/faiss.index`: FAISS index (if available)

### 5) Configure API Key

Set your Groq API key:

```bash
export GROQ_API_KEY="your-api-key-here"
# Windows PowerShell: $env:GROQ_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```
GROQ_API_KEY=your-api-key-here
```

### 6) Launch Application

```bash
streamlit run app_streamlit.py
```

The application will be available at `http://localhost:8501`

## Usage

### User Roles

- **Employee**: Access to general policies (Leave, Code of Conduct, etc.)
- **Manager**: Access to all employee policies plus management-specific content

### Query Interface

1. Enter your Groq API key (if not set via environment variable)
2. Select your role (Employee/Manager)
3. Choose a Groq model (Llama variants available)
4. Adjust retrieval settings:
   - Top-K passages: Number of relevant chunks to retrieve (3-8)
   - Refusal threshold: Minimum similarity score required (0.0-1.0)
5. Ask questions in natural language

### Example Queries

```
"What is the parental leave policy?"
"How do I report harassment?"
"What are the travel expense limits?"
"How many sick days do I get?"
```

## Configuration

### Role-based Access (Optional)

Create a `roles.json` file to customize access permissions:

```json
{
  "default_allow": ["Employee", "Manager"],
  "file_rules": {
    "People_Leader_Guide.pdf": ["Manager"],
    "Leave_Policy.pdf": ["Employee", "Manager"]
  }
}
```

## Troubleshooting

### Common Issues

**"Couldn't load index" error**:

- Run `python build_index.py` to create the vector index
- Ensure policy documents exist in `data_policies/` directory

**"No FAISS index" warning**:

- FAISS not installed or failed to build
- Application will use slower but functional numpy-based search

**Low-quality answers**:

- Adjust the "Refusal threshold" slider higher (e.g., 0.3-0.5)
- Increase "Top-K passages" for more context

**Import errors**:

- Ensure all requirements are installed: `pip install -r requirements.txt`
- Try reinstalling specific packages: `pip install --upgrade package-name`

### Logs

Query logs are saved to `logs/query_log.csv` with:

- Timestamp, model used, user role
- Query text, retrieved sources, similarity scores
- Response latency

## Project Structure

```
hr-copilot/
├── app_streamlit.py          # Main Streamlit application
├── build_index.py            # Vector index builder
├── generate_sample_policies.py # Sample document generator
├── requirements.txt          # Python dependencies
├── data_policies/            # Policy documents (PDFs/TXTs)
├── index/                    # Vector index and metadata
├── logs/                     # Query logs
└── faiss_index/              # FAISS index files
```

## Dependencies

- **streamlit**: Web interface
- **groq**: LLM API client
- **faiss-cpu**: Vector similarity search
- **sentence-transformers**: Text embeddings
- **pypdf**: PDF processing
- **python-dotenv**: Environment variable management
- **langchain/langchain-community**: Additional utilities
- **numpy**: Numerical computations
- **reportlab**: PDF generation (for samples)
