# 🤖 RAG Chat Application with NVIDIA NIMs

A modern, Retrieval Augmented Generation (RAG) chat application built with NVIDIA NIMs, Streamlit, and FAISS.

## ✨ Features

- 📄 **Multi-format Support**: PDF, DOCX, TXT, CSV, XLSX
- 🔍 **FAISS Vector Search**: Fast similarity search
- 🤖 **NVIDIA NIMs**: State-of-the-art embeddings and LLM

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: NVIDIA NIM (Llama 3.1)
- **Embeddings**: NVIDIA NV-Embed-QA
- **Vector DB**: FAISS
- **Package Manager**: UV
- **Containerization**: Docker
- **CI/CD**: GitHub Actions

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- UV package manager
- NVIDIA API key (free from [build.nvidia.com](https://build.nvidia.com))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-chat-nvidia.git
cd rag-chat-nvidia
```

2. Install UV:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create `.env` file:
```bash
echo "NVIDIA_API_KEY=your_key_here" > .env
```

4. Install dependencies:
```bash
uv sync
```

5. Run the application:
```bash
uv run streamlit run src/app.py
```

### Using Docker
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## 📖 Usage

1. **Upload Documents**: Use the sidebar to upload PDFs, DOCX, or other supported files
2. **Wait for Processing**: The app will extract text and create embeddings
3. **Ask Questions**: Type questions in the chat interface
4. **View Sources**: Expand source citations to see relevant document chunks

## 🔧 Configuration

### Environment Variables

- `NVIDIA_API_KEY`: Your NVIDIA API key (required)

### Customization

- **Chunk Size**: Modify in `DocumentProcessor` (default: 1000)
- **Model**: Change in `RAGChain` and `VectorStore`
- **UI Theme**: Edit CSS in `app.py`

## 📊 Performance

- **Embedding Model**: NV-Embed-QA (1024 dimensions)
- **LLM**: Llama 3.1 8B Instruct
- **Vector Search**: FAISS (L2 distance)
- **Processing Speed**: ~1-2 seconds per page
