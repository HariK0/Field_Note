# Fieldnote — Local AI PDF Study Assistant

Learning examples for document ingestion, chunking, retrieval, and answer generation, plus a local PDF notes RAG app.

## Run the local notes app

The app extracts text from PDF pages, previews configurable chunks, stores local embeddings in Chroma, and answers questions with cited passages. PDF text, embeddings, and answers stay on this computer; the app does not need an OpenAI API key.

1. Install [Ollama for Windows](https://ollama.com/download/windows).
2. In PowerShell, download the local chat and embedding models:

   ```powershell
   ollama pull qwen3:1.7b
   ollama pull nomic-embed-text
   ```

   The Qwen model download is about 1.4 GB and the Nomic embedding model is about 274 MB. They run locally; CPU inference may be slower than a hosted model.
3. Use Python 3.10 or newer and create/activate a virtual environment.
4. Install dependencies: `pip install -r requirements.txt`
5. Start the app from this folder: `streamlit run app.py`
6. Upload PDFs in the sidebar, adjust chunk size/overlap, preview chunks, and select **Index PDFs**.

The app uses `http://localhost:11434` for Ollama by default. The Chroma database is stored in `db/local_notes_chroma/`. Re-uploading the same PDF does not add duplicate chunks. Use **Clear collection** to reset the indexed PDFs. Scanned PDFs without selectable text need OCR before upload.

## Example scripts

The numbered scripts and notebooks demonstrate additional RAG techniques. Most expect `OPENAI_API_KEY` and a database at `db/chroma_db`; run `1_ingestion_pipeline.py` before the retrieval scripts. The notebooks may need extra system tools or API credentials described in their cells.

## Project credit

**Author:** C Hari Kiran  

