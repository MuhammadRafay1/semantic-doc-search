# Vector Store Directory

This directory stores the generated vector databases for quick retrieval.

## Purpose

After building an index through the GUI, the vector store is saved here. This allows you to:
- Reload indexes without re-processing documents
- Switch between different datasets
- Persist your work between sessions

## Contents

Depending on your vector database choice:

### FAISS
- Creates directories like `dataset_name_faiss/`
- Contains `.faiss` index files and `.pkl` metadata

### ChromaDB
- Creates a `chroma_db/` directory
- Contains SQLite database and parquet files

## Notes

- Vector stores are specific to both the embedding model and the dataset
- If you change embedding models, you'll need to rebuild the index
- These files can be large (100MB - 1GB+) depending on dataset size
