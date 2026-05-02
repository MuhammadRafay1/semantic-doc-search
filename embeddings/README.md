# Embeddings Directory

This directory stores cached embedding models downloaded from Hugging Face.

## Purpose

When you first select an embedding model in the GUI, the model files will be downloaded and cached here. This prevents re-downloading on subsequent uses.

## Contents

After running the application, you may see directories like:
- `models--sentence-transformers--all-MiniLM-L6-v2/`
- `models--sentence-transformers--all-mpnet-base-v2/`
- etc.

## Notes

- This directory is managed automatically by the HuggingFace `transformers` library
- You can safely delete these files to free up space, but models will need to be re-downloaded
- Each model is typically 100-500 MB
