"""
Build Index Script for APIMatic Doc Search
Processes markdown documentation and builds a persistent FAISS index.

Usage:
    python build_index.py                          # Build from default data directory
    python build_index.py --data-dir ./data/docs   # Build from custom directory
    python build_index.py --clone                  # Clone repo first, then build
"""

import argparse
import subprocess
import sys
import time
import logging
from pathlib import Path

from app.config import (
    DATA_DIR,
    APIMATIC_DOCS_REPO,
    APIMATIC_DOCS_DIR,
    FAISS_INDEX_PATH,
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL_NAME,
)
from app.utils import (
    DocumentLoader,
    TextProcessor,
    EmbeddingEngine,
    VectorStoreManager,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("build_index.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def clone_docs_repo(repo_url: str, target_dir: Path, branch: str = None) -> bool:
    """
    Clone or update the APIMatic docs repository.

    Args:
        repo_url: Git repository URL
        target_dir: Directory to clone into
        branch: Optional branch name

    Returns:
        True if successful
    """
    if target_dir.exists() and (target_dir / ".git").exists():
        logger.info(f"📦 Updating existing repo at {target_dir}...")
        try:
            subprocess.run(
                ["git", "-C", str(target_dir), "pull", "--ff-only"],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("✅ Repository updated successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Git pull failed: {e.stderr}")
            return False
    else:
        logger.info(f"📥 Cloning {repo_url} into {target_dir}...")
        cmd = ["git", "clone"]
        if branch:
            cmd.extend(["--branch", branch])
        cmd.extend(["--depth", "1", repo_url, str(target_dir)])

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✅ Repository cloned successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Git clone failed: {e.stderr}")
            return False


def build_index(data_dir: Path, save_path: Path = None) -> dict:
    """
    Build FAISS index from documents in the data directory.

    Args:
        data_dir: Directory containing documents
        save_path: Path to save the FAISS index

    Returns:
        Statistics dictionary
    """
    if save_path is None:
        save_path = FAISS_INDEX_PATH

    total_start = time.time()

    # Step 1: Load documents
    logger.info(f"\n{'='*60}")
    logger.info(f"📂 Loading documents from: {data_dir}")
    logger.info(f"{'='*60}")

    load_start = time.time()
    documents, stats = DocumentLoader.load_documents_from_directory(data_dir)
    load_time = time.time() - load_start

    if not documents:
        logger.error("❌ No documents found! Check the data directory.")
        return {"error": "No documents found"}

    logger.info(f"  ✅ Loaded {stats['loaded_files']}/{stats['total_files']} documents")
    logger.info(f"  📊 File types: {stats['file_types']}")
    logger.info(f"  💾 Total size: {stats['total_size_bytes'] / 1024:.1f} KB")
    logger.info(f"  ⏱️ Load time: {load_time:.2f}s")

    # Step 2: Chunk documents
    logger.info(f"\n{'='*60}")
    logger.info(f"✂️ Chunking documents...")
    logger.info(f"{'='*60}")

    chunk_start = time.time()
    processor = TextProcessor()
    chunks = processor.split_documents(documents)
    chunk_time = time.time() - chunk_start

    logger.info(f"  ✅ Created {len(chunks)} chunks")
    logger.info(f"  ⏱️ Chunk time: {chunk_time:.2f}s")

    # Step 3: Generate embeddings & build FAISS index
    logger.info(f"\n{'='*60}")
    logger.info(f"🧠 Generating embeddings with {EMBEDDING_MODEL_NAME}...")
    logger.info(f"{'='*60}")

    embed_start = time.time()
    engine = EmbeddingEngine.get_instance()
    vector_manager = VectorStoreManager("FAISS", engine)
    vector_manager.create_vector_store(chunks)
    embed_time = time.time() - embed_start

    logger.info(f"  ✅ Embeddings generated and indexed")
    logger.info(f"  ⏱️ Embedding time: {embed_time:.2f}s")

    # Step 4: Save index
    logger.info(f"\n{'='*60}")
    logger.info(f"💾 Saving FAISS index to: {save_path}")
    logger.info(f"{'='*60}")

    save_start = time.time()
    vector_manager.vector_store.save_local(str(save_path))
    save_time = time.time() - save_start

    total_time = time.time() - total_start

    # Final stats
    stats.update({
        "total_chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "index_path": str(save_path),
        "load_time_s": round(load_time, 2),
        "chunk_time_s": round(chunk_time, 2),
        "embed_time_s": round(embed_time, 2),
        "save_time_s": round(save_time, 2),
        "total_time_s": round(total_time, 2),
    })

    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 Index built successfully!")
    logger.info(f"{'='*60}")
    logger.info(f"  📄 Documents: {stats['loaded_files']}")
    logger.info(f"  🧩 Chunks: {stats['total_chunks']}")
    logger.info(f"  ⏱️ Total time: {total_time:.2f}s")
    logger.info(f"  📍 Saved to: {save_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for APIMatic Doc Search")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing documents (default: data/apimatic-docs or data/)",
    )
    parser.add_argument(
        "--clone",
        action="store_true",
        help="Clone/update the APIMatic docs repo before building",
    )
    parser.add_argument(
        "--repo-url",
        type=str,
        default=APIMATIC_DOCS_REPO,
        help="Git repository URL to clone",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Git branch to clone",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the FAISS index",
    )

    args = parser.parse_args()

    # Clone repo if requested
    if args.clone:
        success = clone_docs_repo(args.repo_url, APIMATIC_DOCS_DIR, args.branch)
        if not success:
            logger.error("Failed to clone repository. Exiting.")
            sys.exit(1)

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif APIMATIC_DOCS_DIR.exists():
        data_dir = APIMATIC_DOCS_DIR
    else:
        data_dir = DATA_DIR
        logger.info(f"Using default data directory: {data_dir}")

    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)

    # Build index
    output_path = Path(args.output) if args.output else FAISS_INDEX_PATH
    stats = build_index(data_dir, output_path)

    if "error" in stats:
        sys.exit(1)


if __name__ == "__main__":
    main()
