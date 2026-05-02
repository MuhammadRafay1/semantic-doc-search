"""
Test script for AI Research Assistant
Tests the system with different configurations and documents.
"""

import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils import create_semantic_search_system
from app.config import EMBEDDING_MODELS


def test_semantic_search(data_dir: str, models_to_test: list, vector_stores: list):
    """
    Test semantic search with different configurations.
    
    Args:
        data_dir: Path to test dataset
        models_to_test: List of embedding model keys to test
        vector_stores: List of vector store types to test
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "Explain neural networks",
        "What are the benefits of AI?"
    ]
    
    print("=" * 80)
    print("AI RESEARCH ASSISTANT - AUTOMATED TESTING")
    print("=" * 80)
    print()
    
    results = {}
    
    for model_key in models_to_test:
        for store_type in vector_stores:
            config_name = f"{model_key}_{store_type}"
            print(f"\n{'='*80}")
            print(f"Testing Configuration: {config_name}")
            print(f"Embedding Model: {EMBEDDING_MODELS[model_key]['name']}")
            print(f"Vector Store: {store_type}")
            print(f"{'='*80}\n")
            
            try:
                # Build index
                print(f"Building index...")
                start_time = time.time()
                
                vector_manager, stats = create_semantic_search_system(
                    data_path,
                    model_key,
                    store_type
                )
                
                build_time = time.time() - start_time
                
                print(f"✓ Index built in {build_time:.2f}s")
                print(f"  - Documents: {stats['loaded_files']}")
                print(f"  - Chunks: {stats['total_chunks']}")
                print()
                
                # Test queries
                results[config_name] = {
                    'build_time': build_time,
                    'stats': stats,
                    'queries': {}
                }
                
                for query in test_queries:
                    print(f"Query: \"{query}\"")
                    
                    search_start = time.time()
                    search_results = vector_manager.similarity_search(query, k=3)
                    search_time = time.time() - search_start
                    
                    print(f"  Search time: {search_time:.4f}s")
                    print(f"  Top result:")
                    if search_results:
                        doc, score = search_results[0]
                        print(f"    - Score: {score:.4f}")
                        print(f"    - Source: {doc.metadata.get('filename', 'Unknown')}")
                        print(f"    - Preview: {doc.page_content[:100]}...")
                    print()
                    
                    results[config_name]['queries'][query] = {
                        'search_time': search_time,
                        'top_score': search_results[0][1] if search_results else None,
                        'top_source': search_results[0][0].metadata.get('filename') if search_results else None
                    }
                
            except Exception as e:
                print(f"✗ Error testing {config_name}: {e}")
                results[config_name] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for config, data in results.items():
        if 'error' in data:
            print(f"\n{config}: FAILED - {data['error']}")
        else:
            print(f"\n{config}:")
            print(f"  Build Time: {data['build_time']:.2f}s")
            print(f"  Documents: {data['stats']['loaded_files']}")
            print(f"  Chunks: {data['stats']['total_chunks']}")
            print(f"  Average Search Time: {sum(q['search_time'] for q in data['queries'].values()) / len(data['queries']):.4f}s")
            avg_score = sum(q['top_score'] for q in data['queries'].values() if q['top_score']) / len(data['queries'])
            print(f"  Average Top Score: {avg_score:.4f}")


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "../data"  # Modify this to point to your test dataset
    
    # Models to test (test at least 3)
    MODELS = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "multi-qa-MiniLM-L6-cos-v1"
    ]
    
    # Vector stores to test
    STORES = ["FAISS", "ChromaDB"]
    
    print("Starting automated tests...")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models: {MODELS}")
    print(f"Vector Stores: {STORES}")
    print()
    
    test_semantic_search(DATA_DIR, MODELS, STORES)
    
    print("\n" + "=" * 80)
    print("Testing complete! Review results above for your report.")
    print("=" * 80)
