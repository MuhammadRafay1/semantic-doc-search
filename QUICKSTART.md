# 🚀 Quick Start Guide - AI Research Assistant

## Prerequisites
- Python 3.8 or higher
- pip package manager
- At least 2GB of free disk space (for embedding models)

## Installation (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirement.txt
```

**Note:** First-time installation may take 5-10 minutes depending on your internet speed.

### Step 2: Verify Installation
```bash
python -c "import langchain; import sentence_transformers; print('Installation successful!')"
```

## Running the Application

### Start the GUI
```bash
python -m app.main
```

The GUI window will open automatically.

## Using the Application (10 minutes)

### 1. Select Dataset
- Click **"Browse Folder"**
- Navigate to `data/sample_dataset` (or your own dataset)
- The system will display dataset statistics

### 2. Configure System
- **Embedding Model**: Select `all-mpnet-base-v2` (recommended for best quality)
  - First run will download ~420MB
  - Subsequent runs use cached model
- **Vector Database**: Select `FAISS` (faster) or `ChromaDB` (persistent)
- Click **"Build Index"**
  - Wait for "Index built!" message
  - Should take 1-3 minutes for sample dataset

### 3. Search
Try these sample queries:
- "What is machine learning?"
- "Explain neural networks"
- "How does reinforcement learning work?"
- "Applications of AI in healthcare"
- "What is explainable AI?"

Adjust **Top-K** (1-20) to see more/fewer results.

## Sample Dataset

The included `data/sample_dataset` contains 13 documents covering:
- AI Introduction
- Machine Learning Fundamentals
- Deep Learning and Neural Networks
- Natural Language Processing
- Computer Vision
- Reinforcement Learning
- Data Science
- AI Ethics
- Transfer Learning
- Generative AI and LLMs
- AI in Healthcare
- Explainable AI
- Edge AI and TinyML

## Testing Different Configurations

### Try Different Embedding Models:
1. `all-MiniLM-L6-v2` - Fast, lightweight
2. `all-mpnet-base-v2` - Best quality (recommended)
3. `multi-qa-MiniLM-L6-cos-v1` - Optimized for Q&A

### Compare Vector Stores:
- **FAISS**: In-memory, very fast searches
- **ChromaDB**: Persistent storage, survives application restart

## Automated Testing

For comprehensive testing:
```bash
cd experiments
python test_system.py
```

This will:
- Test 3 embedding models
- Test both vector databases
- Run standard queries
- Generate performance metrics

## Troubleshooting

**Issue: "ModuleNotFoundError"**
- Solution: Run `pip install -r requirement.txt` again

**Issue: Model download fails**
- Solution: Check internet connection, try again
- Models download from Hugging Face (requires internet first time only)

**Issue: "No documents loaded"**
- Solution: Ensure dataset directory contains .txt, .pdf, .docx, or .md files

**Issue: Application freezes during indexing**
- Solution: This is normal for first run (model download + indexing)
- Wait for progress bar to complete

**Issue: Out of memory**
- Solution: Try smaller embedding model (`all-MiniLM-L6-v2`)
- Reduce dataset size

## Next Steps

1. **Experiment**: Try different embedding models and queries
2. **Test**: Use your own documents (minimum 10-15 recommended)
3. **Document**: Fill out `experiments/report/report_template.md` with your observations
4. **Analyze**: Compare retrieval quality across configurations

## Performance Expectations

**First Run:**
- Model download: 2-10 minutes (depending  on model and internet)
- Index building: 1-3 minutes (for sample dataset)

**Subsequent Runs:**
- Index building: 30-90 seconds (models cached)
- Search queries: <1 second

## Tips for Best Results

1. **Use quality queries**: Be specific and clear
2. **Experiment with Top-K**: Try 3-10 for most use cases
3. **Larger datasets work better**: 20-50+ documents ideal
4. **Homogeneous content**: Documents on related topics retrieve better
5. **Document quality**: Well-written documents yield better results

## Support

- Check `README.md` for detailed documentation
- Review `experiments/report/report_template.md` for analysis guidelines
- Examine code comments for implementation details

---

**Ready to start?**
```bash
python -m app.main
```

Happy searching! 🔍
