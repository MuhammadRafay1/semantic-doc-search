# Data Directory

This directory is used to store your document datasets for semantic search.

## Supported File Formats

The system supports the following document formats:
- `.txt` - Plain text files
- `.pdf` - PDF documents
- `.docx` - Microsoft Word documents
- `.md` - Markdown files

## Usage

1. Create a subdirectory for your dataset (e.g., `data/my_dataset/`)
2. Place your documents in that directory
3. Use the GUI to select this directory
4. The system will automatically load and process all supported documents

## Example Structure

```
data/
├── academic_papers/
│   ├── paper1.pdf
│   ├── paper2.pdf
│   └── notes.txt
├── lecture_notes/
│   ├── week1.md
│   ├── week2.md
│   └── syllabus.docx
└── policies/
    ├── academic_integrity.pdf
    └── grading_policy.txt
```

## Notes

- No datasets are hard-coded in the application
- All document loading is dynamic based on user selection
- Minimum recommended: 10-15 documents for meaningful retrieval
