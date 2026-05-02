"""
GUI Application for AI Research Assistant
Provides interface for dataset selection, embedding configuration, and semantic search.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
from typing import Optional

from app.config import (
    WINDOW_TITLE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    EMBEDDING_MODELS,
    VECTOR_STORES,
    DEFAULT_TOP_K,
    MAX_TOP_K
)
from app.utils import (
    DocumentLoader,
    TextProcessor,
    EmbeddingEngine,
    VectorStoreManager,
    create_semantic_search_system
)


class AIResearchAssistantGUI:
    """Main GUI application for AI Research Assistant."""
    
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        
        # Application state
        self.data_directory: Optional[Path] = None
        self.dataset_stats: Optional[dict] = None
        self.vector_manager: Optional[VectorStoreManager] = None
        self.selected_embedding_model: Optional[str] = None
        self.selected_vector_store: Optional[str] = None
        
        # Setup GUI
        self._setup_styles()
        self._create_widgets()
        
    def _setup_styles(self):
        """Configure ttk styles for better appearance."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'), foreground='#2c3e50')
        style.configure('Section.TLabel', font=('Arial', 11, 'bold'), foreground='#34495e')
        style.configure('Info.TLabel', font=('Arial', 9), foreground='#7f8c8d')
        style.configure('Action.TButton', font=('Arial', 10, 'bold'))
        
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container with padding
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)
        
        # Title
        title = ttk.Label(main_container, text="🔍 AI Research Assistant - Semantic Search", 
                         style='Title.TLabel')
        title.grid(row=0, column=0, pady=(0, 20), sticky=tk.W)
        
        # Create three main sections
        row_idx = 1
        
        # Section 1: Data Selection
        self._create_data_selection_section(main_container, row_idx)
        row_idx += 1
        
        # Separator
        ttk.Separator(main_container, orient='horizontal').grid(
            row=row_idx, column=0, sticky=(tk.W, tk.E), pady=15
        )
        row_idx += 1
        
        # Section 2: Configuration
        self._create_configuration_section(main_container, row_idx)
        row_idx += 1
        
        # Separator
        ttk.Separator(main_container, orient='horizontal').grid(
            row=row_idx, column=0, sticky=(tk.W, tk.E), pady=15
        )
        row_idx += 1
        
        # Section 3: Search
        self._create_search_section(main_container, row_idx)
        
        # Configure weights for proper expansion
        for i in range(row_idx + 1):
            main_container.rowconfigure(i, weight=0)
        main_container.rowconfigure(row_idx, weight=1)  # Search section gets extra space
        
    def _create_data_selection_section(self, parent, row):
        """Create data selection panel."""
        frame = ttk.LabelFrame(parent, text="📁 Step 1: Dataset Selection", padding="10")
        frame.grid(row=row, column=0, sticky=(tk.W, tk.E, tk.N), pady=5)
        frame.columnconfigure(1, weight=1)
        
        # Browse button
        ttk.Label(frame, text="Select Data Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        self.browse_btn = ttk.Button(button_frame, text="Browse Folder", 
                                     command=self._browse_directory, style='Action.TButton')
        self.browse_btn.pack(side=tk.LEFT, padx=5)
        
        self.dir_label = ttk.Label(button_frame, text="No directory selected", 
                                   foreground='gray', font=('Arial', 9))
        self.dir_label.pack(side=tk.LEFT, padx=10)
        
        # Dataset info
        ttk.Label(frame, text="Dataset Info:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.dataset_info_text = tk.Text(frame, height=4, width=80, state='disabled',
                                         bg='#ecf0f1', relief=tk.FLAT, font=('Courier', 9))
        self.dataset_info_text.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        
    def _create_configuration_section(self, parent, row):
        """Create embedding and vector store configuration panel."""
        frame = ttk.LabelFrame(parent, text="⚙️ Step 2: Configuration", padding="10")
        frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=5)
        frame.columnconfigure(1, weight=1)
        
        # Embedding model selection
        ttk.Label(frame, text="Embedding Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        model_frame = ttk.Frame(frame)
        model_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        self.embedding_var = tk.StringVar()
        self.embedding_combo = ttk.Combobox(model_frame, textvariable=self.embedding_var,
                                            state='readonly', width=40)
        self.embedding_combo['values'] = list(EMBEDDING_MODELS.keys())
        self.embedding_combo.current(0)
        self.embedding_combo.pack(side=tk.LEFT, padx=5)
        self.embedding_combo.bind('<<ComboboxSelected>>', self._on_model_selected)
        
        self.model_info_label = ttk.Label(model_frame, text="", style='Info.TLabel')
        self.model_info_label.pack(side=tk.LEFT, padx=10)
        self._update_model_info()
        
        # Vector store selection
        ttk.Label(frame, text="Vector Database:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        store_frame = ttk.Frame(frame)
        store_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        
        self.vector_store_var = tk.StringVar()
        self.vector_store_combo = ttk.Combobox(store_frame, textvariable=self.vector_store_var,
                                               state='readonly', width=40)
        self.vector_store_combo['values'] = list(VECTOR_STORES.keys())
        self.vector_store_combo.current(0)
        self.vector_store_combo.pack(side=tk.LEFT, padx=5)
        self.vector_store_combo.bind('<<ComboboxSelected>>', self._on_store_selected)
        
        self.store_info_label = ttk.Label(store_frame, text="", style='Info.TLabel')
        self.store_info_label.pack(side=tk.LEFT, padx=10)
        self._update_store_info()
        
        # Build index button
        build_frame = ttk.Frame(frame)
        build_frame.grid(row=2, column=0, columnspan=2, pady=15)
        
        self.build_btn = ttk.Button(build_frame, text="🔨 Build Index", 
                                    command=self._build_index, style='Action.TButton',
                                    state='disabled')
        self.build_btn.pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(build_frame, mode='indeterminate', length=300)
        self.progress.pack(side=tk.LEFT, padx=10)
        
        self.status_label = ttk.Label(build_frame, text="", foreground='green')
        self.status_label.pack(side=tk.LEFT, padx=10)
        
    def _create_search_section(self, parent, row):
        """Create semantic search panel."""
        frame = ttk.LabelFrame(parent, text="🔎 Step 3: Semantic Search", padding="10")
        frame.grid(row=row, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        
        # Query input
        query_frame = ttk.Frame(frame)
        query_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        query_frame.columnconfigure(1, weight=1)
        
        ttk.Label(query_frame, text="Query:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.query_entry = ttk.Entry(query_frame, width=60, font=('Arial', 10))
        self.query_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.query_entry.bind('<Return>', lambda e: self._perform_search())
        
        # Top-k selection
        ttk.Label(query_frame, text="Top-K:").grid(row=0, column=2, sticky=tk.W, padx=(20, 5))
        self.topk_var = tk.IntVar(value=DEFAULT_TOP_K)
        self.topk_spinner = ttk.Spinbox(query_frame, from_=1, to=MAX_TOP_K, 
                                       textvariable=self.topk_var, width=5)
        self.topk_spinner.grid(row=0, column=3, padx=5)
        
        self.search_btn = ttk.Button(query_frame, text="Search", 
                                     command=self._perform_search, 
                                     style='Action.TButton', state='disabled')
        self.search_btn.grid(row=0, column=4, padx=10)
        
        # Results display
        ttk.Label(frame, text="Results:", style='Section.TLabel').grid(
            row=1, column=0, sticky=tk.W, pady=(15, 5)
        )
        
        self.results_text = scrolledtext.ScrolledText(frame, height=15, width=100,
                                                      font=('Courier', 9), state='disabled')
        self.results_text.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Configure text tags for formatting
        self.results_text.tag_configure('header', foreground='#2c3e50', font=('Arial', 10, 'bold'))
        self.results_text.tag_configure('score', foreground='#27ae60', font=('Arial', 9, 'bold'))
        self.results_text.tag_configure('source', foreground='#3498db', font=('Arial', 9, 'italic'))
        self.results_text.tag_configure('content', foreground='#34495e', font=('Arial', 9))
        
    # Event handlers
    
    def _browse_directory(self):
        """Handle directory selection."""
        directory = filedialog.askdirectory(title="Select Dataset Directory")
        if directory:
            self.data_directory = Path(directory)
            self.dir_label.config(text=str(self.data_directory), foreground='black')
            
            # Load and display dataset info
            self._load_dataset_info()
            
            # Enable build button
            self.build_btn.config(state='normal')
            
    def _load_dataset_info(self):
        """Load and display dataset statistics."""
        if not self.data_directory:
            return
        
        try:
            _, stats = DocumentLoader.load_documents_from_directory(self.data_directory)
            self.dataset_stats = stats
            
            # Display info
            info_text = f"📊 Dataset Statistics:\n"
            info_text += f"  • Total Files: {stats['total_files']}\n"
            info_text += f"  • Loaded Successfully: {stats['loaded_files']}\n"
            info_text += f"  • Total Size: {stats['total_size_bytes'] / 1024:.2f} KB\n"
            info_text += f"  • File Types: {dict(stats['file_types'])}"
            
            self.dataset_info_text.config(state='normal')
            self.dataset_info_text.delete(1.0, tk.END)
            self.dataset_info_text.insert(1.0, info_text)
            self.dataset_info_text.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset info:\n{str(e)}")
    
    def _on_model_selected(self, event=None):
        """Handle embedding model selection."""
        self._update_model_info()
        
    def _update_model_info(self):
        """Update embedding model info label."""
        model_key = self.embedding_var.get()
        if model_key and model_key in EMBEDDING_MODELS:
            info = EMBEDDING_MODELS[model_key]
            self.model_info_label.config(
                text=f"({info['description']}, {info['dimension']} dim)"
            )
    
    def _on_store_selected(self, event=None):
        """Handle vector store selection."""
        self._update_store_info()
        
    def _update_store_info(self):
        """Update vector store info label."""
        store_key = self.vector_store_var.get()
        if store_key and store_key in VECTOR_STORES:
            self.store_info_label.config(text=f"({VECTOR_STORES[store_key]})")
    
    def _build_index(self):
        """Build vector store index in background thread."""
        if not self.data_directory:
            messagebox.showwarning("Warning", "Please select a dataset directory first!")
            return
        
        # Disable buttons during processing
        self.build_btn.config(state='disabled')
        self.search_btn.config(state='disabled')
        self.progress.start(10)
        self.status_label.config(text="Building index...", foreground='orange')
        
        # Run in background thread
        threading.Thread(target=self._build_index_thread, daemon=True).start()
    
    def _build_index_thread(self):
        """Background thread for building index."""
        try:
            self.selected_embedding_model = self.embedding_var.get()
            self.selected_vector_store = self.vector_store_var.get()
            
            # Create semantic search system
            self.vector_manager, stats = create_semantic_search_system(
                self.data_directory,
                self.selected_embedding_model,
                self.selected_vector_store
            )
            
            # Update UI on main thread
            self.root.after(0, lambda: self._build_complete(stats))
            
        except Exception as e:
            self.root.after(0, lambda: self._build_failed(str(e)))
    
    def _build_complete(self, stats):
        """Handle successful index build."""
        self.progress.stop()
        self.status_label.config(
            text=f"✓ Index built! ({stats['total_chunks']} chunks)", 
            foreground='green'
        )
        self.search_btn.config(state='normal')
        self.build_btn.config(state='normal')
        
        messagebox.showinfo("Success", 
                           f"Index built successfully!\n\n"
                           f"Documents: {stats['loaded_files']}\n"
                           f"Chunks: {stats['total_chunks']}")
    
    def _build_failed(self, error_msg):
        """Handle failed index build."""
        self.progress.stop()
        self.status_label.config(text="✗ Build failed", foreground='red')
        self.build_btn.config(state='normal')
        messagebox.showerror("Error", f"Failed to build index:\n{error_msg}")
    
    def _perform_search(self):
        """Perform semantic search."""
        query = self.query_entry.get().strip()
        
        if not query:
            messagebox.showwarning("Warning", "Please enter a search query!")
            return
        
        if not self.vector_manager:
            messagebox.showwarning("Warning", "Please build the index first!")
            return
        
        try:
            k = self.topk_var.get()
            results = self.vector_manager.similarity_search(query, k=k)
            
            self._display_results(query, results)
            
        except Exception as e:
            messagebox.showerror("Error", f"Search failed:\n{str(e)}")
    
    def _display_results(self, query, results):
        """Display search results in the text widget."""
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        
        # Header
        self.results_text.insert(tk.END, f"Query: \"{query}\"\n", 'header')
        self.results_text.insert(tk.END, f"Found {len(results)} results\n\n", 'header')
        self.results_text.insert(tk.END, "=" * 100 + "\n\n", 'header')
        
        # Results
        for idx, (doc, score) in enumerate(results, 1):
            # Result header
            self.results_text.insert(tk.END, f"[{idx}] ", 'header')
            self.results_text.insert(tk.END, f"Relevance Score: {score:.4f}\n", 'score')
            
            # Source
            source = doc.metadata.get('filename', doc.metadata.get('source', 'Unknown'))
            self.results_text.insert(tk.END, f"Source: {source}\n\n", 'source')
            
            # Content preview (first 300 characters)
            content = doc.page_content[:300]
            if len(doc.page_content) > 300:
                content += "..."
            self.results_text.insert(tk.END, f"{content}\n", 'content')
            
            self.results_text.insert(tk.END, "\n" + "-" * 100 + "\n\n", 'content')
        
        self.results_text.config(state='disabled')
        self.results_text.see(1.0)  # Scroll to top


def main():
    """Main entry point for GUI application."""
    root = tk.Tk()
    app = AIResearchAssistantGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
