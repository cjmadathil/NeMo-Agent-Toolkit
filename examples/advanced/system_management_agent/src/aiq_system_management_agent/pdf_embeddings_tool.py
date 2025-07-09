# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import PyPDF2
from langchain_core.tools import tool
from pydantic import Field
from sentence_transformers import SentenceTransformer

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.profiler.decorators.function_tracking import track_function

logger = logging.getLogger(__name__)


class PDFEmbeddingsToolConfig(FunctionBaseConfig, name="pdf_embeddings_tool"):
    """Configuration for the PDF Embeddings Tool."""
    
    pdf_directory: str = Field(
        default="./data/pdfs",
        description="Directory containing PDF files to process"
    )
    embeddings_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    chunk_size: int = Field(
        default=1000,
        description="Size of text chunks for processing"
    )
    chunk_overlap: int = Field(
        default=100,
        description="Overlap between chunks"
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return"
    )
    similarity_threshold: float = Field(
        default=0.5,
        description="Minimum similarity threshold for results"
    )
    cache_embeddings: bool = Field(
        default=True,
        description="Whether to cache embeddings for faster retrieval"
    )
    cache_directory: str = Field(
        default="./data/embeddings_cache",
        description="Directory to store cached embeddings"
    )


@register_function(config_type=PDFEmbeddingsToolConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def pdf_embeddings_tool(config: PDFEmbeddingsToolConfig, builder: Builder):
    """
    PDF embeddings tool that can load PDFs, create embeddings, and search for relevant content.
    """
    
    # Initialize the sentence transformer model
    model = SentenceTransformer(config.embeddings_model)
    
    # Initialize storage for embeddings and metadata
    embeddings_index = None
    chunk_metadata = []
    
    @track_function()
    def _extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1} of {pdf_path}: {str(e)}")
                        continue
            return text
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
            return f"Error reading PDF {pdf_path}: {str(e)}"
    
    @track_function()
    def _chunk_text(text: str, pdf_path: str) -> List[Dict]:
        """Split text into chunks with metadata."""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), config.chunk_size - config.chunk_overlap):
            chunk_words = words[i:i + config.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "source": pdf_path,
                "chunk_index": len(chunks),
                "word_count": len(chunk_words)
            })
        
        return chunks
    
    @track_function()
    def _load_cached_embeddings() -> Optional[tuple]:
        """Load cached embeddings if available."""
        cache_dir = Path(config.cache_directory)
        if not cache_dir.exists():
            return None
        
        index_path = cache_dir / "embeddings.index"
        metadata_path = cache_dir / "metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            try:
                index = faiss.read_index(str(index_path))
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                logger.info(f"Loaded cached embeddings with {index.ntotal} vectors")
                return index, metadata
            except Exception as e:
                logger.warning(f"Error loading cached embeddings: {str(e)}")
        
        return None
    
    @track_function()
    def _save_embeddings_cache(index, metadata):
        """Save embeddings to cache."""
        if not config.cache_embeddings:
            return
        
        try:
            cache_dir = Path(config.cache_directory)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            index_path = cache_dir / "embeddings.index"
            metadata_path = cache_dir / "metadata.pkl"
            
            faiss.write_index(index, str(index_path))
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved embeddings cache with {index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error saving embeddings cache: {str(e)}")
    
    @track_function()
    def _process_pdfs():
        """Process all PDF files and create embeddings."""
        nonlocal embeddings_index, chunk_metadata
        
        # Try to load cached embeddings first
        if config.cache_embeddings:
            cached_data = _load_cached_embeddings()
            if cached_data:
                embeddings_index, chunk_metadata = cached_data
                return
        
        pdf_dir = Path(config.pdf_directory)
        if not pdf_dir.exists():
            logger.warning(f"PDF directory {pdf_dir} does not exist")
            return
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return
        
        logger.info(f"Processing {len(pdf_files)} PDF files...")
        
        all_chunks = []
        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name}...")
            text = _extract_text_from_pdf(str(pdf_file))
            if text and not text.startswith("Error"):
                chunks = _chunk_text(text, str(pdf_file))
                all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("No text chunks extracted from PDFs")
            return
        
        # Create embeddings
        logger.info(f"Creating embeddings for {len(all_chunks)} chunks...")
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        embeddings_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        embeddings_index.add(embeddings)
        
        chunk_metadata = all_chunks
        
        # Save to cache
        _save_embeddings_cache(embeddings_index, chunk_metadata)
        
        logger.info(f"Created embeddings index with {embeddings_index.ntotal} vectors")
    
    @track_function()
    def _search_embeddings(query: str, max_results: int = None) -> List[Dict]:
        """Search for relevant content using embeddings."""
        if embeddings_index is None or not chunk_metadata:
            return []
        
        max_results = max_results or config.max_results
        
        # Create query embedding
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = embeddings_index.search(query_embedding, max_results)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if score >= config.similarity_threshold:
                metadata = chunk_metadata[idx].copy()
                metadata["similarity_score"] = float(score)
                metadata["rank"] = i + 1
                results.append(metadata)
        
        return results
    
    @tool
    async def load_pdf_embeddings() -> str:
        """
        Load and process all PDF files in the configured directory to create embeddings.
        
        Returns:
            Status message about the loading process
        """
        try:
            _process_pdfs()
            if embeddings_index is None:
                return "No PDF files found or processed successfully"
            return f"Successfully loaded embeddings for {embeddings_index.ntotal} text chunks from PDF files"
        except Exception as e:
            logger.error(f"Error loading PDF embeddings: {str(e)}")
            return f"Error loading PDF embeddings: {str(e)}"
    
    @tool
    async def search_pdf_content(query: str, max_results: int = 5) -> str:
        """
        Search for relevant content in loaded PDF documents.
        
        Args:
            query: Search query to find relevant content
            max_results: Maximum number of results to return (default: 5)
            
        Returns:
            Relevant content from PDF documents
        """
        try:
            if embeddings_index is None:
                return "No PDF embeddings loaded. Please run load_pdf_embeddings first."
            
            results = _search_embeddings(query, max_results)
            
            if not results:
                return f"No relevant content found for query: {query}"
            
            formatted_results = []
            for result in results:
                formatted_results.append(
                    f"**Source:** {os.path.basename(result['source'])}\n"
                    f"**Similarity:** {result['similarity_score']:.3f}\n"
                    f"**Content:** {result['text'][:500]}{'...' if len(result['text']) > 500 else ''}\n"
                )
            
            return f"Found {len(results)} relevant results for '{query}':\n\n" + "\n---\n".join(formatted_results)
        
        except Exception as e:
            logger.error(f"Error searching PDF content: {str(e)}")
            return f"Error searching PDF content: {str(e)}"
    
    @tool
    async def list_loaded_pdfs() -> str:
        """
        List all loaded PDF files and their statistics.
        
        Returns:
            Information about loaded PDF files
        """
        try:
            if not chunk_metadata:
                return "No PDF files loaded"
            
            pdf_stats = {}
            for chunk in chunk_metadata:
                source = chunk["source"]
                if source not in pdf_stats:
                    pdf_stats[source] = {"chunks": 0, "total_words": 0}
                pdf_stats[source]["chunks"] += 1
                pdf_stats[source]["total_words"] += chunk["word_count"]
            
            result = f"Loaded {len(pdf_stats)} PDF files with {len(chunk_metadata)} total chunks:\n\n"
            for pdf_path, stats in pdf_stats.items():
                filename = os.path.basename(pdf_path)
                result += f"- {filename}: {stats['chunks']} chunks, ~{stats['total_words']} words\n"
            
            return result
        
        except Exception as e:
            logger.error(f"Error listing loaded PDFs: {str(e)}")
            return f"Error listing loaded PDFs: {str(e)}"
    
    # Initialize embeddings on startup
    try:
        _process_pdfs()
    except Exception as e:
        logger.error(f"Error initializing PDF embeddings: {str(e)}")
    
    # Return the tools
    yield [load_pdf_embeddings, search_pdf_content, list_loaded_pdfs]