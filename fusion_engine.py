"""
GestaltView Fusion Engine - Multi-Modal Processing
Keith Soyka's Consciousness-Serving AI Platform
Copyright © 2025 Keith Soyka. All Rights Reserved.

Processes and fuses text, image (OCR), and audio (transcription) 
into unified understanding for consciousness-serving AI.
"""

import os
import json
import logging
import base64
import io
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

# Image processing
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("PIL/pytesseract not available: OCR functionality disabled.")

# Audio processing
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("whisper not available: Audio transcription disabled.")

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available: Using TF-IDF fallback.")

from sklearn.feature_extraction.text import TfidfVectorizer


class FusionEngine:
    """
    Multi-Modal Fusion Engine for GestaltView
    
    Combines text, image (OCR), and audio (transcription) inputs
    into unified semantic understanding for consciousness-serving AI.
    
    Features:
    - Graceful degradation when optional dependencies unavailable
    - Semantic embeddings with multiple backend options
    - Memory-efficient processing
    - Error handling and logging
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the fusion engine with specified embedding model"""
        
        self.embedding_model = None
        self.whisper_model = None
        self.vectorizer = TfidfVectorizer(max_features=1024, stop_words='english')
        self._tfidf_fitted = False
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(model_name)
                logging.info(f"✅ Loaded SentenceTransformer model: {model_name}")
            except Exception as e:
                logging.warning(f"Failed to load SentenceTransformer: {e}")
                self.embedding_model = None
        
        # Initialize Whisper model (lazy loading)
        if WHISPER_AVAILABLE:
            logging.info("🎤 Whisper available for audio transcription")
        
        logging.info("🔗 FusionEngine initialized - Ready for multi-modal processing")
    
    def process_text(self, text: str) -> str:
        """Process and clean text input"""
        if not text:
            return ""
        
        # Basic text cleaning and normalization
        cleaned = text.strip()
        
        # Remove excessive whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Ensure reasonable length (truncate if necessary)
        if len(cleaned) > 10000:
            cleaned = cleaned[:10000] + "... [truncated]"
            logging.warning("Text input truncated to 10,000 characters")
        
        return cleaned
    
    def process_image_b64(self, b64_image: str) -> str:
        """Process base64 encoded image and extract text via OCR"""
        if not OCR_AVAILABLE or not b64_image:
            logging.warning("OCR not available or no image provided")
            return ""
        
        try:
            # Handle data URL format
            if ',' in b64_image:
                b64_image = b64_image.split(',')[1]
            
            # Decode base64 image
            image_data = base64.b64decode(b64_image)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text using Tesseract OCR
            extracted_text = pytesseract.image_to_string(image).strip()
            
            if extracted_text:
                logging.info(f"🖼️ OCR extracted {len(extracted_text)} characters from image")
                return extracted_text
            else:
                logging.info("🖼️ No text found in image")
                return ""
                
        except Exception as e:
            logging.error(f"Image OCR processing failed: {e}")
            return ""
    
    def process_image_file(self, image_path: str) -> str:
        """Process image file and extract text via OCR"""
        if not OCR_AVAILABLE or not image_path:
            return ""
        
        try:
            if not os.path.exists(image_path):
                logging.error(f"Image file not found: {image_path}")
                return ""
            
            # Load and process image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            extracted_text = pytesseract.image_to_string(image).strip()
            
            if extracted_text:
                logging.info(f"🖼️ OCR extracted {len(extracted_text)} characters from {image_path}")
                return extracted_text
            else:
                logging.info(f"🖼️ No text found in {image_path}")
                return ""
                
        except Exception as e:
            logging.error(f"Image file OCR processing failed: {e}")
            return ""
    
    def process_audio_file(self, audio_path: str) -> str:
        """Process audio file and extract text via transcription"""
        if not WHISPER_AVAILABLE or not audio_path:
            return ""
        
        try:
            if not os.path.exists(audio_path):
                logging.error(f"Audio file not found: {audio_path}")
                return ""
            
            # Lazy load Whisper model
            if not self.whisper_model:
                logging.info("Loading Whisper model (this may take a moment)...")
                self.whisper_model = whisper.load_model('tiny')  # Fastest model
                logging.info("✅ Whisper model loaded")
            
            # Transcribe audio
            result = self.whisper_model.transcribe(audio_path, fp16=False)
            transcribed_text = result.get('text', '').strip()
            
            if transcribed_text:
                logging.info(f"🎤 Transcribed {len(transcribed_text)} characters from {audio_path}")
                return transcribed_text
            else:
                logging.info(f"🎤 No speech detected in {audio_path}")
                return ""
                
        except Exception as e:
            logging.error(f"Audio transcription failed: {e}")
            return ""
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text list using best available method"""
        if not texts or all(not text.strip() for text in texts):
            return np.array([])
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        
        if not valid_texts:
            return np.array([])
        
        # Try SentenceTransformers first
        if self.embedding_model:
            try:
                embeddings = self.embedding_model.encode(
                    valid_texts, 
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                logging.debug(f"Generated embeddings using SentenceTransformers: {embeddings.shape}")
                return embeddings
                
            except Exception as e:
                logging.warning(f"SentenceTransformers embedding failed, falling back to TF-IDF: {e}")
        
        # Fallback to TF-IDF
        try:
            # Fit vectorizer if not already fitted
            if not self._tfidf_fitted:
                # Use a larger corpus for better fitting if available
                fit_texts = valid_texts if len(valid_texts) > 10 else valid_texts * 5
                self.vectorizer.fit(fit_texts)
                self._tfidf_fitted = True
            
            embeddings = self.vectorizer.transform(valid_texts).toarray()
            logging.debug(f"Generated embeddings using TF-IDF: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logging.error(f"TF-IDF embedding failed: {e}")
            return np.array([])
    
    def fuse(self, 
             text: str = '', 
             image_b64: Optional[str] = None, 
             image_path: Optional[str] = None,
             audio_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Fuse multiple input modalities into unified understanding
        
        Args:
            text: Direct text input
            image_b64: Base64 encoded image
            image_path: Path to image file
            audio_path: Path to audio file
            
        Returns:
            Dict containing fused_text, embedding, and metadata
        """
        
        input_pieces = []
        metadata = {
            "modalities_processed": [],
            "total_characters": 0,
            "processing_errors": []
        }
        
        # Process text input
        if text:
            processed_text = self.process_text(text)
            if processed_text:
                input_pieces.append(f"Text Input:\n{processed_text}")
                metadata["modalities_processed"].append("text")
                metadata["total_characters"] += len(processed_text)
        
        # Process image via base64
        if image_b64:
            try:
                ocr_text = self.process_image_b64(image_b64)
                if ocr_text:
                    input_pieces.append(f"Image OCR:\n{ocr_text}")
                    metadata["modalities_processed"].append("image_b64")
                    metadata["total_characters"] += len(ocr_text)
            except Exception as e:
                metadata["processing_errors"].append(f"image_b64: {str(e)}")
        
        # Process image via file path
        if image_path:
            try:
                ocr_text = self.process_image_file(image_path)
                if ocr_text:
                    input_pieces.append(f"Image File OCR ({Path(image_path).name}):\n{ocr_text}")
                    metadata["modalities_processed"].append("image_file")
                    metadata["total_characters"] += len(ocr_text)
            except Exception as e:
                metadata["processing_errors"].append(f"image_file: {str(e)}")
        
        # Process audio
        if audio_path:
            try:
                transcribed_text = self.process_audio_file(audio_path)
                if transcribed_text:
                    input_pieces.append(f"Audio Transcription ({Path(audio_path).name}):\n{transcribed_text}")
                    metadata["modalities_processed"].append("audio")
                    metadata["total_characters"] += len(transcribed_text)
            except Exception as e:
                metadata["processing_errors"].append(f"audio: {str(e)}")
        
        # Combine all processed inputs
        if input_pieces:
            fused_text = "\n\n---\n\n".join(input_pieces)
        else:
            fused_text = ""
            logging.warning("No valid input found across all modalities")
        
        # Generate embeddings
        embedding = []
        if fused_text:
            try:
                embeddings = self._generate_embeddings([fused_text])
                if embeddings.size > 0:
                    embedding = embeddings[0].tolist()
                    metadata["embedding_dimension"] = len(embedding)
                    metadata["embedding_method"] = "sentence_transformers" if self.embedding_model else "tfidf"
            except Exception as e:
                logging.error(f"Embedding generation failed: {e}")
                metadata["processing_errors"].append(f"embedding: {str(e)}")
        
        # Log fusion results
        if metadata["modalities_processed"]:
            logging.info(f"🔗 Fusion complete: {', '.join(metadata['modalities_processed'])} "
                        f"→ {metadata['total_characters']} chars")
        
        return {
            "fused_text": fused_text,
            "embedding": embedding,
            "metadata": metadata,
            "success": bool(fused_text)
        }
    
    def batch_fuse(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple fusion requests efficiently"""
        results = []
        
        for i, input_data in enumerate(inputs):
            try:
                result = self.fuse(**input_data)
                result["batch_index"] = i
                results.append(result)
            except Exception as e:
                logging.error(f"Batch fusion failed for item {i}: {e}")
                results.append({
                    "fused_text": "",
                    "embedding": [],
                    "metadata": {"processing_errors": [str(e)]},
                    "success": False,
                    "batch_index": i
                })
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about available fusion capabilities"""
        return {
            "ocr_available": OCR_AVAILABLE,
            "whisper_available": WHISPER_AVAILABLE,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "embedding_model": self.embedding_model.get_sentence_embedding_dimension() if self.embedding_model else None,
            "whisper_model_loaded": self.whisper_model is not None,
            "tfidf_fitted": self._tfidf_fitted,
            "supported_modalities": [
                "text",
                "image_b64" if OCR_AVAILABLE else None,
                "image_file" if OCR_AVAILABLE else None,
                "audio" if WHISPER_AVAILABLE else None
            ]
        }


# Convenience function for single-use fusion
def quick_fuse(text: str = '', 
               image_b64: Optional[str] = None, 
               image_path: Optional[str] = None,
               audio_path: Optional[str] = None) -> Dict[str, Any]:
    """Quick fusion without persistent engine instance"""
    engine = FusionEngine()
    return engine.fuse(text=text, image_b64=image_b64, image_path=image_path, audio_path=audio_path)


# Module exports
__all__ = ['FusionEngine', 'quick_fuse']
