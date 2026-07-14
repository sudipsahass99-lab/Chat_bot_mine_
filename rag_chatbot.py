import os
import glob
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Tuple
import json

class IITKGPKnowledgeBase:
    def __init__(self, data_directory: str):
        self.data_directory = data_directory
        self.documents = []
        self.document_paths = []
        
    def load_documents(self):
        """Load all text files from the directory and subdirectories"""
        txt_files = glob.glob(os.path.join(self.data_directory, "**/*.txt"), recursive=True)
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read().strip()
                    # if content and len(content) > 50:  # Filter out very short files
                    self.documents.append(content)
                    self.document_paths.append(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
        
        print(f"Loaded {len(self.documents)} documents")
        return self.documents

class VectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.document_chunks = []
        self.chunk_metadata = []
        
    def chunk_documents(self, documents: List[str], chunk_size: int = 512, chunk_overlap: int = 50):
        """Split documents into smaller chunks for better retrieval"""
        for doc_idx, document in enumerate(documents):
            # Simple chunking by character length
            words = document.split()
            for i in range(0, len(words), chunk_size - chunk_overlap):
                chunk = " ".join(words[i:i + chunk_size])
                if len(chunk) > 100:  # Minimum chunk length
                    self.document_chunks.append(chunk)
                    self.chunk_metadata.append({
                        "doc_index": doc_idx,
                        "chunk_index": len(self.document_chunks) - 1
                    })
        
        print(f"Created {len(self.document_chunks)} chunks from {len(documents)} documents")
        return self.document_chunks
    
    def create_embeddings(self):
        """Create embeddings for all document chunks"""
        if not self.document_chunks:
            raise ValueError("No document chunks available. Call chunk_documents first.")
        
        print("Creating embeddings...")
        embeddings = self.embedding_model.encode(self.document_chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        
        print(f"Created FAISS index with {self.index.ntotal} vectors")
        return self.index
        
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar chunks"""
        if self.index is None:
            raise ValueError("Index not created. Call create_embeddings first.")
        
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # FAISS search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.document_chunks):
                results.append((self.document_chunks[idx], float(score)))
        
        return results

class IITKGPChatbot:
    def __init__(self, knowledge_base_dir: str):
        self.knowledge_base = IITKGPKnowledgeBase(knowledge_base_dir)
        self.vector_store = VectorStore()
        self.retriever = None
        
        # Initialize the LLM
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"Loading model {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # System prompt for IIT KGP chatbot
        self.system_prompt = """You are IIT KGP Senior, a helpful assistant for Indian Institute of Technology Kharagpur. 
You provide accurate information about IIT KGP based on the provided context. 
If you don't know something based on the context, honestly say you don't have that information.
Be precise, helpful, and maintain the professional tone of an educational institution."""

    def setup_retrieval_system(self):
        """Set up the complete retrieval system"""
        print("Loading documents...")
        documents = self.knowledge_base.load_documents()
        
        print("Chunking documents...")
        self.vector_store.chunk_documents(documents)
        
        print("Creating embeddings...")
        self.vector_store.create_embeddings()
        
        print("Retrieval system setup complete!")
    
    def format_context(self, search_results: List[Tuple[str, float]]) -> str:
        """Format search results into context string"""
        context_parts = []
        for i, (chunk, score) in enumerate(search_results):
            context_parts.append(f"[Document {i+1}, Relevance: {score:.3f}]\n{chunk}\n")
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, conversation_history: List[Dict] = None, max_new_tokens: int = 512) -> str:
        """Generate response using RAG pipeline"""
        if conversation_history is None:
            conversation_history = []
        
        # Retrieve relevant context
        search_results = self.vector_store.search(query, k=5)
        context = self.format_context(search_results)
        
        # Prepare messages with context
        messages = [
            {"role": "system", "content": self.system_prompt},
            *conversation_history
        ]
        
        # Add context and current query
        user_message = f"""Based on the following information about IIT Kharagpur, please answer the question.

Context Information:
{context}

Question: {query}

Please provide a helpful and accurate answer based on the context above."""
        
        messages.append({"role": "user", "content": user_message})
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate response
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract only the new tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
    
    def chat(self):
        """Start an interactive chat session"""
        print("Welcome to IIT KGP Senior Chatbot!")
        print("Type 'quit' to exit, 'clear' to clear conversation history.")
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'clear':
                    conversation_history = []
                    print("Conversation history cleared.")
                    continue
                elif not user_input:
                    continue
                
                print("Thinking...")
                response = self.generate_response(user_input, conversation_history)
                print(f"\nIIT KGP Senior: {response}")
                
                # Update conversation history (keep last 6 exchanges to manage context length)
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": response})
                
                # Keep only recent history to avoid context overflow
                if len(conversation_history) > 12:  # Last 6 exchanges
                    conversation_history = conversation_history[-12:]
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                print("Please try again.")

def main():
    # Initialize the chatbot with your data directory
    data_directory = "/home/du1/21CS30035/data_analytics_project/iit-kgp"  # Update this path
    
    if not os.path.exists(data_directory):
        print(f"Data directory {data_directory} not found!")
        return
    
    chatbot = IITKGPChatbot(data_directory)
    
    # Set up the retrieval system (this might take some time first time)
    print("Setting up retrieval system...")
    chatbot.setup_retrieval_system()
    
    # Start interactive chat
    chatbot.chat()

if __name__ == "__main__":
    main()