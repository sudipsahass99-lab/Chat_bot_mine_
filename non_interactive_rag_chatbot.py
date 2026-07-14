import os
import glob
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Tuple
import time

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
            if not document or len(document.strip()) == 0:
                continue
                
            # Simple chunking by character length
            words = document.split()
            for i in range(0, len(words), chunk_size - chunk_overlap):
                chunk = " ".join(words[i:i + chunk_size])
                if len(chunk) > 50:
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
        
        # Initialize the LLM
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"Loading model {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                padding_side='left'
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # System prompt for IIT KGP chatbot
        self.system_prompt = """You are an IIT Kharagpur senior and a helpful academic assistant.

Your primary responsibility is to answer questions using the provided context. 
If the context contains relevant information, always prioritize it and clearly base your answer on it.

If the context does NOT contain the required information, you may answer using your general and commonly known knowledge about IIT Kharagpur, its campus life, academics, hostels, culture, events, and facilities—but only if you are reasonably confident.

If you do not know something, or if it is highly specific and not covered in either the context or common IIT KGP knowledge, say that the information is not available.

Maintain a tone that is:
- precise and helpful
- friendly like a senior
- professional like an educational institution
"""

    def setup_retrieval_system(self):
        """Set up the complete retrieval system"""
        print("Loading documents...")
        documents = self.knowledge_base.load_documents()
        
        if not documents:
            print("No documents loaded! Check your data directory.")
            return False
            
        print("Chunking documents...")
        self.vector_store.chunk_documents(documents)
        
        if not self.vector_store.document_chunks:
            print("No document chunks created!")
            return False
            
        print("Creating embeddings...")
        self.vector_store.create_embeddings()
        
        print("Retrieval system setup complete!")
        return True
    
    def format_context(self, search_results: List[Tuple[str, float]]) -> str:
        """Format search results into context string"""
        context_parts = []
        for i, (chunk, score) in enumerate(search_results):
            context_parts.append(f"[Document {i+1}]: {chunk}")
        
        return "\n\n".join(context_parts)
    
    def generate_response(self, query: str, conversation_history: List[Dict] = None, max_new_tokens: int = 512) -> str:
        """Generate response using RAG pipeline"""
        if conversation_history is None:
            conversation_history = []
        
        try:
            # Retrieve relevant context
            search_results = self.vector_store.search(query, k=3)
            context = self.format_context(search_results)
            
            # Prepare messages with context
            messages = [
                {"role": "system", "content": self.system_prompt},
                *conversation_history
            ]
            
            # Add context and current query
            user_message = f"""Using the information below as the primary source, answer the question.  
If the context does not provide the answer, you may use general knowledge about IIT Kharagpur.  
If both are insufficient, clearly say the information is not available.

Context Information:
{context}

Question: {query}

Please provide a helpful and accurate answer."""
            
            messages.append({"role": "user", "content": user_message})
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Generate response
            model_inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Extract only the new tokens
            response_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            return f"I encountered an error while generating a response: {str(e)}"
    
    def run_test_queries(self, queries: List[str] = None):
        """Run a series of test queries in non-interactive mode"""
        if queries is None:
            # Default test queries about IIT KGP
            queries = [
                "What are the PhD admission requirements at IIT KGP?",
                "Tell me about CSE Dual Degree programs at IIT Kharagpur",
                "suggest Micro vs minor in CSE?",
                "What is the duration of PhD program at IIT KGP?",
                "Tell me about the campus facilities at IIT Kharagpur",
                "What are the different departments at IIT KGP?",
                "How is the placement scenario at IIT Kharagpur?",
                "What is requirements for MCM scholarship?",
                "Tell me about the MMM Hall"
            ]
        
        print("\n" + "="*70)
        print("IIT KGP SENIOR CHATBOT - NON-INTERACTIVE MODE")
        print("="*70)
        
        conversation_history = []
        total_start_time = time.time()
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*50}")
            print(f"QUERY {i}/{len(queries)}: {query}")
            print(f"{'='*50}")
            
            start_time = time.time()
            response = self.generate_response(query, conversation_history)
            end_time = time.time()
            
            print(f"RESPONSE: {response}")
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": response})
            
            # Keep only recent history
            if len(conversation_history) > 8:
                conversation_history = conversation_history[-8:]
            
            # Small delay between queries
            time.sleep(1)
        
        total_end_time = time.time()
        print(f"\n{'='*70}")
        print(f"COMPLETED ALL {len(queries)} QUERIES")
        print(f"Total time: {total_end_time - total_start_time:.2f} seconds")
        print(f"Average time per query: {(total_end_time - total_start_time)/len(queries):.2f} seconds")
        print("="*70)
    
    def process_single_query(self, query: str):
        """Process a single query and return the response"""
        print(f"\nProcessing query: {query}")
        start_time = time.time()
        
        response = self.generate_response(query)
        
        end_time = time.time()
        print(f"Response: {response}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
        return response

def main():
    # Initialize the chatbot with your data directory
    data_directory = "/home/du1/21CS30035/data_analytics_project/iit-kgp"
    
    if not os.path.exists(data_directory):
        print(f"Data directory {data_directory} not found!")
        return
    
    try:
        # Initialize chatbot
        chatbot = IITKGPChatbot(data_directory)
        
        # Set up the retrieval system
        print("Setting up retrieval system...")
        success = chatbot.setup_retrieval_system()
        
        if not success:
            print("Failed to set up retrieval system. Exiting.")
            return
        
        print("\n" + "="*70)
        print("SYSTEM READY - STARTING NON-INTERACTIVE TESTING")
        print("="*70)
        
        # Option 1: Run predefined test queries
        chatbot.run_test_queries()
        
        # Option 2: Process custom queries (uncomment to use)
        # custom_queries = [
        #     "What is the fee structure for MTech programs?",
        #     "Tell me about the library facilities",
        #     "What sports facilities are available?"
        # ]
        # chatbot.run_test_queries(custom_queries)
        
        # Option 3: Process single query (uncomment to use)
        # chatbot.process_single_query("What are the eligibility criteria for PhD admissions?")
        
    except Exception as e:
        print(f"Failed to initialize chatbot: {e}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()