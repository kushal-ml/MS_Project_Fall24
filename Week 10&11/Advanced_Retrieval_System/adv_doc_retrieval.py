import warnings
import logging
import os
import sys
from typing import List, Tuple
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.evaluation import load_evaluator
from datasets import load_dataset
import transformers
from tqdm import tqdm
import shutil
from dotenv import load_dotenv
import gc
import torch

# Constants for tiny dataset
MAX_PASSAGES = 100  # Extremely small dataset
MAX_QA_PAIRS = 20   # Very few QA pairs
BATCH_SIZE = 2      # Process just 2 documents at a time

# Remove existing data
if os.path.exists('data'):
    shutil.rmtree('data')
os.makedirs('data', exist_ok=True)

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class BiomedicalRAG:
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0):
        # Force CPU usage to avoid GPU memory issues
        self.embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # Smallest model
            model_kwargs={'device': 'cpu'},  # Force CPU
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.vectorstore = None
        self.persist_directory = "data/chroma_db"
        
        # Create directory
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
        
        # Initialize prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful biomedical research assistant. Answer briefly."),
            ("user", "Context: {context}\n\nQuestion: {question}")
        ])
        
        # Setup chain
        self._setup_chain()

    def _setup_chain(self):
        """Setup the RAG chain"""
        if self.vectorstore is None:
            return
            
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        self.chain = (
            {"context": retriever, "question": RunnablePassthrough()} 
            | self.prompt 
            | self.llm 
            | StrOutputParser()
        )

    def _download_and_save_dataset(self):
        """Download a tiny subset of the dataset"""
        try:
            # Load datasets
            passages_dataset = load_dataset("rag-datasets/rag-mini-bioasq", 'text-corpus')
            qa_dataset = load_dataset("rag-datasets/rag-mini-bioasq", 'question-answer-passages')
            
            # Get correct splits
            passages_split = next(iter(passages_dataset.keys()))
            qa_split = next(iter(qa_dataset.keys()))
            
            # Sample tiny subsets
            passages_df = pd.DataFrame(passages_dataset[passages_split]).sample(
                n=min(MAX_PASSAGES, len(passages_dataset[passages_split])),
                random_state=42
            )
            qa_df = pd.DataFrame(qa_dataset[qa_split]).sample(
                n=min(MAX_QA_PAIRS, len(qa_dataset[qa_split])),
                random_state=42
            )
            
            # Save to disk
            passages_df.to_parquet('data/passages.parquet')
            qa_df.to_parquet('data/qa_pairs.parquet')
            
            return passages_df, qa_df
            
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            raise

    def _create_vector_store_in_tiny_batches(self, texts: List[str]):
        """Create vector store processing just a few documents at a time"""
        try:
            # Process first two documents
            print("\nInitializing vector store with first batch...")
            initial_texts = texts[:BATCH_SIZE]
            self.vectorstore = Chroma.from_texts(
                texts=initial_texts,
                embedding=self.embed_model,
                persist_directory=self.persist_directory
            )
            self.vectorstore.persist()
            
            # Process remaining documents in tiny batches
            remaining_texts = texts[BATCH_SIZE:]
            
            for i in tqdm(range(0, len(remaining_texts), BATCH_SIZE), desc="Processing documents"):
                # Clear memory
                gc.collect()
                
                # Get batch
                batch_texts = remaining_texts[i:i + BATCH_SIZE]
                
                # Add to vector store
                self.vectorstore.add_texts(texts=batch_texts)
                self.vectorstore.persist()
                
                # Small delay
                import time
                time.sleep(0.1)
            
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            raise

    def load_and_process_data(self) -> None:
        """Load data and create/load vector store"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if os.path.exists('data/passages.parquet') and os.path.exists('data/qa_pairs.parquet'):
                    print("Loading from local files...")
                    self.passages = pd.read_parquet('data/passages.parquet')
                    self.test_data = pd.read_parquet('data/qa_pairs.parquet')
                else:
                    print("Local files not found. Downloading dataset...")
                    print("\nThis may take a few minutes for the first run...")
                    # Removed max_samples parameter since it's not used
                    passages_df, qa_df = self._download_and_save_dataset()
                    
                    # Process passages
                    passages_df["valid"] = passages_df.passage.apply(lambda x: len(str(x).split()) > 20)
                    self.passages = passages_df[passages_df.valid].reset_index(drop=True)
                    self.test_data = qa_df
                
                print(f"\nPassages shape: {self.passages.shape}")
                print(f"QA pairs shape: {self.test_data.shape}")
                
                # Create or load vector store
                if os.path.exists(os.path.join(self.persist_directory, 'chroma.sqlite3')):
                    print("\nLoading existing vector store...")
                    self.vectorstore = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embed_model
                    )
                else:
                    print("\nCreating new vector store...")
                    texts = self.passages.passage.tolist()
                    self._create_vector_store_in_tiny_batches(texts)
                
                # Setup chain
                self._setup_chain()
                print("Data loading and processing completed")
                
            except Exception as e:
                print(f"Error in data loading: {str(e)}")
                raise

    def query(self, question: str) -> Tuple[str, List[str], str]:
        """Query the RAG system and find actual answer if available"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Get retriever
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )
                
                # Get contexts
                retrieved_docs = retriever.get_relevant_documents(question)
                context = [doc.page_content for doc in retrieved_docs]
                
                # Get predicted answer
                predicted_answer = self.chain.invoke(question)
                
                # Try to find actual answer
                actual_answer = None
                try:
                    # Case-insensitive search for the question
                    matching_qa = self.test_data[
                        self.test_data['question'].str.lower() == question.lower()
                    ]
                    if not matching_qa.empty:
                        actual_answer = matching_qa.iloc[0]['answer']
                except Exception as e:
                    print(f"Error finding actual answer: {str(e)}")
                    actual_answer = None
                
                return predicted_answer, context, actual_answer
                
            except Exception as e:
                print(f"Error during query: {str(e)}")
                raise

    def evaluate(self, num_samples: int = 5):
        """Simple evaluation without external evaluator"""
        try:
            # Sample test cases
            test_samples = self.test_data.sample(n=min(num_samples, len(self.test_data)))
            scores = []
            
            print("\nEvaluating responses...")
            for _, row in tqdm(test_samples.iterrows(), desc="Processing samples"):
                gc.collect()  # Clear memory between iterations
                
                try:
                    # Get prediction
                    question = row['question']
                    print(f"\nQuestion: {question}")
                    
                    pred_answer, contexts, actual_answer = self.query(question)
                    print(f"Predicted Answer: {pred_answer}")
                    print(f"Actual Answer: {row['answer']}")
                    
                    # Let user rate the answer
                    while True:
                        try:
                            score = float(input("Rate this answer (0-1, where 1 is perfect): "))
                            if 0 <= score <= 1:
                                break
                            print("Please enter a number between 0 and 1")
                        except ValueError:
                            print("Please enter a valid number")
                    
                    scores.append(score)
                    print(f"Current average score: {sum(scores)/len(scores):.2f}")
                    
                except Exception as e:
                    print(f"Error evaluating question: {str(e)}")
                    continue
            
            if scores:
                final_score = sum(scores) / len(scores)
                print(f"\nFinal evaluation score: {final_score:.2f}")
                return final_score
            else:
                print("\nNo valid evaluations completed")
                return 0.0
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise

def main():
    try:
        print("Initializing minimal RAG system...")
        rag = BiomedicalRAG()
        
        print("\nLoading and processing small dataset...")
        rag.load_and_process_data()
        
        while True:
            print("\n" + "="*50)
            print("1. Ask a question")
            print("2. Run evaluation")
            print("3. View sample questions")
            print("4. Exit")
            choice = input("\nEnter your choice (1-4): ")
            
            if choice == "1":
                question = input("\nEnter your question: ")
                if not question:
                    question = "Where is the protein Pannexin1 located?"
                    print(f"Using default question: {question}")
                
                print("\nProcessing query...")
                answer, contexts, actual_answer = rag.query(question)
                
                print(f"\nPredicted Answer: {answer}")
                if actual_answer:
                    print(f"Actual Answer: {actual_answer}")
                else:
                    print("No ground truth answer available for this question")
                
                print("\nRetrieved Contexts:")
                for i, ctx in enumerate(contexts, 1):
                    print(f"\nContext {i}: {ctx[:200]}...")
                
                # Optional: Let user rate the answer
                rating = input("\nWould you like to rate this answer? (y/n): ")
                if rating.lower() == 'y':
                    while True:
                        try:
                            score = float(input("Rate this answer (0-1, where 1 is perfect): "))
                            if 0 <= score <= 1:
                                print(f"\nYou rated this answer: {score:.2f}")
                                break
                            print("Please enter a number between 0 and 1")
                        except ValueError:
                            print("Please enter a valid number")
            
            elif choice == "2":
                num_samples = input("\nEnter number of samples to evaluate (default: 3): ")
                try:
                    num_samples = int(num_samples)
                except:
                    num_samples = 3
                
                print(f"\nRunning evaluation with {num_samples} samples...")
                score = rag.evaluate(num_samples=num_samples)
                print(f"\nFinal average evaluation score: {score:.2f}")
            
            elif choice == "3":
                print("\nSample questions from the dataset:")
                sample_questions = rag.test_data['question'].sample(n=min(5, len(rag.test_data))).tolist()
                for i, q in enumerate(sample_questions, 1):
                    print(f"{i}. {q}")
                
                use_sample = input("\nWould you like to use one of these questions? (1-5/n): ")
                if use_sample.isdigit() and 1 <= int(use_sample) <= len(sample_questions):
                    question = sample_questions[int(use_sample)-1]
                    print(f"\nUsing question: {question}")
                    answer, contexts, actual_answer = rag.query(question)
                    
                    print(f"\nPredicted Answer: {answer}")
                    if actual_answer:
                        print(f"Actual Answer: {actual_answer}")
                    
                    print("\nRetrieved Contexts:")
                    for i, ctx in enumerate(contexts, 1):
                        print(f"\nContext {i}: {ctx[:200]}...")
            
            elif choice == "4":
                print("\nExiting...")
                break
            
            else:
                print("\nInvalid choice. Please try again.")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    transformers.logging.set_verbosity_error()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    main()