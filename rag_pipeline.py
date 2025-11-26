"""
Phase 4: RAG Pipeline for Disaster Chatbot
Combines vector database retrieval with Gemini LLM for intelligent responses
"""

import os
from typing import List, Dict, Optional
# from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory  # ← Fixed
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import logging

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DisasterChatbot:
    """RAG-based chatbot for disaster risk reduction and preparedness"""

    def __init__(
        self,
        db_dir: str = "./chroma_db",
        gemini_api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.3
    ):
        """
        Initialize the disaster chatbot

        Args:
            db_dir: Path to ChromaDB vector database
            gemini_api_key: Google Gemini API key (or set GOOGLE_API_KEY env var)
            model_name: Gemini model to use
            temperature: Response creativity (0=factual, 1=creative)
        """
        self.db_dir = db_dir

        # Setup API key
        if gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
        elif "GOOGLE_API_KEY" not in os.environ:
            raise ValueError(
                "Gemini API key required. Either pass gemini_api_key parameter "
                "or set GOOGLE_API_KEY environment variable"
            )

        logger.info("Initializing Disaster Risk Reduction Chatbot...")

        # Initialize embeddings (same as ingestion)
        logger.info("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Load vector database
        logger.info(f"Loading vector database from {db_dir}...")
        self.vectorstore = Chroma(
            persist_directory=db_dir,
            embedding_function=self.embeddings,
            collection_name="disaster_knowledge"
        )

        # Initialize LLM
        logger.info(f"Initializing Gemini model: {model_name}...")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            convert_system_message_to_human=True
        )

        # Setup retriever with optimized parameters
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": 4,           # Retrieve top 4 chunks
                "fetch_k": 20     # Consider top 20 before MMR filtering
            }
        )

        # Create custom prompt template
        self.prompt_template = self._create_prompt_template()

        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Create the conversational chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.prompt_template},
            verbose=False
        )

        logger.info("✅ Chatbot initialized successfully!")

    def _create_prompt_template(self) -> PromptTemplate:
        """Create custom prompt template for disaster preparedness context"""

        template = """You are an expert assistant specialized in Disaster Risk Reduction and Preparedness. 
                    Your role is to provide accurate, helpful, and actionable information to help people understand and prepare for disasters.
                    
                    IMPORTANT GUIDELINES:
                    1. Base your answers PRIMARILY on the context provided below
                    2. Be clear, concise, and practical in your responses
                    3. If the context doesn't contain enough information, say so honestly
                    4. Always prioritize safety and official guidelines
                    5. Use simple language that anyone can understand
                    6. Include specific steps or recommendations when applicable
                    7. Cite disaster types (e.g., earthquakes, floods) when relevant
                    
                    CONTEXT FROM KNOWLEDGE BASE:
                    {context}
                    
                    QUESTION: {question}
                    
                    HELPFUL ANSWER:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def ask(self, question: str, return_sources: bool = True) -> Dict:
        """
        Ask a question to the chatbot

        Args:
            question: User's question
            return_sources: Whether to return source documents

        Returns:
            Dictionary with 'answer' and optionally 'sources'
        """
        try:
            # Get response from chain
            response = self.chain({"question": question})

            result = {
                "answer": response["answer"],
                "question": question
            }

            # Add source information if requested
            if return_sources and "source_documents" in response:
                sources = self._format_sources(response["source_documents"])
                result["sources"] = sources

            return result

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error processing your question. Please try rephrasing it.",
                "error": str(e)
            }

    def _format_sources(self, documents: List[Document]) -> List[Dict]:
        """Format source documents for display"""
        sources = []

        for i, doc in enumerate(documents, 1):
            source_info = {
                "number": i,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "disaster_type": doc.metadata.get("disaster_type", "General"),
                "content_preview": doc.page_content[:200] + "..."
            }
            sources.append(source_info)

        return sources

    def search_by_disaster_type(self, query: str, disaster_type: str, k: int = 3) -> List[Document]:
        """
        Search for information filtered by disaster type

        Args:
            query: Search query
            disaster_type: Filter by disaster type
            k: Number of results to return

        Returns:
            List of relevant documents
        """
        results = self.vectorstore.similarity_search(
            query,
            k=k,
            filter={"disaster_type": disaster_type}
        )
        return results

    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history"""
        if hasattr(self.memory, 'chat_memory'):
            messages = self.memory.chat_memory.messages
            history = []

            for msg in messages:
                history.append({
                    "role": "user" if msg.type == "human" else "assistant",
                    "content": msg.content
                })

            return history
        return []

    def clear_history(self):
        """Clear conversation history"""
        self.memory.clear()
        logger.info("Conversation history cleared")

    def get_database_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        collection = self.vectorstore._collection

        return {
            "total_chunks": collection.count(),
            "embedding_dimension": len(self.embeddings.embed_query("test")),
            "database_path": self.db_dir
        }


def demo_chatbot():
    """Demonstration of the chatbot functionality"""

    print("="*70)
    print("DISASTER RISK REDUCTION CHATBOT - DEMO")
    print("="*70)

    # Initialize chatbot (make sure to set your API key!)
    # Option 1: Set environment variable
    # export GOOGLE_API_KEY="your-api-key-here"

    # Option 2: Pass directly (not recommended for production)
    # chatbot = DisasterChatbot(gemini_api_key="your-api-key-here")

    try:
        chatbot = DisasterChatbot()
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTo use this demo, you need to set your Gemini API key:")
        print("1. Get a free API key from: https://ai.google.dev/")
        print("2. Set environment variable: export GOOGLE_API_KEY='your-key'")
        print("3. Or create a .env file with: GOOGLE_API_KEY=your-key")
        return

    # Show database stats
    stats = chatbot.get_database_stats()
    print(f"\n📊 Knowledge Base Stats:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Embedding dimension: {stats['embedding_dimension']}")
    print(f"   Database: {stats['database_path']}")

    # Example questions
    questions = [
        "What is disaster risk reduction and why is it important?",
        "How does climate change affect disaster risk?",
        "What can communities do to prepare for earthquakes?",
        "What is the relationship between poverty and disaster vulnerability?"
    ]

    print("\n" + "="*70)
    print("ASKING SAMPLE QUESTIONS")
    print("="*70)

    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*70}")
        print(f"Question {i}: {question}")
        print(f"{'─'*70}")

        response = chatbot.ask(question, return_sources=True)

        print(f"\n💬 Answer:")
        print(response["answer"])

        if "sources" in response:
            print(f"\n📚 Sources ({len(response['sources'])} documents):")
            for source in response["sources"]:
                print(f"   [{source['number']}] {source['source']} (Page {source['page']}, {source['disaster_type']})")

        print()

    # Show conversation history
    print("\n" + "="*70)
    print("CONVERSATION HISTORY")
    print("="*70)
    history = chatbot.get_conversation_history()
    print(f"Total exchanges: {len(history) // 2}")

    print("\n✅ Demo completed successfully!")


def interactive_mode():
    """Interactive chat mode"""

    print("="*70)
    print("DISASTER RISK REDUCTION CHATBOT - INTERACTIVE MODE")
    print("="*70)
    print("\nType 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("Type 'stats' to see database statistics")
    print("="*70)

    try:
        chatbot = DisasterChatbot()
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease set your GOOGLE_API_KEY environment variable")
        return

    print("\n✅ Chatbot ready! Ask me anything about disaster preparedness.\n")

    while True:
        try:
            # Get user input
            question = input("You: ").strip()

            if not question:
                continue

            # Handle special commands
            if question.lower() in ['quit', 'exit']:
                print("\n👋 Thank you for using the Disaster Risk Reduction Chatbot!")
                break

            if question.lower() == 'clear':
                chatbot.clear_history()
                print("✓ Conversation history cleared\n")
                continue

            if question.lower() == 'stats':
                stats = chatbot.get_database_stats()
                print(f"\n📊 Knowledge Base Statistics:")
                print(f"   Total chunks: {stats['total_chunks']}")
                print(f"   Database: {stats['database_path']}\n")
                continue

            # Get response
            print("\n🤖 Assistant: ", end="", flush=True)
            response = chatbot.ask(question, return_sources=False)
            print(response["answer"])
            print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


if __name__ == "__main__":
    import sys

    # Check if user wants interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        demo_chatbot()
        print("\n💡 Tip: Run with --interactive flag for chat mode:")
        print("   python rag_pipeline.py --interactive")