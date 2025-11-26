"""
Phase 2: Data Processing & Ingestion for Disaster Chatbot
Processes multiple PDFs and creates a vector database using local embeddings
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DisasterDataIngestion:
    """Handles PDF processing and vector database creation"""

    def __init__(
            self,
            data_dir: str = "./data",
            db_dir: str = "./chroma_db",
            chunk_size: int = 1000,
            chunk_overlap: int = 150
    ):
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize local embeddings
        logger.info("Initializing local embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✓ Embeddings model loaded successfully")

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_disaster_type(self, filename: str) -> str:
        """Extract disaster type from filename"""
        filename_lower = filename.lower()

        disaster_keywords = {
            'climate': 'Climate Change',
            'disaster_risk': 'Disaster Risk',
            'environmental': 'Environmental',
            'exposure': 'Exposure & Vulnerability',
            'economic': 'Economic Development',
            'hazard': 'Hazards',
            'urban': 'Urban Planning',
            'poverty': 'Social Issues',
            'resilience': 'Resilience',
            'sovereign': 'Governance',
            'vulnerability': 'Vulnerability',
            'governance': 'Governance'
        }

        for keyword, disaster_type in disaster_keywords.items():
            if keyword in filename_lower:
                return disaster_type

        return 'General Disaster Risk'

    def load_single_pdf(self, pdf_path: str) -> list:
        """Load a single PDF and add metadata"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            # Extract filename and disaster type
            filename = os.path.basename(pdf_path)
            disaster_type = self.extract_disaster_type(filename)

            # Add metadata to each page
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'source': filename,
                    'page': i + 1,
                    'disaster_type': disaster_type,
                    'total_pages': len(documents)
                })

            logger.info(f"✓ Loaded {filename}: {len(documents)} pages")
            return documents

        except Exception as e:
            logger.error(f"✗ Error loading {pdf_path}: {str(e)}")
            return []

    def load_all_pdfs(self) -> list:
        """Load all PDFs from the data directory"""
        logger.info(f"Loading PDFs from {self.data_dir}...")

        pdf_files = list(Path(self.data_dir).glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.data_dir}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files")

        all_documents = []
        for pdf_file in pdf_files:
            docs = self.load_single_pdf(str(pdf_file))
            all_documents.extend(docs)

        logger.info(f"✓ Total pages loaded: {len(all_documents)}")
        return all_documents

    def chunk_documents(self, documents: list) -> list:
        """Split documents into smaller chunks"""
        logger.info("Splitting documents into chunks...")

        chunks = self.text_splitter.split_documents(documents)

        logger.info(f"✓ Created {len(chunks)} chunks")
        logger.info(f"  Average chunk size: ~{self.chunk_size} characters")
        logger.info(f"  Chunk overlap: {self.chunk_overlap} characters")

        return chunks

    def create_vectorstore(self, chunks: list, batch_size: int = 50) -> Chroma:
        """Create vector database from chunks"""
        logger.info("Creating vector database...")
        logger.info(f"Processing {len(chunks)} chunks in batches of {batch_size}")

        vectorstore = None

        # Process in batches with progress bar
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
            batch = chunks[i:i + batch_size]

            if vectorstore is None:
                # Create vectorstore with first batch
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    persist_directory=self.db_dir,
                    collection_name="disaster_knowledge"
                )
            else:
                # Add subsequent batches
                vectorstore.add_documents(batch)

        logger.info(f"✓ Vector database created at {self.db_dir}")
        return vectorstore

    def get_statistics(self, chunks: list) -> dict:
        """Get statistics about the processed data"""
        disaster_types = {}
        sources = set()

        for chunk in chunks:
            disaster_type = chunk.metadata.get('disaster_type', 'Unknown')
            source = chunk.metadata.get('source', 'Unknown')

            disaster_types[disaster_type] = disaster_types.get(disaster_type, 0) + 1
            sources.add(source)

        return {
            'total_chunks': len(chunks),
            'total_sources': len(sources),
            'chunks_by_disaster_type': disaster_types
        }

    def display_statistics(self, stats: dict):
        """Display processing statistics"""
        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total chunks created: {stats['total_chunks']}")
        logger.info(f"Total PDF sources: {stats['total_sources']}")
        logger.info("\nChunks by disaster type:")
        for disaster_type, count in stats['chunks_by_disaster_type'].items():
            logger.info(f"  • {disaster_type}: {count} chunks")
        logger.info("=" * 60 + "\n")

    def run(self):
        """Main execution pipeline"""
        logger.info("Starting data ingestion pipeline...")
        logger.info("=" * 60)

        # Step 1: Load PDFs
        documents = self.load_all_pdfs()
        if not documents:
            logger.error("No documents to process. Exiting.")
            return None

        # Step 2: Chunk documents
        chunks = self.chunk_documents(documents)

        # Step 3: Get statistics
        stats = self.get_statistics(chunks)
        self.display_statistics(stats)

        # Step 4: Create vector database
        vectorstore = self.create_vectorstore(chunks)

        logger.info("✅ Data ingestion completed successfully!")
        logger.info(f"Vector database saved to: {self.db_dir}")
        logger.info("\nYou can now use this database in your chatbot application.")

        return vectorstore


def main():
    """Main execution function"""

    # Initialize ingestion pipeline
    ingestion = DisasterDataIngestion(
        data_dir="./data",  # PDFs folder
        db_dir="./chroma_db",  # Where to save the database
        chunk_size=1000,  # Characters per chunk
        chunk_overlap=150  # Overlap between chunks
    )

    # Run the pipeline
    vectorstore = ingestion.run()

    # Test the database
    if vectorstore:
        logger.info("\n" + "=" * 60)
        logger.info("Testing vector database with sample query...")
        logger.info("=" * 60)

        test_query = "What is disaster risk reduction?"
        results = vectorstore.similarity_search(test_query, k=3)

        logger.info(f"\nQuery: '{test_query}'")
        logger.info(f"Retrieved {len(results)} relevant chunks:\n")

        for i, doc in enumerate(results, 1):
            logger.info(f"Result {i}:")
            logger.info(f"  Source: {doc.metadata.get('source', 'Unknown')}")
            logger.info(f"  Page: {doc.metadata.get('page', 'Unknown')}")
            logger.info(f"  Type: {doc.metadata.get('disaster_type', 'Unknown')}")
            logger.info(f"  Preview: {doc.page_content[:200]}...")
            logger.info("")


if __name__ == "__main__":
    main()