"""
Quick test script for RAG pipeline
Run this to verify your setup is working correctly
"""

import os
from dotenv import load_dotenv
from rag_pipeline import DisasterChatbot

# Load environment variables from .env file
load_dotenv()


def test_basic_functionality():
    """Test basic chatbot functionality"""

    print("=" * 70)
    print("TESTING RAG PIPELINE")
    print("=" * 70)

    # Check if API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n❌ GOOGLE_API_KEY not found!")
        print("\nPlease:")
        print("1. Copy .env.example to .env")
        print("2. Add your Gemini API key to .env")
        print("3. Get free API key from: https://ai.google.dev/")
        return False

    print("\n✓ API key found")

    # Initialize chatbot
    print("\n1️⃣ Initializing chatbot...")
    try:
        chatbot = DisasterChatbot()
        print("✓ Chatbot initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return False

    # Test database connection
    print("\n2️⃣ Testing database connection...")
    try:
        stats = chatbot.get_database_stats()
        print(f"✓ Database loaded: {stats['total_chunks']} chunks available")
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

    # Test retrieval (without LLM)
    print("\n3️⃣ Testing document retrieval...")
    try:
        results = chatbot.vectorstore.similarity_search(
            "earthquake safety",
            k=2
        )
        print(f"✓ Retrieved {len(results)} relevant documents")
        print(f"   Example: {results[0].metadata.get('source', 'Unknown')}")
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        return False

    # Test full RAG pipeline
    print("\n4️⃣ Testing full RAG pipeline (with Gemini)...")
    test_question = "What is disaster risk?"

    try:
        print(f"   Question: '{test_question}'")
        response = chatbot.ask(test_question, return_sources=True)

        print(f"✓ Response generated successfully")
        print(f"\n   Answer preview: {response['answer'][:150]}...")

        if 'sources' in response:
            print(f"   Sources used: {len(response['sources'])} documents")

    except Exception as e:
        print(f"❌ RAG pipeline error: {e}")
        return False

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nYour RAG pipeline is working correctly!")
    print("Next step: Run the full demo or interactive mode:")
    print("  • python rag_pipeline.py")
    print("  • python rag_pipeline.py --interactive")

    return True


if __name__ == "__main__":
    test_basic_functionality()