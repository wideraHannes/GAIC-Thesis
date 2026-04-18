"""
Embed all training sentences with OpenAI and store in ChromaDB.
One-time setup, reuse for all retrieval experiments.

Usage:
    uv run gaic/embeddings.py
"""

import json
import os

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from loguru import logger

from config.paths import GAIC_DATA_DIR, PROJECT_ROOT

load_dotenv()

CHROMA_PATH = PROJECT_ROOT / "data" / "embeddings"
COLLECTION_NAME = "train_sentences"


def load_train_data() -> tuple[dict[str, str], dict[str, str]]:
    """Load train split sentences and labels."""
    texts, labels = {}, {}
    with open(GAIC_DATA_DIR / "train.jsonl") as f:
        for line in f:
            item = json.loads(line)
            texts[item["id"]] = item["sentence"]
    with open(GAIC_DATA_DIR / "train_labels.jsonl") as f:
        for line in f:
            item = json.loads(line)
            labels[item["id"]] = item["label"]
    return texts, labels


def create_embedding_db():
    """Embed all ~10k training sentences and store in ChromaDB."""
    logger.info("Loading training data...")
    texts, labels = load_train_data()
    logger.info(f"Loaded {len(texts)} sentences")

    # Setup ChromaDB with OpenAI embeddings
    logger.info(f"Setting up ChromaDB at {CHROMA_PATH}")
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small",
    )

    # Delete existing collection if exists, create fresh
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
    )

    # Prepare data
    ids = list(texts.keys())
    sentences = [texts[id_] for id_ in ids]
    metadatas = [
        {"label": labels[id_], "dataset": id_.rsplit("-", 2)[0]} for id_ in ids
    ]

    # Add in batches (ChromaDB has limits)
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_docs = sentences[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_meta)
        logger.info(f"Embedded batch {i // batch_size + 1}/{(len(ids) + batch_size - 1) // batch_size}")

    logger.info(f"Embedded {len(ids)} sentences to {CHROMA_PATH}")
    return collection


def get_collection():
    """Load existing ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small",
    )
    return client.get_collection(COLLECTION_NAME, embedding_function=openai_ef)


def retrieve_similar_demos(
    test_sentence: str, k: int, collection=None, dataset: str = None
) -> list[dict]:
    """Retrieve k most similar Arguments + k most similar No-Arguments.

    Args:
        test_sentence: The sentence to find similar examples for.
        k: Number of examples per label (total = 2k).
        collection: ChromaDB collection (loaded if None).
        dataset: If provided, only retrieve from this dataset's training data.

    Returns 2k total demos, grouped: [all Args, then all No-Args].
    """
    if collection is None:
        collection = get_collection()

    # Build where filter
    if dataset:
        # Filter by both label AND dataset
        args_filter = {"$and": [{"label": "Argument"}, {"dataset": dataset}]}
        noargs_filter = {"$and": [{"label": "No-Argument"}, {"dataset": dataset}]}
    else:
        args_filter = {"label": "Argument"}
        noargs_filter = {"label": "No-Argument"}

    # Query for top-k Arguments
    args = collection.query(
        query_texts=[test_sentence],
        n_results=k,
        where=args_filter,
    )

    # Query for top-k No-Arguments
    noargs = collection.query(
        query_texts=[test_sentence],
        n_results=k,
        where=noargs_filter,
    )

    # Group by label: all Arguments first, then all No-Arguments
    demos = []
    for doc in args["documents"][0]:
        demos.append({"sentence": doc, "label": "Argument"})
    for doc in noargs["documents"][0]:
        demos.append({"sentence": doc, "label": "No-Argument"})

    return demos


def inspect_collection():
    """View statistics and sample entries from the vector store."""
    try:
        collection = get_collection()
    except Exception as e:
        print(f"Collection not found. Run 'uv run gaic/embeddings.py create' first.\nError: {e}")
        return

    # Get collection stats
    count = collection.count()
    print(f"=== ChromaDB Collection: {COLLECTION_NAME} ===")
    print(f"Path: {CHROMA_PATH}")
    print(f"Total documents: {count}")

    # Sample some entries
    sample = collection.peek(limit=5)
    print(f"\n=== Sample Entries ===")
    for i, (id_, doc, meta) in enumerate(zip(sample["ids"], sample["documents"], sample["metadatas"])):
        print(f"\n[{i+1}] ID: {id_}")
        print(f"    Label: {meta['label']}, Dataset: {meta['dataset']}")
        print(f"    Text: {doc[:100]}...")

    # Count by label and dataset
    print(f"\n=== Distribution ===")
    all_meta = collection.get(include=["metadatas"])["metadatas"]

    label_counts = {}
    dataset_counts = {}
    for m in all_meta:
        label_counts[m["label"]] = label_counts.get(m["label"], 0) + 1
        dataset_counts[m["dataset"]] = dataset_counts.get(m["dataset"], 0) + 1

    print("By label:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    print("\nBy dataset:")
    for ds, count in sorted(dataset_counts.items()):
        print(f"  {ds}: {count}")


def test_retrieval(query: str = "This drug significantly improves patient outcomes.", k: int = 3):
    """Test retrieval with a sample query."""
    print(f"=== Testing Retrieval ===")
    print(f"Query: {query}")
    print(f"k: {k} (will return {2*k} demos)")

    demos = retrieve_similar_demos(query, k)
    print(f"\n=== Retrieved {len(demos)} Demonstrations ===")
    for i, d in enumerate(demos, 1):
        print(f"\n[{i}] Label: {d['label']}")
        print(f"    {d['sentence'][:150]}...")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "create":
            create_embedding_db()
        elif cmd == "inspect":
            inspect_collection()
        elif cmd == "test":
            query = sys.argv[2] if len(sys.argv) > 2 else "This drug significantly improves patient outcomes."
            test_retrieval(query)
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: uv run gaic/embeddings.py [create|inspect|test]")
    else:
        print("Usage:")
        print("  uv run gaic/embeddings.py create   # Create embedding database")
        print("  uv run gaic/embeddings.py inspect  # View collection stats")
        print("  uv run gaic/embeddings.py test     # Test retrieval")
