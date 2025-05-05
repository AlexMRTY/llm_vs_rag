
import chromadb
import ollama
import pprint
import json
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm


def load_documents(file_path):
    print("🔄 Loading documents...")
    docs = {}
    # counter = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # if counter == 1000:
            #     break
            # counter += 1
            data = json.loads(line.strip())
            docs[data["id"]] = data["content"]
    return docs

def load_QA_pairs(file_path):
    print("🔄 Loading QA pairs...")
    pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            pairs.append({
                "id": data["id"],
                "content": data["document"],
                "question": data["question"],
                "answer": data["answer"],
                })
    return pairs

def embed_documents(documents, embedding, batch_size=32):
    embeddings = []
    batch = []
    with tqdm(total=len(documents), desc="Embedding documents: ") as pbar:
        for doc in documents:
            batch.append(doc)
            if len(batch) >= batch_size:
                batch_embeddings = embedding.embed_documents(batch)
                embeddings.extend(batch_embeddings)
                batch = []
                id_batch = []
                pbar.update(32)
        # Remaining documents
        batch_embeddings = embedding.embed_documents(batch)
        embeddings.extend(batch_embeddings)
        pbar.update(len(batch))
    return embeddings

def add_documents(collection, embedding):
    # Separate documents and ids
    documents = load_documents("data/refined-web-100k-updated.jsonl")
    document_contents = []
    document_ids = []
    for key in documents.keys():
        document_contents.append(documents[key])
        document_ids.append(key)

    # embeddings = embed_documents(document_contents, embedding, 32)

    # Add data to a collection
    # collection.add(
    #     documents=document_contents,
    #     embeddings=embeddings,
    #     ids=document_ids
    # )

    batch = []
    id_batch = []
    with tqdm(total=len(document_contents), desc="Embedding documents: ") as pbar:
        for index, doc in enumerate(document_contents):
            batch.append(doc)
            id_batch.append(document_ids[index])
            if len(batch) >= 32:
                batch_embeddings = embedding.embed_documents(batch)
                collection.add(
                    documents=batch,
                    embeddings=batch_embeddings,
                    ids=id_batch
                )
                batch = []
                id_batch = []
                pbar.update(32)
        # Remaining documents
        batch_embeddings = embedding.embed_documents(batch)
        collection.add(
            documents=batch,
            embeddings=batch_embeddings,
            ids=id_batch
        )
        pbar.update(len(batch))


def main():
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    client = chromadb.PersistentClient("data/101k-test/chroma")
    collection = client.get_or_create_collection(name="101k_performance_test")
    # add_documents(collection, embedding)


    # print(collection.count())
    # pprint.pp(collection.)
    query_embedding = embedding.embed_query("When were the WHO/CIOMS International Ethical Guidelines for Biomedical Research Involving Human Subjects published?")
    pprint.pp(collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    ))

if __name__ == '__main__':
    main()









