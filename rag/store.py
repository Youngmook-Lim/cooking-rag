import chromadb
from rag.models import Chunk, Retrieved


class VectorStore:
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        raise NotImplementedError

    def query(self, vector: list[float], k: int) -> list[Retrieved]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class ChromaStore(VectorStore):
    def __init__(self, collection: str, persist_dir: str | None = None):
        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.EphemeralClient()
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self._next_id = self._col.count()

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        ids, metadatas, documents = [], [], []
        for chunk in chunks:
            ids.append(str(self._next_id))
            self._next_id += 1
            metadatas.append(
                {"post_id": chunk.post_id, "question_id": chunk.question_id}
            )
            documents.append(chunk.text)
        self._col.add(ids=ids, embeddings=vectors,
                      metadatas=metadatas, documents=documents)

    def query(self, vector: list[float], k: int) -> list[Retrieved]:
        res = self._col.query(query_embeddings=[vector], n_results=k)
        out: list[Retrieved] = []
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for text, meta, dist in zip(docs, metas, dists):
            chunk = Chunk(text=text, post_id=int(meta["post_id"]),
                          question_id=int(meta["question_id"]))
            out.append(Retrieved(chunk=chunk, score=1.0 - dist))  # cosine sim
        return out

    def count(self) -> int:
        return self._col.count()
