"""Ingest documents and anchors into a vector store using LangChain.

This script will:
- Recursively load PDF files from a docs directory
- Load JSON anchors from `tier0_anchors.json`
- Split documents into chunks
- Create embeddings (selectable provider: `openai`, `groq`, or `hf`)
- Persist a FAISS vector store to disk

Usage:
	python src/ingest.py --docs_dir docs --anchors_file data/tier0_anchors.json --out_dir vector_store

Configure the embedding provider with environment variables, for example:
- `OPENAI_API_KEY` for OpenAI
- `GROQ_API_URL` and `GROQ_API_KEY` for Groq
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional
 
import requests

from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

try:
	from langchain.embeddings import HuggingFaceEmbeddings
except Exception:
	HuggingFaceEmbeddings = None


def find_pdfs(docs_dir: Path) -> List[Path]:
	return [p for p in docs_dir.rglob("*.pdf")]


def load_pdfs(paths: List[Path]) -> List[Document]:
	docs: List[Document] = []
	for p in paths:
		try:
			loader = PyPDFLoader(str(p))
			loaded = loader.load()
			for d in loaded:
				d.metadata = dict(d.metadata or {})
				d.metadata.update({"source": str(p)})
			docs.extend(loaded)
		except Exception:
			# fallback: treat as text file
			try:
				loader = TextLoader(str(p))
				loaded = loader.load()
				for d in loaded:
					d.metadata = dict(d.metadata or {})
					d.metadata.update({"source": str(p)})
				docs.extend(loaded)
			except Exception as e:
				print(f"Failed to load {p}: {e}")
	return docs


def load_anchors(anchors_path: Path) -> List[Document]:
	docs: List[Document] = []
	if not anchors_path.exists():
		return docs
	with anchors_path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	# Expecting a list or dict of anchors. Normalize to list of entries.
	if isinstance(data, dict):
		items = data.get("anchors") or data.get("items") or list(data.values())
	else:
		items = data

	for i, item in enumerate(items):
		if isinstance(item, dict):
			text = item.get("text") or item.get("content") or json.dumps(item)
			meta = {k: v for k, v in item.items() if k != "text" and k != "content"}
		else:
			text = str(item)
			meta = {}
		meta.update({"source": str(anchors_path), "anchor_index": i})
		docs.append(Document(page_content=text, metadata=meta))
	return docs


def split_docs(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
	splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
	texts = []
	for d in docs:
		parts = splitter.split_text(d.page_content)
		for i, p in enumerate(parts):
			meta = dict(d.metadata or {})
			meta.update({"chunk": i})
			texts.append(Document(page_content=p, metadata=meta))
	return texts


# Note: embeddings implementations are imported/instantiated dynamically
# based on the `--embeddings` CLI flag. See `build_vector_store_with_embeddings`.


def main():
	parser = argparse.ArgumentParser(description="Ingest docs into a vector store")
	parser.add_argument("--docs_dir", type=str, default="docs", help="Directory containing docs (PDFs)")
	parser.add_argument("--anchors_file", type=str, default="tier0_anchors.json", help="JSON anchors file path")
	parser.add_argument("--out_dir", type=str, default="vector_store", help="Output directory for vector store")
	parser.add_argument("--chunk_size", type=int, default=1000)
	parser.add_argument("--chunk_overlap", type=int, default=200)
	parser.add_argument("--embeddings", type=str, default="openai", choices=["openai", "groq", "hf"], help="Embedding provider to use")
	parser.add_argument("--groq_url", type=str, default=None, help="(optional) Groq API URL override")
	parser.add_argument("--groq_key", type=str, default=None, help="(optional) Groq API key override")
	args = parser.parse_args()

	base = Path.cwd()
	docs_dir = (base / args.docs_dir).resolve()
	anchors_path = (base / args.anchors_file).resolve()
	out_dir = (base / args.out_dir).resolve()

	print(f"Scanning PDFs in {docs_dir}")
	pdf_paths = find_pdfs(docs_dir)
	print(f"Found {len(pdf_paths)} PDF(s)")

	pdf_docs = load_pdfs(pdf_paths)
	print(f"Loaded {len(pdf_docs)} pages/segments from PDFs")

	anchors = load_anchors(anchors_path)
	print(f"Loaded {len(anchors)} anchor documents from {anchors_path}")

	all_docs = pdf_docs + anchors
	if not all_docs:
		print("No documents found to ingest. Exiting.")
		return

	print("Splitting documents into chunks...")
	chunks = split_docs(all_docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
	print(f"Created {len(chunks)} chunks")

	# select embeddings
	emb_choice = args.embeddings.lower()
	if emb_choice == "openai":
		# import OpenAIEmbeddings only when requested to avoid hard dependency
		try:
			from langchain.embeddings import OpenAIEmbeddings
		except Exception as e:
			raise RuntimeError("OpenAIEmbeddings not available in this environment") from e
		embeddings = OpenAIEmbeddings()
	elif emb_choice == "hf":
		if HuggingFaceEmbeddings is None:
			raise RuntimeError("HuggingFaceEmbeddings not available in this environment")
		embeddings = HuggingFaceEmbeddings()
	elif emb_choice == "groq":
		groq_url = args.groq_url or os.getenv("GROQ_API_URL")
		groq_key = args.groq_key or os.getenv("GROQ_API_KEY")
		embeddings = GroqEmbeddings(api_url=groq_url, api_key=groq_key)
	else:
		raise ValueError(f"Unknown embeddings provider: {args.embeddings}")

	print("Building vector store (this will call your embedding provider)...")
	build_vector_store_with_embeddings(chunks, out_dir, embeddings)


def build_vector_store_with_embeddings(docs: List[Document], out_dir: Path, embeddings):
	"""Build and persist FAISS vector store using provided embeddings instance."""
	vs = FAISS.from_documents(docs, embeddings)
	out_dir.mkdir(parents=True, exist_ok=True)
	vs.save_local(str(out_dir))
	print(f"Saved vector store to {out_dir}")


class GroqEmbeddings:
	"""Minimal wrapper to call a Groq-compatible embeddings HTTP API.

	The wrapper expects an API that accepts POST JSON with `input: [str...]`
	and returns a JSON object with a `data` list containing `embedding` arrays,
	similar to many embedding APIs. Set `GROQ_API_URL` and `GROQ_API_KEY` env vars
	or pass `api_url` / `api_key` to the constructor.
	"""

	def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
		self.api_url = api_url or os.getenv("GROQ_API_URL")
		self.api_key = api_key or os.getenv("GROQ_API_KEY")
		if not self.api_url or not self.api_key:
			raise ValueError("GROQ_API_URL and GROQ_API_KEY must be set for GroqEmbeddings")

	def _request(self, texts: List[str]):
		headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
		body = {"input": texts}
		resp = requests.post(self.api_url, headers=headers, json=body, timeout=60)
		resp.raise_for_status()
		return resp.json()

	def embed_documents(self, texts: List[str]) -> List[List[float]]:
		data = self._request(texts)
		# Accept several common response shapes
		if isinstance(data, dict) and "data" in data:
			out = []
			for item in data["data"]:
				emb = item.get("embedding") or item.get("embeddings")
				if emb is None:
					raise ValueError("Unexpected Groq response format: missing embedding")
				out.append(emb)
			return out
		if isinstance(data, dict) and "embeddings" in data:
			return data["embeddings"]
		raise ValueError("Unexpected Groq API response format")

	def embed_query(self, text: str) -> List[float]:
		return self.embed_documents([text])[0]


if __name__ == "__main__":
	main()

