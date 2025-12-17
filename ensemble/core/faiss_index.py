"""
FAISS Indexer for DeTeCtive detector
Handles KNN search for style-based AI text detection
"""

import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np
from tqdm import tqdm


class Indexer:
    """
    FAISS-based vector indexer for KNN style clustering.
    Used by DeTeCtive to match text embeddings against known AI/Human samples.
    """

    def __init__(self, vector_sz: int, device: str = 'cpu'):
        """
        Initialize the indexer
        
        Args:
            vector_sz: Dimension of embedding vectors
            device: Device to use ('cpu' recommended for faiss-cpu)
        """
        self.index = faiss.IndexFlatIP(vector_sz)
        # Force CPU mode - faiss-gpu has compatibility issues with AMD
        self.device = 'cpu'
        self.index_id_to_db_id = []

    def index_data(self, ids: List, embeddings: np.ndarray):
        """Add embeddings to the index"""
        self._update_id_mapping(ids)
        embeddings = embeddings.astype('float32')
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)
        print(f'Total data indexed {self.index.ntotal}')

    def search_knn(
        self, 
        query_vectors: np.ndarray, 
        top_docs: int, 
        index_batch_size: int = 8
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Search for k nearest neighbors
        
        Args:
            query_vectors: Query embedding vectors
            top_docs: Number of neighbors to return
            index_batch_size: Batch size for search
            
        Returns:
            List of (ids, scores) tuples
        """
        query_vectors = query_vectors.astype('float32')
        result = []
        nbatch = (len(query_vectors) - 1) // index_batch_size + 1
        
        for k in range(nbatch):
            start_idx = k * index_batch_size
            end_idx = min((k + 1) * index_batch_size, len(query_vectors))
            q = query_vectors[start_idx:end_idx]
            scores, indexes = self.index.search(q, top_docs)
            
            # Convert to external ids
            db_ids = [
                [str(self.index_id_to_db_id[i]) for i in query_top_idxs] 
                for query_top_idxs in indexes
            ]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
            
        return result

    def serialize(self, dir_path: str):
        """Save index to disk"""
        os.makedirs(dir_path, exist_ok=True)
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        
        print(f'Serializing index to {index_file}, meta data to {meta_file}')
        faiss.write_index(self.index, index_file)
        
        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, dir_path: str):
        """Load index from disk"""
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        
        print(f'Loading index from {index_file}')
        self.index = faiss.read_index(index_file)
        print(f'Loaded index of size {self.index.ntotal}')

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
            
        assert len(self.index_id_to_db_id) == self.index.ntotal, \
            'Deserialized index_id_to_db_id should match faiss index size'

    def _update_id_mapping(self, db_ids: List):
        """Update ID mapping for new entries"""
        self.index_id_to_db_id.extend(db_ids)

    def reset(self):
        """Reset the index"""
        self.index.reset()
        self.index_id_to_db_id = []
        print(f'Index reset, total data indexed {self.index.ntotal}')
