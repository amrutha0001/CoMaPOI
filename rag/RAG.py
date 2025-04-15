import pandas as pd
import numpy as np
import faiss
#from BCEmbedding import EmbeddingModel  # Make sure to correctly import your BCE embedding model
import jsonlines
import os
import re
import logging
logging.getLogger().setLevel(logging.ERROR)
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import json


class RAG_Finder:
    def __init__(self, data, num_test, top_k, max_id, args, mode):
        self.data = data
        self.num_test = num_test
        self.top_k = top_k
        self.max_id = max_id
        self.args = args
        self.mode = mode
        # Load POI information file
        self.poi_info_file = f'dataset_all/{self.data}/{self.data}_poi_info.csv'
        self.poi_data = self.load_poi_data(self.poi_info_file)

        # Initialize BCE embedding model
        self.bce_model = EmbeddingModel(model_name_or_path="models/bce_embedding")


        # Generate and save POI embedding database (if it doesn't exist yet)
        self.faiss_index_file = f'dataset_all/{self.data}/poi_faiss_index.bin'
        if not os.path.exists(self.faiss_index_file):
            self.init_poi_databank()
        else:
            self.faiss_index = self.load_faiss_index(self.faiss_index_file)

        # Build mapping from POI ID to index
        self.poi_id_to_index = {row['poi_id']: idx for idx, row in self.poi_data.iterrows()}
        self.index_to_poi_id = {idx: row['poi_id'] for idx, row in self.poi_data.iterrows()}
    def init_poi_databank(self):
        self.embeddings = self.generate_poi_embeddings(self.poi_data)
        self.faiss_index = self.create_faiss_index(self.embeddings)
        self.save_faiss_index(self.faiss_index, self.faiss_index_file)

    def load_poi_data(self, file_path):
        # Load POI information CSV file
        return pd.read_csv(file_path)

    def generate_poi_embeddings(self, poi_data):
        # Generate embeddings for POI descriptions
        embeddings = []
        for _, row in tqdm(poi_data.iterrows(), total=len(poi_data), desc="Generating POI embeddings"):
            description = f"POI ID: {row['poi_id']}, Category: {row['category']}, Location: {row['lat']}, {row['lon']}"
            embedding = self.bce_model.encode(description)
            embeddings.append(embedding)
        return np.array(embeddings)

    def create_faiss_index(self, embeddings):
        # Create FAISS index for fast similarity search
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index

    def save_faiss_index(self, index, file_path):
        # Save FAISS index to file
        faiss.write_index(index, file_path)

    def load_faiss_index(self, file_path):
        # Load FAISS index from file
        return faiss.read_index(file_path)

    def search_similar_pois(self, query, k=10):
        # Search for similar POIs based on query
        query_embedding = self.bce_model.encode(query)
        query_embedding = np.array([query_embedding])
        distances, indices = self.faiss_index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            poi_id = self.index_to_poi_id[idx]
            poi_info = self.poi_data.iloc[idx]
            results.append({
                'poi_id': poi_id,
                'category': poi_info['category'],
                'lat': poi_info['lat'],
                'lon': poi_info['lon'],
                'distance': distances[0][i]
            })

        return results

    def process_single_sample(self, sample):
        # Process a single sample to find candidate POIs
        try:
            # Extract user ID and current trajectory
            messages = sample.get('messages', [])
            user_id = None
            current_trajectory = None

            for msg in messages:
                if msg.get('role') == 'user':
                    content = msg.get('content', '')
                    user_match = re.search(r'"user_id":\s*"?(\d+)"?', content)
                    if user_match:
                        user_id = user_match.group(1)
                    current_trajectory = content

            if not user_id or not current_trajectory:
                return None

            # Generate query from current trajectory
            query = f"User trajectory: {current_trajectory}"

            # Search for similar POIs
            candidates = self.search_similar_pois(query, k=self.top_k)

            # Extract POI IDs
            candidate_poi_ids = [int(candidate['poi_id']) for candidate in candidates]

            return {
                'user_id': user_id,
                'candidates': candidate_poi_ids
            }

        except Exception as e:
            print(f"Error processing sample: {e}")
            return None

    def generate_candidates(self):
        # Generate candidate POIs for all samples
        # Load samples
        samples = []
        with open(f'dataset_all/{self.data}/{self.mode}/{self.data}_{self.mode}.jsonl', 'r') as f:
            for line in f:
                samples.append(json.loads(line))

        num_samples = min(self.num_test, len(samples))
        samples = samples[:num_samples]

        print(f"Generating candidates for {num_samples} samples...")

        # Process samples in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.args.batch_size) as executor:
            futures = [executor.submit(self.process_single_sample, sample) for sample in samples]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
                result = future.result()
                if result:
                    results.append(result)

        # Save results
        output_file = f'dataset_all/{self.data}/{self.mode}/{self.data}_{self.mode}_candidates.jsonl'
        with jsonlines.open(output_file, mode='w') as writer:
            for result in results:
                writer.write(result)

        print(f"Candidates saved to {output_file}")

        return output_file


class EmbeddingModel:
    """
    Placeholder for BCE embedding model.
    In a real implementation, this would be replaced with an actual embedding model.

    This class provides a deterministic embedding generation for demonstration purposes.
    When using in production, replace this with a proper embedding model like:
    - sentence-transformers
    - BCE embedding model
    - OpenAI embeddings API
    """
    def __init__(self, model_name_or_path):
        self.model_name_or_path = model_name_or_path
        print(f"Initialized embedding model from {model_name_or_path}")

        # In a real implementation, you would load the model here:
        # try:
        #     from sentence_transformers import SentenceTransformer
        #     self.model = SentenceTransformer(model_name_or_path)
        # except ImportError:
        #     print("Please install sentence-transformers: pip install sentence-transformers")
        #     raise

    def encode(self, text):
        """
        Generate a deterministic embedding for demonstration purposes.

        Args:
            text: Text to encode

        Returns:
            np.ndarray: Embedding vector
        """
        # In a real implementation, you would use the model to generate embeddings:
        # return self.model.encode(text)

        # For demonstration, generate a deterministic embedding based on the text hash
        import hashlib
        hash_object = hashlib.md5(text.encode())
        hash_hex = hash_object.hexdigest()

        # Convert hash to a numeric array (16 dimensions)
        embedding = np.array([int(hash_hex[i:i+2], 16) for i in range(0, 32, 2)], dtype=np.float32)

        # Normalize to unit length (important for cosine similarity)
        embedding = embedding / np.linalg.norm(embedding)

        return embedding
