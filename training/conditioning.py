import torch
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextConditioner:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        self.device = torch.device(device)
        logging.info(f"Loading sentence-transformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        logging.info("Model loaded successfully.")

    def encode(self, captions):
        embeddings = self.model.encode(captions, convert_to_tensor=True)
        return embeddings.to(self.device)

if __name__ == '__main__':
    conditioner = TextConditioner()
    
    sample_caption = "A low quality recording of a ballad song with sustained strings and a mellow piano melody."

    embedding = conditioner.encode(sample_caption)

    logging.info("\n--- Inspecting Conditioning Output ---")
    logging.info(f"Input Caption: '{sample_caption}'")
    logging.info(f"Output Embedding Shape: {embedding.shape}")
    logging.info(f"Output Embedding (first 5 values): {embedding[:5]}")
    logging.info("----------------------------------")

    batch_captions = ["A fast-paced electronic track.", "A slow, sad acoustic guitar song."]
    batch_embeddings = conditioner.encode(batch_captions)
    logging.info("\n--- Inspecting Batched Output ---")
    logging.info(f"Batched Output Embedding Shape: {batch_embeddings.shape}")
    logging.info("----------------------------------")