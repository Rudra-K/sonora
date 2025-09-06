import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MusicCapsDataset(Dataset):
    
    def __init__(self, data_path: str, target_length: int = 862):
        self.data_path = Path(data_path)
        self.spectrogram_dir = self.data_path / "spectrograms"
        self.caption_dir = self.data_path / "captions"
        self.target_length = target_length

        self.spectrogram_files = sorted([p for p in self.spectrogram_dir.glob("*.pt")])

        if not self.spectrogram_files:
            raise FileNotFoundError(f"No spectrogram files found in {self.spectrogram_dir}")

    def __len__(self):
        return len(self.spectrogram_files)

    def __getitem__(self, idx):
        spec_path = self.spectrogram_files[idx]
        
        caption_path = self.caption_dir / f"{spec_path.stem}.txt"
        
        spectrogram = torch.load(spec_path)
        
        current_length = spectrogram.shape[2]
        if current_length > self.target_length:
            spectrogram = spectrogram[:, :, :self.target_length]
        elif current_length < self.target_length:
            padding_needed = self.target_length - current_length
            spectrogram = torch.nn.functional.pad(spectrogram, (0, padding_needed))
        
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
            
        return spectrogram, caption

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from conditioning import TextConditioner # Import our new class

    DATA_PATH = "training/data/musiccaps"
    BATCH_SIZE = 4
    
    conditioner = TextConditioner()

    def collate_fn(batch):
        spectrograms = []
        captions = []
        for spec, cap in batch:
            if spec.shape[0] == 2:
                spec = torch.mean(spec, dim=0, keepdim=True)
            spectrograms.append(spec)
            captions.append(cap)
        
        batched_spectrograms = torch.stack(spectrograms)
        
        batched_embeddings = conditioner.encode(captions)
        
        return batched_spectrograms, batched_embeddings

    dataset = MusicCapsDataset(data_path=DATA_PATH)
    logging.info(f"Successfully found {len(dataset)} samples.")
    
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True
    )
    
    first_batch = next(iter(data_loader))
    batched_spectrograms, batched_embeddings = first_batch
    
    logging.info("\n--- Inspecting First Batch from DataLoader ---")
    logging.info(f"Batch of spectrograms shape: {batched_spectrograms.shape}")
    logging.info(f"Batch of embeddings shape: {batched_embeddings.shape}")
    logging.info("------------------------------------------")