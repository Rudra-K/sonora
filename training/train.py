# training/train.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import re
import torch.multiprocessing as mp

from dataset import MusicCapsDataset
from conditioning import TextConditioner
from model import GeneratorUNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrainingConfig:
    DATA_PATH = "/content/sonora/training/data/musiccaps"
    CHECKPOINT_DIR = Path("/content/drive/MyDrive/Sonora_Checkpoints")
    
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    
    RESUME_CHECKPOINT = "/content/drive/MyDrive/Sonora_Checkpoints/sonora_epoch_95.pth" 
    
    MODEL_CHANNELS = 32
    CHANNEL_MULTS = (1, 2, 4)
    TIMESTEPS = 1000

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        self.dataset = MusicCapsDataset(data_path=self.config.DATA_PATH)
        self.conditioner = TextConditioner(device=self.device)
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.config.BATCH_SIZE,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=0
        )
        
        self.model = GeneratorUNet().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)

        beta_start, beta_end = 0.0001, 0.02
        betas = torch.linspace(beta_start, beta_end, self.config.TIMESTEPS, device=self.device)
        alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.start_epoch = 0
        self.config.CHECKPOINT_DIR.mkdir(exist_ok=True)

        if self.config.RESUME_CHECKPOINT and Path(self.config.RESUME_CHECKPOINT).exists():
            logging.info(f"Resuming from checkpoint: {self.config.RESUME_CHECKPOINT}")
            checkpoint = torch.load(self.config.RESUME_CHECKPOINT, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            match = re.search(r'epoch_(\d+)', str(self.config.RESUME_CHECKPOINT))
            if match:
                self.start_epoch = int(match.group(1))
                logging.info(f"Resuming training from epoch {self.start_epoch + 1}")

    def collate_fn(self, batch):
        """Custom collate function to prepare batches for training."""
        spectrograms, captions = zip(*batch)
        
        mono_spectrograms = [torch.mean(spec, dim=0, keepdim=True) for spec in spectrograms]
        batched_spectrograms = torch.stack(mono_spectrograms)
        
        batched_embeddings = self.conditioner.encode(list(captions))
        return batched_spectrograms, batched_embeddings

    def _add_noise(self, x_start, t):
        """Adds noise to the spectrograms according to the timestep t."""
        noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.alphas_cumprod.sqrt()[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod)[t].sqrt().view(-1, 1, 1, 1)

        noisy_spectrogram = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_spectrogram, noise

    def train(self):
        """The main training loop."""
        logging.info("Starting training...")
        for epoch in range(self.start_epoch, self.config.NUM_EPOCHS):
            progress_bar = tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            for step, (spectrograms, embeddings) in enumerate(progress_bar):
                self.optimizer.zero_grad()
                
                spectrograms = spectrograms.to(self.device)
                embeddings = embeddings.to(self.device)
                
                t = torch.randint(0, self.config.TIMESTEPS, (spectrograms.shape[0],), device=self.device).long()
                
                noisy_spectrograms, noise = self._add_noise(spectrograms, t)
                
                predicted_noise = self.model(noisy_spectrograms, t, embeddings)
                
                loss = F.mse_loss(predicted_noise, noise)
                
                loss.backward()
                self.optimizer.step()
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            checkpoint_path = self.config.CHECKPOINT_DIR / f"sonora_epoch_{epoch+1}.pth"
            torch.save(self.model.state_dict(), checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    config = TrainingConfig()
    trainer = Trainer(config)
    trainer.train()