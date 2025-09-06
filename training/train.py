import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm

# Import our custom modules
from dataset import MusicCapsDataset
from conditioning import TextConditioner
from model import GeneratorUNet

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Training Configuration ---
class TrainingConfig:
    DATA_PATH = "training/data/musiccaps"
    CHECKPOINT_DIR = Path("training/checkpoints")
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4 # Adjust based on your CPU's memory
    NUM_EPOCHS = 100 # Total number of training epochs
    
    # Model parameters
    MODEL_CHANNELS = 64
    CHANNEL_MULTS = (1, 2, 4)
    
    # Diffusion parameters
    TIMESTEPS = 1000

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cpu") # Forcing CPU as per project goal
        
        logging.info("Initializing training components...")
        
        # --- Data Pipeline ---
        self.dataset = MusicCapsDataset(data_path=self.config.DATA_PATH)
        self.conditioner = TextConditioner()
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.config.BATCH_SIZE,
            collate_fn=self.collate_fn,
            shuffle=True
        )
        
        # --- Model and Optimizer ---
        self.model = GeneratorUNet(
            model_channels=self.config.MODEL_CHANNELS,
            channel_mults=self.config.CHANNEL_MULTS
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)

        # --- Diffusion Noise Schedule ---
        # We'll use a simple linear noise schedule
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, self.config.TIMESTEPS, device=self.device)
        alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)

        # Ensure checkpoint directory exists
        self.config.CHECKPOINT_DIR.mkdir(exist_ok=True)
        logging.info("Initialization complete.")

    def collate_fn(self, batch):
        """Custom collate function to prepare batches for training."""
        spectrograms, captions = zip(*batch)
        
        # Convert stereo to mono by averaging channels
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
        for epoch in range(self.config.NUM_EPOCHS):
            progress_bar = tqdm(self.data_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            for step, (spectrograms, embeddings) in enumerate(progress_bar):
                self.optimizer.zero_grad()
                
                spectrograms = spectrograms.to(self.device)
                embeddings = embeddings.to(self.device)
                
                # 1. Sample a random timestep for each spectrogram in the batch
                t = torch.randint(0, self.config.TIMESTEPS, (spectrograms.shape[0],), device=self.device).long()
                
                # 2. Create the noisy version of the spectrogram
                noisy_spectrograms, noise = self._add_noise(spectrograms, t)
                
                # 3. Get the model's prediction of the noise
                predicted_noise = self.model(noisy_spectrograms, t, embeddings)
                
                # 4. Calculate the loss
                loss = F.mse_loss(predicted_noise, noise)
                
                # 5. Backpropagate and update weights
                loss.backward()
                self.optimizer.step()
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # --- Save a checkpoint at the end of each epoch ---
            checkpoint_path = self.config.CHECKPOINT_DIR / f"sonora_epoch_{epoch+1}.pth"
            torch.save(self.model.state_dict(), checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    config = TrainingConfig()
    trainer = Trainer(config)
    trainer.train()