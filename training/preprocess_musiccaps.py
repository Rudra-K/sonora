import pandas as pd
import torch
import torchaudio
import yt_dlp
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CSV_PATH = "musiccaps-public.csv"

SAVE_DIR = Path("training/data/musiccaps")
TEMP_AUDIO_DIR = Path("training/data/temp_audio")
SAMPLE_RATE = 44100

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

def download_and_clip(ytid, start_s, end_s):
    output_template = TEMP_AUDIO_DIR / f"{ytid}.%(ext)s"
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_template),
        'quiet': True,
        'nocheckcertificate': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(f'https://www.youtube.com/watch?v={ytid}', download=True)
            downloaded_file = ydl.prepare_filename(info_dict)

        if not downloaded_file or not os.path.exists(downloaded_file):
             logging.error(f"File not found after download for {ytid}")
             return None, None

        waveform, sr = torchaudio.load(downloaded_file)
        
        start_frame = int(start_s * sr)
        end_frame = int(end_s * sr)
        clipped_waveform = waveform[:, start_frame:end_frame]
        
        if clipped_waveform.numel() == 0:
            logging.warning(f"Clipped waveform for {ytid} is empty (bad timestamps?). Skipping.")
            os.remove(downloaded_file)
            return None, None
            
        os.remove(downloaded_file)
        return clipped_waveform, sr

    except Exception as e:
        logging.error(f"Failed to process {ytid}: {e}")
        return None, None

def main():
    if not Path(CSV_PATH).exists():
        logging.error(f"CSV file not found at: {CSV_PATH}")
        logging.error("Please update the CSV_PATH variable in the script.")
        return

    logging.info(f"Creating save directories in {SAVE_DIR}")
    spectrogram_dir = SAVE_DIR / "spectrograms"
    caption_dir = SAVE_DIR / "captions"
    spectrogram_dir.mkdir(parents=True, exist_ok=True)
    caption_dir.mkdir(parents=True, exist_ok=True)
    TEMP_AUDIO_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    total_samples = len(df)
    logging.info(f"Found {total_samples} samples in CSV.")
    
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )

    successful_samples = 0
    for i, row in df.iterrows():
        ytid = row['ytid']
        caption = row['caption']
        
        spec_save_path = spectrogram_dir / f"{ytid}.pt"
        caption_save_path = caption_dir / f"{ytid}.txt"

        if spec_save_path.exists() and caption_save_path.exists():
            continue

        clipped_waveform, sr = download_and_clip(ytid, row['start_s'], row['end_s'])
        
        if clipped_waveform is None:
            continue

        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            resampled_waveform = resampler(clipped_waveform)
        else:
            resampled_waveform = clipped_waveform
        
        mel_spec = mel_spectrogram_transform(resampled_waveform)
        
        torch.save(mel_spec, spec_save_path)
        with open(caption_save_path, 'w', encoding='utf-8') as f:
            f.write(caption)
            
        successful_samples += 1
        if successful_samples > 0 and successful_samples % 25 == 0:
            logging.info(f"Processed {successful_samples} new samples successfully ({i + 1}/{total_samples} total reviewed).")

    logging.info(f"Preprocessing complete. Successfully processed {successful_samples} new samples.")

if __name__ == "__main__":
    main()