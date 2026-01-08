import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import logging
import time
import glob
import math

from core.state import StateManager
from config.settings import settings

logger = logging.getLogger("State.TrainAE")

#structural similarity index (SSIM) loss
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        C1 = 0.01**2
        C2 = 0.03**2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        #robust device handling
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            window = window.to(img1.device).type_as(img1)
            self.window = window
            self.channel = channel
            
        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

#standard AE architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 2, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#access dataset
#access dataset
class ZoneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # FIX: Look for both .jpg and .png files recursively
        # We use a set to avoid duplicates if any weird overlap happens
        jpgs = glob.glob(str(root_dir / "zone_*" / "*.jpg"), recursive=True)
        pngs = glob.glob(str(root_dir / "zone_*" / "*.png"), recursive=True)
        
        self.image_paths = sorted(jpgs + pngs)
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            return torch.zeros(3, settings.AE_IMG_SIZE, settings.AE_IMG_SIZE)

#training task
class TrainingTask:
    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.state = StateManager()
        self.mode = settings.TRAINING_MODE.lower() # "mse" or "ssim"
        self.epochs = 25
        self.batch_size = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        try:
            logger.info(f"Starting Training Task ({self.mode.upper()})...")
            
            #preprocess data
            transform = transforms.Compose([
                transforms.Resize((settings.AE_IMG_SIZE, settings.AE_IMG_SIZE)),
                transforms.ToTensor(),
            ])
            
            dataset = ZoneDataset(settings.DATA_DIR, transform)
            if len(dataset) == 0:
                raise RuntimeError("No training data found! Run Calibration first.")
                
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self.state.update(log=f"Found {len(dataset)} images. Training {self.mode.upper()} on {self.device}...")

            model = ConvAutoencoder().to(self.device)
            
            #metric definitions
            mse_criterion = nn.MSELoss() if self.mode == "mse" else nn.L1Loss()
            ssim_criterion = SSIMLoss().to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            #training loop
            model.train()
            for epoch in range(self.epochs):
                if self.stop_event.is_set(): break
                
                total_loss = 0
                for img_clean in dataloader:
                    img_clean = img_clean.to(self.device)
                    
                    optimizer.zero_grad()

                    if self.mode == "ssim":
                        #denoising injection: add noise to input
                        noise = torch.randn_like(img_clean) * 0.05
                        img_noisy = img_clean + noise
                        img_noisy = torch.clamp(img_noisy, 0., 1.)
                        outputs = model(img_noisy)
                        
                        #loss = 80% SSIM + 20% L1
                        loss_ssim = ssim_criterion(outputs, img_clean)
                        loss_l1 = mse_criterion(outputs, img_clean)
                        loss = (0.8 * loss_ssim) + (0.2 * loss_l1)
                    else:
                        #standard MSE baseline
                        outputs = model(img_clean)
                        loss = mse_criterion(outputs, img_clean)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * img_clean.size(0)

                avg_loss = total_loss / len(dataset)
                
                #update UI
                progress = int(((epoch + 1) / self.epochs) * 100)
                msg = f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.6f}"
                logger.info(msg)
                self.state.update(progress=progress, status=msg)

            #export as ONNX format
            if not self.stop_event.is_set():
                filename = f"autoencoder.onnx"
                self.state.update(status=f"Exporting to {filename}...", progress=99)
                self._export_onnx(model, filename)
                self.state.update(status="Training Complete", progress=100)
                time.sleep(3)

        except Exception as e:
            logger.error(f"Training Failed: {e}")
            self.state.update(status=f"Training Error: {str(e)}", mode="ERROR")
        finally:
            if not self.state.get_snapshot()['mode'] == "ERROR":
                self.state.set_mode("IDLE")

    #convert PyTorch model to ONNX for sentry loop
    def _export_onnx(self, model, filename):
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        onnx_path = settings.MODELS_DIR / filename
        model.eval()
        dummy_input = torch.randn(1, 3, settings.AE_IMG_SIZE, settings.AE_IMG_SIZE).to(self.device)
        torch.onnx.export(
            model, 
            dummy_input, 
            str(onnx_path), 
            input_names=["input"], 
            output_names=["output"], 
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, 
            opset_version=11
        )
        logger.info(f"Model saved to {onnx_path}")