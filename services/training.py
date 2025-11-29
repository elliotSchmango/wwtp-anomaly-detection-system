import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import logging
import time
import glob

from core.state import StateManager
from config.settings import settings

logger = logging.getLogger("State.TrainAE")

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 2, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#Access dataset
class ZoneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Recursively find all images in data/zone_*
        self.image_paths = glob.glob(str(root_dir / "zone_*" / "*.jpg"))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, img #input and target are same
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            #return dummy tensor if error
            dummy = torch.zeros(3, settings.AE_IMG_SIZE, settings.AE_IMG_SIZE)
            return dummy, dummy

#Training:
class TrainingTask:
    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.state = StateManager()
        self.epochs = 20
        self.batch_size = 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        try:
            logger.info("Starting Training Task...")
            
            #preprocess data
            transform = transforms.Compose([
                transforms.Resize((settings.AE_IMG_SIZE, settings.AE_IMG_SIZE)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1), #light augmentations
                transforms.ToTensor(),
            ])
            
            dataset = ZoneDataset(settings.DATA_DIR, transform)
            if len(dataset) == 0:
                raise RuntimeError("No training data found! Run Calibration first.")
                
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self.state.update(log=f"Found {len(dataset)} images. Training on {self.device}...")

            model = ConvAutoencoder().to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            #training loop
            model.train()
            for epoch in range(self.epochs):
                if self.stop_event.is_set(): break
                
                total_loss = 0
                for imgs, _ in dataloader:
                    imgs = imgs.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(imgs)
                    loss = criterion(outputs, imgs)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * imgs.size(0)

                avg_loss = total_loss / len(dataset)
                
                #update UI
                progress = int(((epoch + 1) / self.epochs) * 100)
                msg = f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.6f}"
                logger.info(msg)
                self.state.update(progress=progress, status=msg)

            #export as ONNX format
            if not self.stop_event.is_set():
                self.state.update(status="Exporting to ONNX...", progress=99)
                self._export_onnx(model)
                self.state.update(status="Training Complete", progress=100)
                logger.info("Training and Export Complete.")
                time.sleep(3)

        except Exception as e:
            logger.error(f"Training Failed: {e}")
            self.state.update(status=f"Training Error: {str(e)}", mode="ERROR")
        finally:
            if not self.state.get_snapshot()['mode'] == "ERROR":
                self.state.set_mode("IDLE")

    #convert PyTorch model to ONNX for sentry loop
    def _export_onnx(self, model):
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        onnx_path = settings.MODELS_DIR / "autoencoder.onnx"
        
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