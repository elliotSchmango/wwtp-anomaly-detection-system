import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import logging
from pathlib import Path
import time
from core.state import StateManager
from config.settings import settings

logger = logging.getLogger("State.TrainClassifier")

class ClassifierTrainingTask:
    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.state = StateManager()
        self.epochs = 15
        self.batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #paths
        self.data_dir = settings.DATA_DIR / "classifier_data"
        self.model_save_path = settings.MODELS_DIR / "classifier.onnx"
        
        #ignore testing folder from older project iterations
        self.ignored_folders = ["test_realworld"]

    def run(self):
        try:
            logger.info("Starting Classifier Training...")
            self.state.update(status="Preparing Classifier Data...", progress=0)

            #data transforms
            data_transforms = transforms.Compose([
                transforms.Resize((settings.CLS_IMG_SIZE, settings.CLS_IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            #load dataset
            if not self.data_dir.exists():
                raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

            classes = [d.name for d in self.data_dir.iterdir() 
                       if d.is_dir() and d.name not in self.ignored_folders]
            classes.sort()
            
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            logger.info(f"Training on classes: {classes}")

            full_dataset = datasets.ImageFolder(
                str(self.data_dir), 
                transform=data_transforms,
                is_valid_file=lambda path: Path(path).parent.name in classes
            )
            
            full_dataset.class_to_idx = class_to_idx
            full_dataset.classes = classes

            dataloader = torch.utils.data.DataLoader(
                full_dataset, batch_size=self.batch_size, shuffle=True
            )

            self.state.update(log=f"Found {len(full_dataset)} images across {len(classes)} classes.")

            #setup model
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            
            #freeze params for faster training
            for param in model.parameters():
                param.requires_grad = False
                
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(classes))
            model = model.to(self.device)

            criterion = nn.CrossEntropyLoss()
            # Only optimize the final layer
            optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

            #training loop
            for epoch in range(self.epochs):
                if self.stop_event.is_set(): break
                
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                epoch_loss = running_loss / len(full_dataset)
                epoch_acc = correct / total
                
                msg = f"Clf Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2%}"
                logger.info(msg)
                
                #UI updates
                progress = int(((epoch + 1) / self.epochs) * 100)
                self.state.update(progress=progress, status=msg)

            #export to ONNX
            if not self.stop_event.is_set():
                self.state.update(status="Exporting Classifier to ONNX...", progress=99)
                self._export_onnx(model)
                self.state.update(status="Classifier Training Complete", progress=100)
                logger.info(f"Classifier saved to {self.model_save_path}")
                time.sleep(3)

        except Exception as e:
            logger.error(f"Classifier Training Failed: {e}")
            self.state.update(status=f"Clf Error: {str(e)}", mode="ERROR")
        finally:
            if not self.state.get_snapshot()['mode'] == "ERROR":
                self.state.set_mode("IDLE")

    def _export_onnx(self, model):
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.eval()
        
        dummy_input = torch.randn(1, 3, settings.CLS_IMG_SIZE, settings.CLS_IMG_SIZE).to(self.device)
        
        torch.onnx.export(
            model,
            dummy_input,
            str(self.model_save_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=11
        )