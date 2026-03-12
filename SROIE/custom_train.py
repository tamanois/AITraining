import torch
from torch.optim import AdamW
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
import numpy as np
from dataset import SROIEDataset
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info(f"Using device: {device}")

train_dataset_file_path = "/home/joel/Desktop/Sync/latitude-7420-shared/datasets/SROIE2019/sroie_train.json"
test_dataset_file_path = "/home/joel/Desktop/Sync/latitude-7420-shared/datasets/SROIE2019/sroie_test.json"
base_model = "microsoft/layoutlmv3-base"
model_save_dir = "/home/joel/projects/AITraining/SROIE/models/torch-layoutlmv3-sroie"

LABELS = [
"O",
"B-COMPANY","I-COMPANY",
"B-DATE","I-DATE",
"B-ADDRESS","I-ADDRESS",
"B-TOTAL","I-TOTAL"
]

label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for l,i in label2id.items()}

logging.info("Loading processor and model...")
processor = LayoutLMv3Processor.from_pretrained(
    base_model,
    apply_ocr=False
)

# Manually override the image resizing parameters
processor.image_processor.size = {"height": 224, "width": 224} 
    
logging.info("Processor loaded successfully.")

logging.info("Loading training datasets...")
train_dataset = SROIEDataset(
    train_dataset_file_path,
    processor,
    label2id
)
logging.info(f"Training dataset loaded with {len(train_dataset)} samples.")

logging.info("Loading test datasets...")
test_dataset = SROIEDataset(
    test_dataset_file_path,
    processor,
    label2id
)
logging.info(f"Test dataset loaded with {len(test_dataset)} samples.")

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
    persistent_workers=True)

val_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=False,
    persistent_workers=True
)



model = LayoutLMv3ForTokenClassification.from_pretrained(
    base_model,
    num_labels=len(LABELS)
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)



############### FUNCTIONS ###############
def compute_accuracy(preds, labels):

    preds = preds.flatten()
    labels = labels.flatten()

    mask = labels != -100

    preds = preds[mask]
    labels = labels[mask]

    correct = (preds == labels).sum()

    return correct / len(labels)

def train_epoch(model, dataloader, optimizer):

    model.train()

    total_loss = 0
    total_acc = 0
    logging.info(f"Training on {len(dataloader.dataset)} samples with batch size {dataloader.batch_size}...")
    processed = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        bbox = batch["bbox"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=-1)

        acc = compute_accuracy(
            preds.detach().cpu().numpy(),
            labels.detach().cpu().numpy()
        )

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        total_acc += acc
        processed += len(batch["input_ids"])
        logging.info(f"Processed {processed}/{len(dataloader.dataset)} samples")

    return total_loss / len(dataloader), total_acc / len(dataloader)

def validate_epoch(model, dataloader):

    model.eval()

    total_loss = 0
    total_acc = 0

    with torch.no_grad():

        processed = 0
        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            bbox = batch["bbox"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)

            acc = compute_accuracy(
                preds.cpu().numpy(),
                labels.cpu().numpy()
            )

            total_loss += loss.item()
            total_acc += acc

    return total_loss / len(dataloader), total_acc / len(dataloader)


######## ###### TRAINING LOOP ###############

num_epochs = 10
patience = 5

best_val_acc = 0
best_epoch = 0
no_improve_counter = 0

for epoch in range(1, num_epochs + 1):

    logging.info(f"\nEpoch {epoch}")

    train_loss, train_acc = train_epoch(model, train_loader, optimizer)

    val_loss, val_acc = validate_epoch(model, val_loader)

    logging.info(f"Train Loss: {train_loss:.4f}")
    logging.info(f"Train Acc : {train_acc:.4f}")
    logging.info(f"Val Loss  : {val_loss:.4f}")
    logging.info(f"Val Acc   : {val_acc:.4f}")

    # Save checkpoint
    save_path = f"layoutlmv3_epoch_{epoch}.pt"
    torch.save(model.state_dict(), save_path)

    logging.info(f"Model saved → {save_path}")

    # Check best model
    if val_acc > best_val_acc:

        best_val_acc = val_acc
        best_epoch = epoch

        no_improve_counter = 0

        logging.info("🔥 NEW BEST MODEL")
        logging.info(f"Best Val Accuracy: {best_val_acc:.4f}")

        torch.save(model.state_dict(), "layoutlmv3_best.pt")

    else:

        no_improve_counter += 1

    # Early stopping
    if no_improve_counter >= patience:

        logging.info("\n⛔ Early stopping triggered")
        logging.info(f"Best model was at epoch {best_epoch}")
        break