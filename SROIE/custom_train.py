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
TRAIN_BATCH_SIZE = 4 # Adjust based on your GPU memory
LOG_THRESHOLD = 100  # Log every x samples
NUM_EPOCHS = 10 # Number of epochs for training
PATIENCE = 2 # Early stopping patience in case of no improvement in validation accuracy

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
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
    persistent_workers=True)

val_loader = DataLoader(
    test_dataset,
    batch_size=TRAIN_BATCH_SIZE,
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

def train_epoch(model, dataloader, optimizer, epoch):

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
        if processed % LOG_THRESHOLD == 0:
            logging.info(f"TRAINING | Epoch {epoch} | Processed {processed}/{len(dataloader.dataset)} samples")

    return total_loss / len(dataloader), total_acc / len(dataloader)

def validate_epoch(model, dataloader, epoch):

    model.eval()

    total_loss = 0
    total_acc = 0
    logging.info(f"Validating on {len(dataloader.dataset)} samples with batch size {dataloader.batch_size}...")
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
            processed += len(batch["input_ids"])
            if processed % LOG_THRESHOLD == 0:
                logging.info(f"VALIDATING | Epoch {epoch} | Processed {processed}/{len(dataloader.dataset)} samples")

    return total_loss / len(dataloader), total_acc / len(dataloader)


######## ###### TRAINING LOOP ###############


best_val_acc = 0
best_epoch = 0
no_improve_counter = 0

for epoch in range(1, NUM_EPOCHS + 1):

    logging.info(f"\nEpoch {epoch}")

    train_loss, train_acc = train_epoch(model, train_loader, optimizer, epoch)
    val_loss, val_acc = validate_epoch(model, val_loader, epoch)

    logging.info(f"Train Loss: {train_loss:.4f}")
    logging.info(f"Train Acc : {train_acc:.4f}")
    logging.info(f"Val Loss  : {val_loss:.4f}")
    logging.info(f"Val Acc   : {val_acc:.4f}")

    # Early stopping for overfitting and increasing validation loss
    OVERFITTING_LOSS_THRESHOLD = 0.1  # Example threshold, adjust as needed
    INCREASING_VAL_LOSS_PATIENCE = 2
    if epoch == 1:
        prev_val_loss = val_loss
        val_loss_increase_counter = 0
    else:
        if train_loss < OVERFITTING_LOSS_THRESHOLD and val_loss > prev_val_loss:
            val_loss_increase_counter += 1
            logging.warning(f"Validation loss increased! Counter: {val_loss_increase_counter}")
        else:
            val_loss_increase_counter = 0
        prev_val_loss = val_loss
        if val_loss_increase_counter >= INCREASING_VAL_LOSS_PATIENCE:
            logging.warning("\n⛔ Early stopping: validation loss increased for consecutive epochs (possible overfitting)")
            break

    # Save checkpoint
    save_path = f"{model_save_dir}/layoutlmv3_epoch_{epoch}.pt"
    torch.save(model.state_dict(), save_path)
    #save config and processor too
    model.config.save_pretrained(model_save_dir)
    processor.save_pretrained(model_save_dir)

    logging.info(f"Model saved → {save_path}")

    # Check best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        no_improve_counter = 0
        logging.info(f"🔥 NEW BEST MODEL: Epoch {best_epoch}")
        logging.info(f"Best Val Accuracy: {best_val_acc:.4f}")
        torch.save(model.state_dict(), model_save_dir + "/layoutlmv3_best.pt")
        model.config.save_pretrained(model_save_dir)
        processor.save_pretrained(model_save_dir)
    else:
        no_improve_counter += 1

    # Early stopping
    if no_improve_counter >= PATIENCE:
        logging.info("\n⛔ Early stopping triggered")
        logging.info(f"Best model was at epoch {best_epoch}")
        break