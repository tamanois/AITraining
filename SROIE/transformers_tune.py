import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from transformers import TrainingArguments, Trainer
from dataset import SROIEDataset
import numpy as np
from seqeval.metrics import classification_report,f1_score

train_dataset_file_path = "/home/joel/Desktop/Sync/latitude-7420-shared/datasets/SROIE2019/sroie_train.json"
test_dataset_file_path = "/home/joel/Desktop/Sync/latitude-7420-shared/datasets/SROIE2019/sroie_test.json"
#base_model = "microsoft/layoutlmv3-base"
base_model = "Theivaprakasham/layoutlmv3-finetuned-sroie"
model_save_dir = "/home/joel/projects/AITraining/SROIE/models/layoutlmv3-sroie-tuned"

###### FINE TUNE VALUES ######
LEARNING_RATE = 1e-5
EPOCHS = 5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
FREEZE = False
BATCH_SIZE = 4
##############################

model = LayoutLMv3ForTokenClassification.from_pretrained(base_model)

# Use the model's existing label mapping
id2label = model.config.id2label
label2id = model.config.label2id
LABELS = [id2label[i] for i in range(len(id2label))]

print("Label to ID mapping:", label2id)

processor = LayoutLMv3Processor.from_pretrained(
    base_model,
    apply_ocr=False, 
)

# Manually override the image resizing parameters
processor.image_processor.size = {"height": 224, "width": 224} 
# Note: LayoutLMv3 works best with square images

train_dataset = SROIEDataset(
    train_dataset_file_path,
    processor,
    label2id,
)

test_dataset = SROIEDataset(
    test_dataset_file_path,
    processor,
    label2id
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

if FREEZE:
    print("Freezing base model parameters...")
    for param in model.base_model.parameters():
        param.requires_grad = False


def compute_metrics(p):

    predictions, labels = p

    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [LABELS[l] for l in label if l != -100]
        for label in labels
    ]

    true_preds = [
        [LABELS[p] for (p,l) in zip(pred,label) if l != -100]
        for pred,label in zip(predictions,labels)
    ]

    return {
        "f1": f1_score(true_labels,true_preds)
    }


training_args = TrainingArguments(
    output_dir= model_save_dir,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    fp16=True,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    dataloader_pin_memory=False,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
)

trainer = Trainer(

    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    #tokenizer=processor,
    compute_metrics=compute_metrics
)

print("Starting training...")
trainer.train()

trainer.save_model(model_save_dir)
processor.save_pretrained(model_save_dir)
print(f"End Training. Model and processor saved to {model_save_dir}")
