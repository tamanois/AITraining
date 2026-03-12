import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from transformers import TrainingArguments, Trainer
from dataset import SROIEDataset
import numpy as np
from seqeval.metrics import classification_report,f1_score

train_dataset_file_path = "/home/joel/Desktop/Sync/latitude-7420-shared/datasets/SROIE2019/sroie_train.json"
test_dataset_file_path = "/home/joel/Desktop/Sync/latitude-7420-shared/datasets/SROIE2019/sroie_test.json"
base_model = "microsoft/layoutlmv3-base"
model_save_dir = "/home/joel/projects/AITraining/SROIE/models/layoutlmv3-sroie2"


LABELS = [
    "O",
    "B-COMPANY","I-COMPANY",
    "B-DATE","I-DATE",
    "B-ADDRESS","I-ADDRESS",
    "B-TOTAL","I-TOTAL"
]

label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for l,i in label2id.items()}

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

model = LayoutLMv3ForTokenClassification.from_pretrained(
    base_model,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)


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

    learning_rate=5e-5,

    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    gradient_accumulation_steps=4,

    fp16=True,

    num_train_epochs=10,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_steps=10,

    load_best_model_at_end=True,

    dataloader_pin_memory=False
)

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,


    #tokenizer=processor,

    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model(model_save_dir)
processor.save_pretrained(model_save_dir)
