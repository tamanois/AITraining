import json
from PIL import Image
from torch.utils.data import Dataset
from transformers import LayoutLMv3Processor

class SROIEDataset(Dataset):

    def __init__(self, json_path, processor, label2id):

        with open(json_path) as f:
            self.data = json.load(f)

        self.processor = processor
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        image = Image.open(item["image_path"]).convert("RGB")

        encoding = self.processor(
            image,
            item["tokens"],
            boxes=item["bboxes"],
            word_labels=[self.label2id[l] for l in item["labels"]],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        encoding = {k:v.squeeze() for k,v in encoding.items()}

        return encoding