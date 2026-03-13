
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch
from paddleocr import PaddleOCR
import numpy as np

image_path = "/home/joel/Desktop/Sync/latitude-7420-shared/imgs/receipts/receipt.png"
model_name = "/home/joel/projects/AITraining/SROIE/models/layoutlmv3-sroie-tuned"
#model_name = "Theivaprakasham/layoutlmv3-finetuned-sroie"

# Load processor and model
processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)

model.eval()

# Load image
image = Image.open(image_path).convert("RGB")
width, height = image.size

print("Image original size:", width, height)

# Run PaddleOCR
ocr = PaddleOCR(
       layout=True,
    table=True,
    ocr=True,
    recovery=False,
    use_pdf2docx_api=False,
    invert=False,
    binarize=False,
    alphacolor=(255, 255, 255),
    lang='en',
    det=True,
    rec=True,
    type='ocr',
    use_angle_cls=True,
    ocr_version='PP-OCRv4',
    structure_version='PP-StructureV2',
    enable_mkldnn=False
)

# Reduce image size for faster OCR processing
max_dim = 1500
if max(width, height) > max_dim:
    scale = max_dim / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

print("Image resized for OCR:", image.size)

ocr_result = ocr.ocr(np.array(image), cls=True)

curr_width, curr_height = image.size


# Extract words and bounding boxes
words = []
boxes = []
for line in ocr_result[0]:
    text = line[1][0]
    box = line[0]
    # box: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    # Get min/max for rectangle
    points = np.array(box)
    xmin = int(min(points[:, 0]))
    ymin = int(min(points[:, 1]))
    xmax = int(max(points[:, 0]))
    ymax = int(max(points[:, 1]))
    # Normalize to 0-1000 as required by LayoutLMv3
    norm_box = [
        int(1000 * xmin / curr_width),
        int(1000 * ymin / curr_height),
        int(1000 * xmax / curr_width),
        int(1000 * ymax / curr_height)
    ]
    words.append(text)
    boxes.append(norm_box)


# Encode image, words, and boxes
encoding = processor(image, text=words, boxes=boxes, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**encoding)

predictions = outputs.logits.argmax(-1)

# Convert ids to labels
id2label = model.config.id2label
print("Labels:", [id2label[i.item()] for i in predictions[0]])
pred_labels = [id2label[p.item()] for p in predictions[0]]

# Get tokens
tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

# Extract entities
entities = {
    "company": [],
    "address": [],
    "date": [],
    "total": []
}

for token, label in zip(tokens, pred_labels):
    # if token in ["[CLS]", "[SEP]", "[PAD]"]:
    #     continue
    token = token.replace("Ġ", "").replace("▁", "")
    if label == "B-COMPANY" or label == "I-COMPANY":
        entities["company"].append(token.strip())
    elif label == "B-ADDRESS" or label == "I-ADDRESS":
        entities["address"].append(token.strip())
    elif label == "B-DATE" or label == "I-DATE":
        entities["date"].append(token.strip())
    elif label == "B-TOTAL" or label == "I-TOTAL":
        entities["total"].append(token.strip())

# Join tokens into strings
result = {
    "company": " ".join(entities["company"]),
    "address": " ".join(entities["address"]),
    "date": " ".join(entities["date"]),
    "total": " ".join(entities["total"])
}

print(result)