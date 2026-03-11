from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch

processor = LayoutLMv3Processor.from_pretrained("models/layoutlmv3-sroie")
model = LayoutLMv3ForTokenClassification.from_pretrained("models/layoutlmv3-sroie")

image = Image.open("receipt.jpg").convert("RGB")

encoding = processor(image, return_tensors="pt")

outputs = model(**encoding)

predictions = outputs.logits.argmax(-1)