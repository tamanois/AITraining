from transformers import Trainer
import torch

trainer = Trainer(
    model=model
)

metrics = trainer.evaluate(test_dataset)

print(metrics)