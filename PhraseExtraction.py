import torch
import numpy as np
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import DataLoader
from transformers import AdamW
import jsonlines
from sklearn.metrics import classification_report

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Create a mapping for labels
labels = ["O", "B-A", "I-A", "B-B", "I-B"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}

# Define the custom NER dataset
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.max_len = 256

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words, lbls = convert_to_bio(item)
        return {"words": words, "labels": lbls}

# Convert a given dataset entry to BIO format
def convert_to_bio(entry):
    text = entry['text']
    phrases = entry['phrases']
    words = tokenizer.tokenize(text)
    labels = ['O'] * len(words)

    for phrase_info in phrases:
        start, end, ptype = phrase_info['start'], phrase_info['end'], phrase_info['type']
        if start == end:
            labels[start] = f"B-{ptype}"
        else:
            labels[start] = f"B-{ptype}"
            for i in range(start + 1, end):
                labels[i] = f"I-{ptype}"

    input_ids = tokenizer.convert_tokens_to_ids(words)
    label_ids = [label2id[label] for label in labels]

    padding_length = max_len - len(input_ids)
    input_ids.extend([0] * padding_length)
    label_ids.extend([-100] * padding_length)  # -100 is ignored for CrossEntropy loss

    return torch.tensor(input_ids), torch.tensor(label_ids)

# Load data
with jsonlines.open('data.jsonl') as reader:
    data = [line for line in reader]

# Dataset and DataLoader for training data
max_len = 256
dataset = NERDataset(data)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model setup
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(labels))
optim = AdamW(model.parameters(), lr=5e-5)

def compute_metrics(true_labels, pred_labels):

    # Debugging statements
    print("Number of Batches in true_labels:", len(true_labels))
    print("Number of Batches in pred_labels:", len(pred_labels))
    print("True Labels Shapes:", [batch.shape for batch in true_labels])
    print("Pred Labels Shapes:", [batch.shape for batch in pred_labels])
    
    # Sample of the actual labels
    print("Sample True Labels:", true_labels[0])
    print("Sample Pred Labels:", pred_labels[0])
    
    flattened_true_labels = np.concatenate([batch.flatten() for batch in true_labels])
    flattened_pred_labels = np.concatenate([batch.flatten() for batch in pred_labels])

    valid_true_labels = [label for label in flattened_true_labels if label != -100]
    valid_pred_labels = [label for idx, label in enumerate(flattened_pred_labels) if flattened_true_labels[idx] != -100]

    # Ensure all labels in true_labels and pred_labels are present in label2id
    all_labels = set(np.unique(np.concatenate(true_labels, axis=0))).union(set(np.unique(np.concatenate(pred_labels, axis=0))))
    missing_labels = [label for label in all_labels if label not in label2id.values()]

    # If there are missing labels, print a warning
    if missing_labels:
        print(f"Warning: Missing labels {missing_labels} not found in label2id. Ignoring these labels for metrics calculation.")

    # Use only valid labels for metrics computation
    valid_labels = [label for label in all_labels if label in label2id.values()]

    report = classification_report(valid_true_labels, valid_pred_labels, target_names=labels, labels=[label2id[l] for l in labels if l in label2id], output_dict=True)

    return {
        "accuracy": report["accuracy"],
        "B-A_f1": report["B-A"]["f1-score"],
        "I-A_f1": report["I-A"]["f1-score"],
        "B-B_f1": report["B-B"]["f1-score"],
        "I-B_f1": report["I-B"]["f1-score"],
        "macro avg f1": report["macro avg"]["f1-score"],
        "macro avg precision": report["macro avg"]["precision"],
        "macro avg recall": report["macro avg"]["recall"]
    }


# Load validation data
with jsonlines.open('validation.jsonl') as reader:
    val_data = [line for line in reader]

val_dataset = NERDataset(val_data)
val_dataloader = DataLoader(val_dataset, batch_size=4)

for epoch in range(3):  # Number of epochs can be changed
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs = batch['words'].to(torch.device('cuda')) if torch.cuda.is_available() else batch['words']
        labels = batch['labels'].to(torch.device('cuda')) if torch.cuda.is_available() else batch['labels']
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optim.step()
        optim.zero_grad()

    print(f"Epoch {epoch + 1}, Training loss: {total_loss / len(dataloader)}")

    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch['words'].to(torch.device('cuda')) if torch.cuda.is_available() else batch['words']
            labels = batch['labels'].to(torch.device('cuda')) if torch.cuda.is_available() else batch['labels']
            outputs = model(inputs)
            preds = torch.argmax(outputs.logits, dim=2)

            true_labels.append(labels.cpu().numpy())
            pred_labels.append(preds.cpu().numpy())

    metrics = compute_metrics(true_labels, pred_labels)
    print(f"Epoch {epoch + 1}, Validation Accuracy: {metrics['accuracy']}, F1: {metrics['macro avg f1']}, Precision: {metrics['macro avg precision']}, Recall: {metrics['macro avg recall']}")

print("Training complete!")