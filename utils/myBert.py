import torch
import numpy as np
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

def multilabel_accuracy(preds, labels, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    correct = (preds == labels).float()
    acc = correct.sum() / correct.numel()
    return acc

def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, total_accuracy = 0, 0

    for batch in dataloader:
        inputs, attention_masks, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        with torch.no_grad():
            outputs = model(input_ids=inputs, attention_mask=attention_masks)
            loss = loss_fn(outputs.logits, labels)
            acc = multilabel_accuracy(outputs.logits, labels)

        total_loss += loss.item()
        total_accuracy += acc.item()

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_accuracy / len(dataloader)
    return avg_loss, avg_acc

def train_bert(train_dataloader, validation_dataloader, test_dataloader, epochs=20, lr=2e-5, freeze=None,early_stopping=5):
    num_labels= len(train_dataloader.dataset[0][2])
    # print(f"Number of labels: {num_labels}")
    # print(len(train_dataloader.dataset), len(validation_dataloader.dataset), len(test_dataloader.dataset))
    # return
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if freeze != None:
        num_layers_to_freeze = freeze
        for i, param in enumerate(model.parameters()):
            if i < num_layers_to_freeze:
                param.requires_grad = False
            else:
                break
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn=torch.nn.BCEWithLogitsLoss()
    train_history={'loss':[], 'acc':[]}
    val_history={'loss':[], 'acc':[]}

    # trainloss_history = []
    # trainacc_hisotory = []
    # valiloss_history = []
    # valiacc_history = []
    min_val_loss = np.inf
    pateince_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss, total_accuracy = 0, 0
        
        for batch in train_dataloader:
            inputs, attention_masks, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()

            outputs = model(input_ids=inputs, attention_mask=attention_masks)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            acc = multilabel_accuracy(outputs.logits, labels)

            total_loss += loss.item()
            total_accuracy += acc.item()
        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_acc = total_accuracy / len(train_dataloader)
        train_history['loss'].append(avg_train_loss)
        train_history['acc'].append(avg_train_acc)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.4f}")

        val_loss, val_accuracy = validate(model, validation_dataloader, loss_fn, device)
        print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
        val_history['loss'].append(val_loss)
        val_history['acc'].append(val_accuracy)
        #early stopping
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        else:
            pateince_counter += 1
            if pateince_counter > early_stopping:
                break
    test_results={'loss':[], 'acc':[]}
    test_loss, test_accuracy = validate(model, test_dataloader, loss_fn, device)
    test_results['loss'].append(test_loss)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
    return model, train_history, val_history, test_results
    
