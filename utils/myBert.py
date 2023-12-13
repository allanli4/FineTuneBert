import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import time 
import matplotlib.pyplot as plt
import os
import csv
from tqdm import tqdm
def multilabel_accuracy(preds, labels, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    correct = (preds == labels).float()
    acc = correct.sum() / correct.numel()
    return acc
class myBert():
    def __init__(self,train_dataloader, validation_dataloader, test_dataloader):
        
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.num_labels= len(train_dataloader.dataset[0][2])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.num_labels)
        self.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.train_history=None
        self.val_history=None
        self.test_results=None
        self.hyperparameters=None

        print("test")

    def validate(self,dataloader, loss_fn, device):
        self.model.eval()
        total_loss, total_accuracy = 0, 0

        for batch in dataloader:
            inputs, attention_masks, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            with torch.no_grad():
                outputs = self.model(input_ids=inputs, attention_mask=attention_masks)
                loss = loss_fn(outputs.logits, labels)
                acc = multilabel_accuracy(outputs.logits, labels)

            total_loss += loss.item()
            total_accuracy += acc.item()

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_accuracy / len(dataloader)
        return avg_loss, avg_acc
    
    def train(self,epochs=20, lr=2e-5, freeze=0,early_stopping=5,decay=1):
        if freeze > 13:
            raise ValueError("freeze should be less or equal than 12")
        
        self.hyperparameters={'epochs':epochs,'lr':lr,'freeze':freeze,'early_stopping':early_stopping,'decay_rate':decay}
        if freeze != 0:
            #freeze bert layers
            for i in range(freeze):
                self.model.bert.encoder.layer[i].requires_grad = False

        optimizer = AdamW(self.model.parameters(), lr=lr, eps=1e-8)
        steps_per_epoch = len(self.train_dataloader)
        scheduler = StepLR(optimizer, step_size=steps_per_epoch, gamma=decay)
        loss_fn=torch.nn.BCEWithLogitsLoss()
        train_history={'loss':[], 'acc':[]}
        val_history={'loss':[], 'acc':[]}
        min_val_loss = np.inf
        patience_counter = 0
        start_time = time.time()
        for epoch in range(epochs):
            self.model.train()
            total_loss, total_accuracy = 0, 0
            
            for batch in tqdm(self.train_dataloader, desc='batches'):
                inputs, attention_masks, labels = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(input_ids=inputs, attention_mask=attention_masks)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                acc = multilabel_accuracy(outputs.logits, labels)

                total_loss += loss.item()
                total_accuracy += acc.item()
            avg_train_loss = total_loss / len(self.train_dataloader)
            avg_train_acc = total_accuracy / len(self.train_dataloader)
            train_history['loss'].append(avg_train_loss)
            train_history['acc'].append(avg_train_acc)
            

            val_loss, val_accuracy = self.validate(self.validation_dataloader, loss_fn, self.device)
            print(f"Epoch {epoch + 1}/{epochs} - train Loss: {avg_train_loss:.4f} train Acc: {avg_train_acc:.4f}"
                  " Valid Loss: {val_loss:.4f} Valid Acc: {val_accuracy:.4f}")
            val_history['loss'].append(val_loss)
            val_history['acc'].append(val_accuracy)
            #early stopping
            if val_loss < min_val_loss:
                min_val_loss = val_loss
            else:
                patience_counter += 1
                if patience_counter > early_stopping:
                    break
        train_time=time.time()-start_time
        train_history['time']=train_time
        print(f"Training time: {train_time//60} minutes {train_time%60:.2f} seconds")

        test_results={'loss':[], 'acc':[]}
        test_loss, test_accuracy = self.validate(self.test_dataloader, loss_fn, self.device)
        test_results['loss'].append(test_loss)
        test_results['acc'].append(test_accuracy)
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
        
        self.train_history=train_history
        self.val_history=val_history
        self.test_results=test_results
        return train_history, val_history, test_results
    
    def save_results(self,save_path=None):
        """save results to csv file
        metrics: learning rate, freeze, decay rate, best train loss, best train acc, 
        best val loss, best val acc, test loss, test acc"""
        file_name="results.csv"
        if save_path:
            file_name=os.path.join(save_path,file_name)
        with open(file_name,  mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            if f.tell() == 0:
                csv_writer.writerow(['lr','freeze','decay rate','train time','best train loss','best train acc','best val loss','best val acc','test loss','test acc'])
            csv_writer.writerow([self.hyperparameters['lr'],
                                self.hyperparameters['freeze'],
                                self.hyperparameters['decay_rate'],
                                min(self.train_history['loss']),
                                max(self.train_history['acc']),
                                min(self.val_history['loss']),
                                max(self.val_history['acc']),
                                self.test_results['loss'][-1],
                                self.test_results['acc'][-1]])
        
        self.plt(save_path=save_path,save=True)
        print("results saved")


    def plt(self,save_path=None,save=False):
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_history['loss'], label='train_loss')
        plt.plot(self.val_history['loss'], label='val_loss')
        plt.title('Loss w.r.t. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')  
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.train_history['acc'], label='train_acc')
        plt.plot(self.val_history['acc'], label='val_acc')
        plt.title('Accuracy w.r.t. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')  
        plt.legend()
        learning_rate = self.hyperparameters['lr']
        freeze = self.hyperparameters['freeze']
        decay = self.hyperparameters['decay_rate']
        file_name=f'bert_lr-{learning_rate}_frz-:{freeze}_dr-{decay}.png'
        if save_path:
            file_name=os.path.join(save_path,file_name)
        if save:
            plt.savefig(file_name,bbox_inches='tight')
        plt.show()
        
