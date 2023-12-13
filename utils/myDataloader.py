from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import TensorDataset

def dataloader_bert(dataset="disaster_response_messages",batch_size=16):
    label_columns = ['related', 'PII', 'request', 'offer', 'aid_related', 'medical_help',
                    'medical_products', 'search_and_rescue', 'security', 'military',
                    'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
                    'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related',
                    'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops',
                    'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm',
                    'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']
    # num_labels = len(label_columns)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def preprocess_for_bert(data):
        # Tokenize the messages and prepare the labels
        input_ids = []
        attention_masks = []
        labels = []

        for i in range(len(data)):
            encoded = tokenizer.encode_plus(
                data[i]['message'],
                add_special_tokens=True,
                max_length=128,  # Adjust as needed
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
            labels.append([data[i][label] for label in label_columns])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels, dtype=torch.float)

        return input_ids, attention_masks, labels
    dataset = load_dataset("disaster_response_messages")
    # Apply preprocessing
    train_input_ids, train_attention_masks, train_labels = preprocess_for_bert(dataset['train'])
    val_input_ids, val_attention_masks, val_labels = preprocess_for_bert(dataset['validation'])
    test_input_ids, test_attention_masks, test_labels = preprocess_for_bert(dataset['test'])
    # Create the DataLoader for our training set
    train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    train_sampler = torch.utils.data.RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    # Create the DataLoader for our validation set
    validation_data = TensorDataset(val_input_ids, val_attention_masks, val_labels)
    validation_sampler = torch.utils.data.SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_input_ids, test_attention_masks, test_labels)
    test_sampler = torch.utils.data.SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    return train_dataloader, validation_dataloader, test_dataloader
