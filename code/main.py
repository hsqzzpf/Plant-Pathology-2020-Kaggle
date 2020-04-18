import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, models
from torch.utils.data import Dataset, Dataloader

from model import init_model
from loss import get_loss_fn
from dataset import ppDataset



def get_device():
	if torch.cuda.is_available():
		device = torch.device("cuda")
		print('There are %d GPU(s) available.' % torch.cuda.device_count())
		print('We will use the GPU:', torch.cuda.get_device_name(0))
	else:
		print('No GPU available, using the CPU instead.')
		device = torch.device("cpu")
	return device


def train(model, tr_dataloader, criterion, optimizer, epoch):
	since = time.time()
	device = get_device()
	model.train()

	model.to(device)

	running_loss = 0.
	running_corrects = 0.
	for inputs, labels, _ in tr_dataloader:
		inputs = inputs.to(device)
		labels = labels.to(device)
		labels = labels.squeeze(-1)
				
		optimizer.zero_grad()
		model.zero_grad()

		outputs = model(inputs)
		loss = criterion(outputs, labels)
		_, preds = torch.max(outputs, 1)

		loss.backward()
		optimizer.step()
				
		running_loss += loss.item() * inputs.size(0)
		running_corrects += torch.sum(labels.argmax(dim=1) == outputs.argmax(dim=1))

	epoch_loss = running_loss / len(tr_dataloader.dataset)
	epoch_acc = running_corrects.double() / len(tr_dataloader.dataset)

	print('[Train]Epoch: {}, Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))

	return epoch_acc, epoch_loss


def validation(model, val_dataloader, criterion, epoch):
	device = get_device()
	model.to(device)
	model.eval()

	running_loss = 0.
	running_corrects = 0.
	with torch.no_grad():
		for inputs, labels, _ in val_dataloader:
			inputs = inputs.to(device)
			labels = labels.to(device)
			labels = labels.squeeze(-1)
					
			model.zero_grad()

			outputs = model(inputs)
			loss = criterion(outputs, labels)
			_, preds = torch.max(outputs, 1)
					
			running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(labels.argmax(dim=1) == outputs.argmax(dim=1))

	epoch_loss = running_loss / len(val_dataloader.dataset)
	epoch_acc = running_corrects.double() / len(val_dataloader.dataset)

	print('[Validation]Epoch: {}, Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
	return epoch_acc, epoch_loss


def write_csv(model, te_dataset, submission_df_path):
	print("Generating prediction...")
	te_dataloader = DataLoader(te_dataset, batch_size=batch_size, shuffle=False)
	submission_df = pd.read_csv(submission_df_path)

	test_pred = None
	model.eval()
	with torch.no_grad():
		for inputs in te_dataloader:
			inputs = inputs.to(device)
			outputs = model_ft(inputs)

			if test_pred is None:
				test_pred = outputs.data.cpu()
			else:
				test_pred = torch.cat((test_pred, outputs.data.cpu()), dim=0)

	test_pred = torch.softmax(test_pred, dim=1, dtype=float)
	submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = test_preds

	submission_df.to_csv('submission_3.csv', index=False)



if __name__ == "__main__":
    model_name = 'resnext'
    num_classes = 4
    batch_size = 16
    num_epochs = 2
    num_dev_samples = 0
    feature_extract = False
    pre_trained = True
    num_cv_folds = 5
    
    train_csv_path = "../data/train.csv"
    test_csv_path = "../data/test.csv"

    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
		transforms.Pad(50, padding_mode='reflect'),        
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomVerticalFlip(p=0.5),
		transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05, hue=0.05),
		transforms.RandomAffine(degrees=[0,45]),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
	'val': transforms.Compose([
		transforms.Resize((224, 244)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]),
	'test': transforms.Compose([
		transforms.Resize((224, 244)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		])
}

	model_ft, input_size = init_model(model_name, num_classes, feature_extract, use_pretrained=pre_trained)
	criterion = get_loss_fn()
	optimizer = torch.optim.AdamW(params_to_update, lr = 2e-5, eps = 1e-8 )

	tr_df_all = pd.read_csv(train_csv_path)
	tr_df, val_df = train_test_split(tr_df_all, test_size = 0.4)
	val_df = val_df.reset_index(drop=True)
	tr_df = tr_df.reset_index(drop=True)
	te_df = pd.read_csv(test_csv_path)
	
	tr_dataset = ppDataset(tr_df, images_dir, return_labels = True, transforms = data_transforms['train'])
	val_dataset = ppDataset(val_df, images_dir, return_labels = True, transforms = data_transforms['val'])
	te_dataset = ppDataset(te_df, images_dir, return_labels = False, transforms = data_transforms['test'])
    
    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
	train_loss_ls = []
	valid_loss_ls = []
	train_accu_ls = []
	valid_accu_ls = []

	for i in range(1, num_epoch+1):
		train_acc, train_loss = train(model, tr_dataloader, criterion, optimizer, epoch)
		val_acc, val_loss = validation(model, val_dataloader, criterion, epoch)

		train_loss_ls.append(train_loss)
		train_accu_ls.append(train_acc)
		valid_loss_ls.append(val_loss)
		valid_accu_ls.append(val_acc)
    submission_df_path = "../data/sample_submission.csv"
	write_csv(model, te_dataset, submission_df_path)

