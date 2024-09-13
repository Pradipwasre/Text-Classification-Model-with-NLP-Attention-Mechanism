import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class AttentionModel(nn.Module):
    def __init__(self, vec_len, seq_len, n_classes, device):
        super(AttentionModel, self).__init__()
        self.vec_len = vec_len
        self.seq_len = seq_len
        self.device = device  # Store device for future use
        
        # Initialize attention weights
        self.attn_weights = torch.cat([torch.tensor([[0.]]),
                                       torch.randn(vec_len, 1) /
                                       torch.sqrt(torch.tensor(vec_len))])
        self.attn_weights.requires_grad = True
        self.attn_weights = nn.Parameter(self.attn_weights)
        
        # Define activation, softmax, and linear layer
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(vec_len + 1, n_classes)

    def forward(self, input_data):
        input_data = input_data.to(self.device)
        hidden = torch.matmul(input_data, self.attn_weights)
        hidden = self.activation(hidden)
        attn = self.softmax(hidden)
        attn = attn.repeat(1, 1, self.vec_len + 1).reshape(attn.shape[0],
                                                           self.seq_len,
                                                           self.vec_len + 1)
        attn_output = input_data * attn
        attn_output = torch.sum(attn_output, axis=1)
        output = self.linear(attn_output)
        return output

def train(train_loader, valid_loader, model, criterion, optimizer, device, num_epochs, model_path):
    """
    Function to train the model
    :param train_loader: Data loader for train dataset
    :param valid_loader: Data loader for validation dataset
    :param model: Model object
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param device: CUDA or CPU
    :param num_epochs: Number of epochs
    :param model_path: Path to save the model
    """
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} of {num_epochs}")
        model.train()
        train_loss = []
        
        # Train loop
        for batch_labels, batch_data in tqdm(train_loader):
            batch_labels = batch_labels.to(device)
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            batch_output = model(batch_data)
            batch_output = torch.squeeze(batch_output)
            
            # Calculate loss
            loss = criterion(batch_output, batch_labels)
            train_loss.append(loss.item())
            
            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()
        
        model.eval()
        valid_loss = []
        
        # Validation loop
        with torch.no_grad():
            for batch_labels, batch_data in tqdm(valid_loader):
                batch_labels = batch_labels.to(device)
                batch_data = batch_data.to(device)
                
                # Forward pass
                batch_output = model(batch_data)
                batch_output = torch.squeeze(batch_output)
                
                # Calculate loss
                loss = criterion(batch_output, batch_labels)
                valid_loss.append(loss.item())
        
        t_loss = np.mean(train_loss)
        v_loss = np.mean(valid_loss)
        print(f"Train Loss: {t_loss}, Validation Loss: {v_loss}")
        
        # Save model if validation loss improves
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), model_path)
        print(f"Best Validation Loss: {best_loss}")

def test(test_loader, model, criterion, device):
    """
    Function to test the model
    :param test_loader: Data loader for test dataset
    :param model: Model object
    :param criterion: Loss function
    :param device: CUDA or CPU
    """
    model.eval()
    test_loss = []
    test_accu = []
    
    with torch.no_grad():
        for batch_labels, batch_data in tqdm(test_loader):
            batch_labels = batch_labels.to(device)
            batch_data = batch_data.to(device)
            
            # Forward pass
            batch_output = model(batch_data)
            batch_output = torch.squeeze(batch_output)
            
            # Calculate loss
            loss = criterion(batch_output, batch_labels)
            test_loss.append(loss.item())
            
            # Calculate accuracy
            batch_preds = torch.argmax(batch_output, axis=1)
            test_accu.append(accuracy_score(batch_labels.cpu().numpy(),
                                            batch_preds.cpu().numpy()))
    
    test_loss = np.mean(test_loss)
    test_accu = np.mean(test_accu)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accu}")
