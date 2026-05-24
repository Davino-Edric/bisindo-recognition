import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import mlflow

from dataset import load_all_classes, split_data, BisindoDataset
from model import BisindoMLP

def train_one_epoch(model,loader,criterion,optimizer):
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
        
        predicted = torch.argmax(outputs, dim=1)
        correct = (predicted == y_batch).sum().item()
        
        total_loss += loss.item()
        total_correct += correct
        total_samples += len(y_batch)
    
    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples * 100
    
    return avg_loss, accuracy

def evaluate(model,loader,criterion):
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            predicted = torch.argmax(outputs, dim=1)
            correct = (predicted == y_batch).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += len(y_batch)
        
        avg_loss = total_loss / len(loader)
        accuracy = total_correct / total_samples * 100
        
        return avg_loss, accuracy
    
if __name__ == "__main__":
    best_val_accuracy = 0
    num_epochs = 50
    X,y,label_map = load_all_classes()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    train_dataset = BisindoDataset(X_train, y_train)
    val_dataset = BisindoDataset(X_val, y_val)
    test_dataset = BisindoDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    model = BisindoMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(model)
    
    mlflow.set_experiment("bisindo-mlp")
    
    with mlflow.start_run():
        mlflow.log_params({
            "model": "BisindoMLP",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 32,
            "num_epochs": num_epochs,
        })
        for epoch in range(num_epochs):
            model.train()
            loss, accuracy = train_one_epoch(model, train_loader, criterion, optimizer)
            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("train_acc", accuracy, step=epoch)
            
            model.eval()
            val_loss, val_accuracy = evaluate(model, val_loader, criterion)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_accuracy, step=epoch)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {loss:.4f}, Train Acc: {accuracy:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), "best_model.pth")
                print(f"  New best model saved with Val Acc: {val_accuracy:.2f}%")