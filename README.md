# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
<img width="909" height="639" alt="Screenshot 2026-02-27 101952" src="https://github.com/user-attachments/assets/cdfd0070-23f6-4f3e-b336-5f326376289c" />


## DESIGN STEPS
### STEP 1: 
Load the Fashion-MNIST dataset and preprocess images using tensor conversion and normalization.

### STEP 2: 
Create DataLoaders for training and testing with a fixed batch size.


### STEP 3: 
Design a CNN model with convolution, ReLU activation, max-pooling, and fully connected layers.



### STEP 4: 
Train the model using Cross-Entropy loss and the Adam optimizer for several epochs.


### STEP 5: 
Test the trained model to calculate accuracy, confusion matrix, and classification report.


### STEP 6: 
Predict and display the class of a single test image with actual and predicted labels.


## PROGRAM

### Name:ARAVINDAN SD

### Register Number: 2122242423001

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Step 1: Load and Preprocess Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Step 2: Load MNIST Dataset
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)

# Check Dataset
image, label = train_dataset[0]
print("Image shape:", image.shape)
print("Number of training samples:", len(train_dataset))

image, label = test_dataset[0]
print("Image shape:", image.shape)
print("Number of testing samples:", len(test_dataset))

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 3: Create CNN Model
class CNNClassifier(nn.Module):

    def __init__(self):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize Model, Loss Function and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Name: ARAVINDAN SD")
print("Register Number: 212224243001")
print("Device:", device)
print(model)


# Step 4: Train the Model
def train_model(model, train_loader, num_epochs=3):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("Name: ARAVINDAN SD")
        print("Register Number: 212224243001")
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Loss: {running_loss / len(train_loader):.4f}"
        )


# Train the Model
train_model(model, train_loader, num_epochs=10)


# Step 5: Test the Model
def test_model(model, test_loader):
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total

    print("Name: ARAVINDAN SD")
    print("Register Number: 212224243001")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10)
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Classification Report
    print("Name: ARAVINDAN SD")
    print("Register Number: 212224243001")
    print("Classification Report:")

    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=[str(i) for i in range(10)]
        )
    )


# Evaluate the Model
test_model(model, test_loader)


# Step 6: Predict a Single Image
def predict_image(model, image_index, dataset):
    model.eval()

    image, label = dataset[image_index]
    input_image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image)
        _, predicted = torch.max(output, 1)

    class_names = [str(i) for i in range(10)]

    print("Name: ARAVINDAN SD")
    print("Register Number: 212224243001")

    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(
        f"Actual: {class_names[label]}\n"
        f"Predicted: {class_names[predicted.item()]}"
    )
    plt.axis("off")
    plt.show()

    print(
        f"Actual: {class_names[label]}, "
        f"Predicted: {class_names[predicted.item()]}"
    )


# Example Prediction
predict_image(model, image_index=80, dataset=test_dataset)
```

### OUTPUT

## Training Loss per Epoch

<img width="610" height="426" alt="image" src="https://github.com/user-attachments/assets/f81f8384-d67d-4a08-b11b-b918ee68fe48" />

## Confusion Matrix

<img width="626" height="525" alt="image" src="https://github.com/user-attachments/assets/379aa983-240d-41cf-96dd-f322370b48fa" />

## Classification Report

<img width="707" height="597" alt="download" src="https://github.com/user-attachments/assets/876587b4-d83e-4f1d-8859-7844cc73ed58" />

<img width="710" height="342" alt="image" src="https://github.com/user-attachments/assets/1ea37f38-5eb4-446c-a4b0-2f6f526ad7f6" />

### New Sample Data Prediction

<img width="389" height="432" alt="download" src="https://github.com/user-attachments/assets/ca5a9637-1b23-4a15-a228-8771b2f3e77f" />

## RESULT
This program has been executed successfully.
