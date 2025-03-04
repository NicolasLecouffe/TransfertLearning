import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transformOriginal = transforms.Compose([
    transforms.ToTensor(), 
])


transform = transforms.Compose([
    transforms.RandomRotation(degrees=20), 
    transforms.RandomHorizontalFlip(),  
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # Translation et zoom aléatoires
    transforms.ToTensor(),  
])

transformRotation = transforms.Compose([
    transforms.RandomRotation(degrees=20),  # Rotation aléatoire jusqu'à 20 degrés
    transforms.ToTensor(), 
])

transformFlip = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Retournement horizontal aléatoire
    transforms.ToTensor(),  
])


# Chargement des données d'entraînement et de test
original_data = datasets.ImageFolder(root='cars_tanks/train/', transform=transformOriginal)
#train1_data = datasets.ImageFolder(root='cars_tanks/train/',transform=transformRotation)
#train2_data = datasets.ImageFolder(root='cars_tanks/train/',transform=transformFlip)
#train3_data = datasets.ImageFolder(root='cars_tanks/train/', transform=transform)
#train_data = ConcatDataset([original_data,train1_data,train2_data,train3_data])
train_data = ConcatDataset([original_data])
test_data = datasets.ImageFolder(root='cars_tanks/test/', transform=transformOriginal)

# Création des DataLoader pour l'entraînement et les tests
batch_size = 25  
eta = 0.01
num_epochs = 10
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

# Définition du modèle
model = torch.nn.Sequential(
    torch.nn.Linear(256*256 * 3, 1000),  
    torch.nn.ReLU(),
    torch.nn.Linear(1000, 2))  

# Initialisation des poids
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.05, 0.05)

model.apply(init_weights)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=eta)

# Entraînement du modèle
for epoch in range(num_epochs):
    model.train() 
    running_accuracy = 0.0
    total = 0
    correct = 0
    running_loss = 0
    for images, labels in train_loader:
        images = images.view(images.size(0), -1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    
    epoch_accuracy = correct / total
    
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Accuracy: Loss: {running_loss/len(train_loader)}")

# Évaluation du modèle
model.eval()  
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(1, -1) 
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += 1
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy on test data: {accuracy:.2%}")