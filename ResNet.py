import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, ConcatDataset

cuda0 = torch.device('cuda:0')

def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        torch.nn.init.uniform_(m.weight, -0.05, 0.05)

if __name__ == '__main__':
    batch_size = 25  
    nb_epochs = 10  
    eta = 0.01  

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
    train1_data = datasets.ImageFolder(root='cars_tanks/train/',transform=transformRotation)
    train2_data = datasets.ImageFolder(root='cars_tanks/train/',transform=transformFlip)
    train3_data = datasets.ImageFolder(root='cars_tanks/train/', transform=transform)
    train_data = ConcatDataset([original_data,train1_data,train2_data,train3_data])
    #train_data = ConcatDataset([original_data])
    test_data = datasets.ImageFolder(root='cars_tanks/test/', transform=transformOriginal)

    # Création des DataLoader pour l'entraînement et les tests
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # Import de ResNet
    model = models.resnet18(pretrained=True)

    # Remplacement de la dernière couche
    for param in model.parameters():
        param.requires_grad = True
    model.fc = torch.nn.Linear(512, 2)
    model.fc.apply(init_weights)

    model.to(cuda0)


    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)

    # Entrainement du modèle
    for epoch in range(nb_epochs):
        model.train() 
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(cuda0), labels.to(cuda0)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{nb_epochs}] - Loss: {running_loss/len(train_loader)}")

    # Évaluation du modèle sur les données de test
    model.eval() 
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(cuda0), labels.to(cuda0)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += 1
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy on test data: {accuracy:.2%}")