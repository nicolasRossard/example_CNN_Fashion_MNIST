import torch
import torchvision
import torchvision.transforms as transforms
 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
 
torch.set_grad_enabled(True)
torch.set_printoptions(linewidth =120)
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix
import itertools 
import pdb


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # First CL D_in=1, K = 6, F = 5, by default: S = 1 and P = 0 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)

        # Second CL D_in=6, K = 12 F = 5, by default: S = 1 and P =0
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # (1) input layer
        t = t

        # (2) First hidden conv layer (CL + RELU + PL)
        # CL
        t = self.conv1(t)

        # if val<0 ==> x = 0
        t = F.relu(t)

        # First hidden pooling layer (PL): F=2 S=2, by default P = 0 D_in = D_out
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) Second hidden conv layer (Cl + RELU + PL)
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)

        return t

#Downloading data from torchvision 
train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

# Informations of the training set:
print("-------------------- Datas informations ----------------------")
print('Type of train_set: {0}'.format(type(train_set))) 
print('Size of train_set: {0}'.format(len(train_set))) # 60 000 images with their labels
print('Type of train_set[0]: {0}'.format(type(train_set[0]))) # tuple(tensor,label)
print('Size of train_set[0]: {0}'.format(len(train_set[0])))  # 28 arrays of 28 cells
print('Type of train_set[0][0]: {0}'.format(type(train_set[0][0]))) # tensor
print('Size of train_set[0][0]: {0}'.format(len(train_set[0][0]))) # [[[28 values ], [28 values] .... 28 arrays ]]]

print('Type of train_set[0][1]: {0}'.format(type(train_set[0][1]))) # Label type int
print('Label of train_set[0][1]: {0}'.format(train_set[0][1]))   #Label value

print("------------------------- End -------------------------------")

network = Network()

# take 100 samples / batch
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
print('Length train_loader: {0}'.format(len(train_loader)))


batch = next(iter(train_loader)) # Getting a batch

# Split images and labels in 2 different tensor
images, labels = batch

print('Images: {0}'.format(images.size()))

preds = network(images) 

print('Preds: {0}'.format(preds.size()))


# cross entropy: loss function to see error of classifications
loss = F.cross_entropy(preds, labels) # Calculating the loss

loss.item()

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

get_num_correct(preds, labels)
#Calculating the Gradients
network.conv1.weight.grad

loss.backward()

network.conv1.weight.grad.shape

#Updating the Weights
optimizer = optim.Adam(network.parameters(), lr=0.01)
optimizer.step() # Updating the weights

preds = network(images)
loss.item()

print('\n-------------------------------- TRAIN USING A SINGLE BACTH ---------------------------------------------------')
#Train Using a Single Batch
network = Network()
 
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01)
 
batch = next(iter(train_loader)) # Get Batch
images, labels = batch
 
preds = network(images) # Pass Batch
loss = F.cross_entropy(preds, labels) # Calculate Loss
 
loss.backward() # Calculate Gradients
optimizer.step() # Update Weights
 
print('loss1:', loss.item())
preds = network(images)
loss = F.cross_entropy(preds, labels)
print('loss2:', loss.item())


print('\n-------------------------------- TRAINING USING ALL BATCHES (= ONE EPOCH) ---------------------------------------------------')
network = Network()
 
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01)
 
total_loss = 0
total_correct = 0
 
for batch in train_loader: # Get Batch
    images, labels = batch 
 
    preds = network(images) # Pass Batch
    loss = F.cross_entropy(preds, labels) # Calculate Loss
 
    optimizer.zero_grad()
    loss.backward() # Calculate Gradients
    optimizer.step() # Update Weights
 
    total_loss += loss.item()
    total_correct += get_num_correct(preds, labels)
    
print(
    "epoch:", 0, 
    "total_correct:", total_correct, 
    "loss:", total_loss
)


print('\n-------------------------------- TRAINING WITH MULTIPLE EPOCHS ---------------------------------------------------')
network = Network()
 
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01)

for epoch in range(10):
    
    total_loss = 0
    total_correct = 0
    
    for batch in train_loader: # Get Batch
        images, labels = batch 
 
        preds = network(images) # Pass Batch
        loss = F.cross_entropy(preds, labels) # Calculate Loss
 
        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights
 
        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)
 
    print(
        "epoch", epoch, 
        "total_correct:", total_correct, 
        "loss:", total_loss
    )


print('\n-------------------------------- CREATING FUNCTION TO GET PREDICTIONS FOR ALL SAMPLES  ---------------------------------------------------')


@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch
 
        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds

with torch.no_grad():
    prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
    train_preds = get_all_preds(network, prediction_loader)

preds_correct = get_num_correct(train_preds, train_set.targets)
 
print('total correct:', preds_correct)
print('accuracy:', preds_correct / len(train_set))   


print('\n-------------------------------- BUILDING CONFUSION MATRIX  ---------------------------------------------------')

train_set.targets
train_preds.argmax(dim=1)

stacked = torch.stack(
    (
        train_set.targets
        ,train_preds.argmax(dim=1)
    )
    ,dim=1
)
 
stacked.shape

cmt = torch.zeros(10,10, dtype=torch.int64)
cmt

for p in stacked:
    tl, pl = p.tolist()
    cmt[tl, pl] = cmt[tl, pl] + 1

print(cmt)


print('\n-------------------------------- PLOTTING CONFUSION MATRIX  ---------------------------------------------------')
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
 
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
 
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


names = (
    'T-shirt/top'
    ,'Trouser'
    ,'Pullover'
    ,'Dress'
    ,'Coat'
    ,'Sandal'
    ,'Shirt'
    ,'Sneaker'
    ,'Bag'
    ,'Ankle boot'
)
plt.figure(figsize=(10,10))
plot_confusion_matrix(cmt, names)
plt.show()


print('\n-------------------------------- BUILDING PERFORMANCES FOR EACH CLASS  ---------------------------------------------------')

#print(cmt.numpy())
#print(len(cmt))
res = cmt.numpy()

# get the sum of each colonne: total of elements for each class
total = np.sum(res, axis=0)

#get the diagonal
predicted=res.diagonal()

# Get accurency for each class
accurency = np.true_divide(predicted,total)

# Get confusion column
row, col = res.shape
listConf = []
for y in range(col):
    maxC = 0
    for x in range(row):
        if ((maxC <= res[x][y]) & (x!=y)):
            maxC = res[x][y]
            index = x

    listConf.append(names[index])

# Create dataframe usable

#tab = np.array(['TOTAL','PREDICTED','ACCURENCY','CONFUSION'],total,predicted,accurency,listConf)
conf = np.array(listConf)

print("\n\n")
print(np.array(['TOTAL','PREDICTED','ACCURENCY','CONFUSION']))
print(total)
print(predicted)
print(accurency)
print(conf)
print("\n\n")

combined = np.array([row for row_group in zip(names,total,predicted,accurency,conf) for row in row_group]).reshape((len(total),5))

df = pd.DataFrame(combined,columns=['Class','Total','Predicted','Accurency','Confusion'])
print(df)

df['Total']=df['Total'].astype(int)
df['Predicted']=df['Predicted'].astype(int)
df['Accurency']=df['Accurency'].astype(float)
df['Accurency']=df['Accurency'].round(2)

print('\n-------------------------------- DISPLAY PERFORMANCES FOR EACH CLASS  ---------------------------------------------------')
df.plot(kind='bar',x='Class',y='Accurency')
plt.show()

