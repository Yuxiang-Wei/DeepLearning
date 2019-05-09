import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import seaborn as sns
from data_loader import Fer2013
sns.set()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)

batch_size = 16
img_size = 48
output_size = 7

# train_dataset = datasets.MNIST('./data',
#                                train=True,
#                                download=False,
#                                transform=transforms.ToTensor())
#
# validation_dataset = datasets.MNIST('./data',
#                                     train=False,
#                                     transform=transforms.ToTensor())
#
train_dataset = Fer2013('fer2013.csv',
                               train=True,
                               transform=transforms.ToTensor())

validation_dataset = Fer2013('fer2013.csv',
                                    train=False,
                                    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(img_size*img_size, 1024)

        # self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 256)
        # self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = x.view(-1, img_size*img_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters())
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()

print(model)


def train(epochs, log_interval=200):

    for i in range(1, epochs+1):
        # Set model to training mode
        model.train()
        # Loop over each batch from the training set
        for batch_idx, (data, target) in enumerate(train_loader):
            # Copy data to GPU if needed
            data = data.to(device)
            target = target.to(device)
            # Zero gradient buffers
            optimizer.zero_grad()

            # Pass data through the network
            output = model(data)

            # Calculate loss
            loss = criterion(output, target)

            # Backpropagate
            loss.backward()

            # Update weights
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data.item()))

        validate(lossv, accv)


def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        # print(pred, target.data)
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)
    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))

lossv, accv = [], []

train(5)
