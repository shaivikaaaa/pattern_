#normalise
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

num_epochs = 11 #can be modified
learning_rate = 0.1
num_classes = 10
batch_size = 128


#model
model_name = "Shaivika_CI"


#data
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomHorizontalFlip(), #regularization and this what fixes our accuracies .. we flip through data left right , augmentatiom
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) #we r not doing augmentation over here, this is real and pure data


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False) #we dont want shuffling in here


#--------------- Model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1): #in_planes:i/p dimensions and how many planes:depth of each convulation
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) #BatchNorm2d:to make sure evrything is normalized in between and chain with convulation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            ) #these are the actual layer names and layers but they are not connected, they get connected through forward fnc


    def forward(self, x): #what happens when we actually pass some data through it(above def init one)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module): #as we add layers we begin to do so many operations that we simply start to lose the data .
    #before we enter a layer we save the result and in the end you add the result back in. it forces all the layers here in to be the residual of i/p
    #our networj will only handle the differences so there is smoothing/loss of data
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()  # Correctly call the parent class constructor
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)




    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
#So, [2, 2, 2, 2] means that ResNet-18 has two residual blocks in each of its four stages, resulting in a total of 18
# convolutional layers (2 * 2 * 2 * 2 = 16 convolutional layers) along with the initial convolutional layer and the final fully connected layers.

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

model = ResNet18()
model = model.to(device=device)

#print(model)

critereon = nn.CrossEntropyLoss() #
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

total_step = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, epochs=num_epochs, steps_per_epoch=len(train_loader))

#to make is faster and accuracy 94% we made the learning rate chnage.. instead of 93% we can get 94% in 1/10of the time of 93%

#running the model
print("Now training")
start = time.time()  # Time measurement

n_total_steps = len(train_loader)
for i in range(num_epochs): #each epoch we get a batch
    model.train()
    total_correct = 0
    total_samples = 0

    for j, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images) #pass it through the models
        loss = critereon(outputs, labels) #compute the loss function

        optimizer.zero_grad() #applying the gradient step
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Compute batch accuracy
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        # Print batch accuracy
        batch_accuracy = 100 * total_correct / total_samples
        print(f'Epoch [{i+1}/{num_epochs}], Step [{j+1}/{total_step}], Batch Accuracy: {batch_accuracy:.2f}%')

end = time.time()
elapsed = end-start
print(elapsed)
#torch.save(model.state_dict(), path)