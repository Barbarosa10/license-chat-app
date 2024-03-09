import torch.nn as nn
import torch.nn.functional as F

class CNNArchitecture(nn.Module):
    def __init__(self, imgWidth, imgHeight):
        super(CNNArchitecture, self).__init__()
        
        inputWidth = imgWidth
        inputHeight = imgHeight
        nrConvFilters = 4
        convFilterSize = 3
        poolSize = 2
        outputSize = 2

        # self.convLayer1 = nn.Conv2d(1, nrConvFilters, convFilterSize, stride=1, padding=1)
        # self.convLayer2 = nn.Conv2d(4, nrConvFilters, convFilterSize, stride=1, padding=1)
        # self.batchNorm2d = nn.BatchNorm2d(4)
        # self.relu = nn.ReLU(inplace=True)
        # self.poolLayer = nn.MaxPool2d(kernel_size=poolSize, stride=2)
        
        # # Calculate the size of the input to the fully connected layer
        # fcInputSize = 4 * ((imgWidth // 2) // 2) * ((imgHeight // 2) // 2)
        # self.fcLayer = nn.Linear(fcInputSize, outputSize)
        # self.softmax = nn.LogSoftmax(dim=1)

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            
        )
        fcInputSize = 4 * ((imgWidth // 2) // 2) * ((imgHeight // 2) // 2)
        # fcInputSize = 4 * ((imgWidth // 2) ) * ((imgHeight // 2))
        self.linear_layers = nn.Sequential(
            nn.Linear(fcInputSize, 2),
            nn.LogSoftmax(dim=1)
        )


        # self.cnn_layers = nn.Sequential(
        # #     # Defining a 2D convolution layer
        # #     nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
        # #     nn.BatchNorm2d(8),
        # #     nn.ReLU(inplace=True),
        # #     nn.MaxPool2d(kernel_size=2, stride=2),
        # #     # Defining another 2D convolution layer
        # #     nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=1),
        # #     nn.BatchNorm2d(16),
        # #     nn.ReLU(inplace=True),
        # #     nn.MaxPool2d(kernel_size=2, stride=2)            

        #     # Defining a 2D convolution layer
        #     nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=1),
        #     nn.BatchNorm2d(6),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=1),
        #     nn.BatchNorm2d(120),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2)            
        # )
        # # )
        # # fcInputSize1 = 4 * ((imgWidth // 2) // 2) * ((imgHeight // 2) // 2)
        # # fcInputSize2 = 4 * ((imgWidth // 2) // 2) * ((imgHeight // 2) // 2)
        # self.linear_layers = nn.Sequential(
        #     nn.Linear(16 * 6 * 6, 120),
        #     nn.Linear(120, 84),
        #     nn.Linear(84, 2),
        #     nn.LogSoftmax(dim=1)
        # )

    def forward(self, x):
        # x = self.convLayer1(x)
        # x = self.batchNorm2d(x)
        # x = self.relu(x)
        # x = self.poolLayer(x)

        # x = self.convLayer2(x)
        # x = self.batchNorm2d(x)
        # x = self.relu(x)
        # x = self.poolLayer(x)

        # x = x.view(x.size(0), -1)
        # x = self.fcLayer(x)
        # x = self.softmax(x)

        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x            
    