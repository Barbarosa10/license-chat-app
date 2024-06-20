import torch.nn as nn
import torch.nn.functional as F

class CNNArchitecture(nn.Module):
    """
    Convolutional Neural Network (CNN) architecture for image sentiment analysis.
    
    Args:
        imgWidth (int): Width of the input images.
        imgHeight (int): Height of the input images.
    """
    def __init__(self, imgWidth, imgHeight):
        super(CNNArchitecture, self).__init__()
        
        inputWidth = imgWidth
        inputHeight = imgHeight
        nrConvFilters = 4
        convFilterSize = 3
        poolSize = 2
        outputSize = 2

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        fcInputSize = 4 * ((imgWidth // 2) // 2) * ((imgHeight // 2) // 2)

        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fcInputSize, 32),
            # nn.Linear(fcInputSize, 2),
            nn.Linear(32, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor with class scores.
        """
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x



# import torch.nn as nn
# import torch.nn.functional as F

# class CNNArchitecture(nn.Module):
#     """
#     Convolutional Neural Network (CNN) architecture for image sentiment analysis.
    
#     Args:
#         imgWidth (int): Width of the input images.
#         imgHeight (int): Height of the input images.
#     """
#     def __init__(self, imgWidth, imgHeight):
#         super(CNNArchitecture, self).__init__()
        
#         inputWidth = imgWidth
#         inputHeight = imgHeight
#         nrConvFilters = 4
#         convFilterSize = 3
#         poolSize = 2
#         outputSize = 2

#         self.cnn_layers = nn.Sequential(
#             nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(4),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(4),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
        
#         fcInputSize = 4 * ((imgWidth // 2) // 2) * ((imgHeight // 2) // 2)

#         self.linear_layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(fcInputSize, 32),
#             # nn.Linear(fcInputSize, 2),
#             nn.Linear(32, 2),
#             nn.LogSoftmax(dim=1)
#         )

#     def forward(self, x):
#         """
#         Forward pass of the network.
        
#         Args:
#             x (torch.Tensor): Input tensor.
        
#         Returns:
#             torch.Tensor: Output tensor with class scores.
#         """
#         x = self.cnn_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear_layers(x)

#         return x
