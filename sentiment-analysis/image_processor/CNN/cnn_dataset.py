from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import cv2

class CNNDataset():

    dataset_dir = 'images/'
    label_map = {'negative': 0,
                'positive': 1}
    RANDOM_STATE = 42
    img_size = 28
    class_to_skip = 'neutral'

    def __init__(self):
        first_image = True
        no_classes = len(os.listdir(self.dataset_dir))
        for class_dir in os.listdir(self.dataset_dir):

            # if class_dir == self.class_to_skip:
            #     continue

            label = int(self.label_map[class_dir])

            image_dir = self.dataset_dir + class_dir + '/' 
            for image_file in os.listdir(image_dir):
                image = mpimg.imread(image_dir + image_file)
                # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                if first_image == True:
                    # self.image_width = image.shape[0]
                    # print(image.shape[0])
                    # print(image.shape[1])
                    # self.image_height = image.shape[1]
                    self.images = np.array([image])
                    self.labels = np.array([label])
                    first_image = False
                else:
                    self.images = np.vstack([self.images, [image]])
                    self.labels = np.append(self.labels, label)

        self.no_images = self.images.shape[0]

        #shuffle data
        # random_idx = np.random.permutation(len(self.images))
        # self.images = self.images[random_idx]
        # self.labels = self.labels[random_idx]


    def split_data(self):
        train_images, temp_images, train_labels, temp_labels = train_test_split(self.images, self.labels,
                                                                        random_state=self.RANDOM_STATE,
                                                                        test_size=0.4)

        test_images, val_images, test_labels, val_labels = train_test_split(temp_images, temp_labels,
                                                                    random_state=self.RANDOM_STATE,
                                                                    test_size=0.5)
        train_images = torch.tensor(train_images / 255.0, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.long)

        val_images = torch.tensor(val_images / 255.0, dtype=torch.float32)
        val_labels = torch.tensor(val_labels, dtype=torch.long)

        test_images = torch.tensor(test_images / 255.0, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        train_images = train_images.view(train_images.shape[0], 1, train_images.shape[2], train_images.shape[1])
        val_images = val_images.view(val_images.shape[0], 1, val_images.shape[2], val_images.shape[1])
        test_images = test_images.view(test_images.shape[0], 1, test_images.shape[2], test_images.shape[1])

        train_images = F.interpolate(train_images, size=(self.img_size, self.img_size), mode='nearest')
        val_images = F.interpolate(val_images, size=(self.img_size, self.img_size), mode='nearest')
        test_images = F.interpolate(test_images, size=(self.img_size, self.img_size), mode='nearest')

        self.image_width = self.img_size
        self.image_height = self.img_size

        print(train_images.shape)
        print(val_images.shape)
        print(test_images.shape)

        # num_images_to_plot = 10
        # fig, axes = plt.subplots(1, num_images_to_plot, figsize=(15, 3))

        # Iterate over the first 10 images
        # for i in range(num_images_to_plot):
        #     # Get the ith image and its label
        #     image = train_images[i].squeeze().numpy()  # Convert torch tensor to NumPy array and remove the channel dimension
        #     label = train_labels[i].item()  # Get the label value
            
        #     # Plot the image
        #     axes[i].imshow(image, cmap='gray')
        #     axes[i].set_title(f"Label: {label}")
        #     axes[i].axis('off')

        # plt.tight_layout()
        # plt.show()

        # image = train_images[0].squeeze().numpy()  # Assuming images are grayscale

        # # Plot the image
        # plt.imshow(image, cmap='gray')  # Assuming images are grayscale
        # plt.show()

        return train_images, val_images, test_images, train_labels, val_labels, test_labels

    def prepare_dataset(self):
        train_images, val_images, self.test_images, self.train_labels, val_labels, self.test_labels  = self.split_data()

        # Convert data to DataLoader
        batch_size = 32
        self.train_dataset = torch.utils.data.TensorDataset(train_images, self.train_labels)
        self.val_dataset = torch.utils.data.TensorDataset(val_images, val_labels)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_images, self.test_labels)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

# cnn_dataset = CNNDataset()
# cnn_dataset.prepare_dataset()
