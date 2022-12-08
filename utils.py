import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt


def load_dataset(output=False):
    print('Generating train and test loaders:')
    # 0.1307 and 0.3081 = Global mean and SD MNIST dataset
    norm = transforms.Normalize((0.1307,), (0.3081,))
    # (class) Compose(transforms: list[ToTensor | Normalize])
    transform_lst = transforms.Compose([transforms.ToTensor(),norm])
    # MNIST(root: str, train: bool = True, transform: ((...) -> Any) | None = None, target_transform: ((...) -> Any) |
    #  None = None, download: bool = False) -> None
    ### Train
    dataset = datasets.MNIST(root='./dataset/',transform=transform_lst,download=True)
    print('Train Dataset download complete')
    train_loader = torch.utils.data.DataLoader(dataset,batch_size=128,shuffle = True)
    print('Train Dataset download complete')
    ### Test
    dataset = datasets.MNIST(root='./dataset/',transform=transform_lst,train=False,download=True)
    print('Train Dataset download complete')
    test_loader = torch.utils.data.DataLoader(dataset,batch_size=128,shuffle = True)
    print('Train Dataset download complete')
    if output:
        print('_'*10,'\nVisualizing few downloaded images')
        images_ = enumerate(test_loader)
        data_index, (show_data, show_labels) = next(images_)
        fig = plt.figure()
        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.imshow(show_data[i][0],interpolation='none')
            plt.title(show_labels[i])
        plt.show

    return train_loader,test_loader










## Reference: https://www.programcreek.com/python/?code=oval-group%2Fdfw%2Fdfw-master%2Fexperiments%2Fdata%2Floaders.py