import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


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


def dim_reduction(reduction,kernel,train_X,test_X,train_y=None,test_y=None):
    match reduction:
        case 'pca':
            from sklearn.decomposition import KernelPCA
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
            print('Starting PCA transform.')
            #  {'linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'}
            pca = KernelPCA(n_components=100,kernel=kernel)
            train_X = pca.fit_transform(train_X)
            test_X = pca.transform(test_X)
            print('PCA transform complete')
        case 'lda':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            # solver : {'svd', 'lsqr', 'eigen'}, default='svd'
            lda = LinearDiscriminantAnalysis(solver='svd',n_components=9)
            train_X = lda.fit_transform(train_X,train_y)
            test_X = lda.transform(test_X)
        case _:
            print('Invalid reduction received. No change in dataset')
    return train_X,test_X

def sigmoid(z):
    exp_ = np.exp(z)
    return (1/sum(exp_) * exp_).reshape((len(z),1))









## Reference: https://www.programcreek.com/python/?code=oval-group%2Fdfw%2Fdfw-master%2Fexperiments%2Fdata%2Floaders.py