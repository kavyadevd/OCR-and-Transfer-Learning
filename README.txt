# Code Solutions:

## Part I HandwrittenDigitRecognition

#### 1. SVM: SVM.ipynb

#### 2. Logistic Regression: LogisticRegression.ipynb

#### 3. Deep Learning: CNN.ipynb

__________________________________________________________

### Part II Transfer Learning

#### 1. Simple CNN: MonkeyCNN.ipynb

#### 2. ResNet50: MonkeyResNet.ipynb


The Folder structure is as follows

```
project
│   README.md
│   README.txt    
│   Report.pdf
|
└───Code
│   │   requirements.txt
│   │
│   └───HandwrittenDigitRecognition
|   |   |
|   |   └───dataset
|   │   │   SVM.ipynb
|   │   │   LogisticRegression.ipynb
|   |   |   CNN.ipynb
|   |   |   utils.py
|   │   
|   └───TransferLearning
|   |   |
|   |   └───monkey_dataset
|   │   MonkeyCNN.ipynb
|   |   MonkeyResNet.ipynb
|   |   constants.py
|   |   utils.py
```

## To run the notebooks:
Run the following command in terminal ad copy the datasets to the repective folders. To automatically download the dataset, in any notebook uncomment the ~train_load,test_load = load_dataset(False)~ line and run the cell

```bash
pip install -r <project-root>/Code/requirements.txt
mkdir <project-root>/Code/HandwrittenDigitRecognition/dataset
mkdir <project-root>/Code/TransferLearning/monkey_dataset
```

Notebooks are already executed and conain the result. To rename them again, open in any compatible editor eg. VSCode and click runall pr Fn5 hotkey.

Full code with results can be viewed at https://github.com/kavyadevd/OCR-and-Transfer-Learning
