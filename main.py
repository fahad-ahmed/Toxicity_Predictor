# This is a sample Python script.
import os

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
global workingDataset

def LoadData(path):
    global dose_level, dose, fullmat, time, y
    print("Dataset = ", path)
    dose_level = pd.read_csv(path+"dose_level", header=None)
    dose = pd.read_csv(path+"dose", header=None)
    fullmat = pd.read_csv(path+"fullmat", header=None)
    time = pd.read_csv(path+"time", header=None)
    y = pd.read_csv(path+"y", header=None)

def TransposeMatrix():
    global fullmat_df, i
    fullmat_df = pd.DataFrame()
    for i in range(len(fullmat)):
        l = fullmat[0][i].split(' ')
        fullmat_df[l[0]] = l[1:]

def ConcatColumns():
    global data
    data = pd.concat([dose_level, dose, time, y], axis=1)
    data.columns = ['dose_level', 'dose', 'time', 'y']
    data = pd.concat([data, fullmat_df], axis=1)

def ApplyLevelEncoder():
    encoder = preprocessing.LabelEncoder()
    data['dose_level'] = encoder.fit_transform(data['dose_level'])
    data['time'] = encoder.fit_transform(data['time'])



def SplitTrainTest():
    global train_x, train_y, test_x, test_y
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    train_x = train.drop('y', axis=1)
    train_y = train['y']
    test_x = test.drop('y', axis=1)
    test_y = test['y']


def Apply_SVC():
    global model, pred
    model = SVC()
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    print("SVC Accuracy: ", accuracy_score(test_y, pred))
    print("SVC F1 Score: ", f1_score(test_y, pred))
    Plot_Confusion_Matrix("SVC")

def Apply_Random_Forest():
    global model, pred
    model = RandomForestClassifier(max_depth=3, random_state=0)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    print("RFC Accuracy: ", accuracy_score(test_y, pred))
    print("RFC F1 Score: ", f1_score(test_y, pred))
    Plot_Confusion_Matrix("RFC")

def Apply_Gaussian_NB():
    global model, pred
    model = GaussianNB()
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    print("GNB Accuracy: ", accuracy_score(test_y, pred))
    print("GNB F1 Score: ", f1_score(test_y, pred))
    Plot_Confusion_Matrix("GNB")

def Plot_Confusion_Matrix(fileName):
    global i
    conf_matrix = confusion_matrix(y_true=test_y, y_pred=pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix '+fileName, fontsize=18)
    # plt.show()
    filepath = workingDataset+"ConfusionMatrix_"+fileName+".png"
    if os.path.exists(filepath):
        os.remove(filepath)
    plt.savefig(filepath)


Datasets = ["Rat/in_vivo/Kidney/Repeat/", "Rat/in_vivo/Kidney/Single/", "Rat/in_vivo/Liver/Repeat/", "Rat/in_vivo/Liver/Single/"]


if os.path.exists('Rat/in_vivo/') == False:
    print("No Dataset found. Make sure  Rat/in_vivo  dataset is available along with the program")
    exit()

print("It may take 4-5 minutes to run the whole program. So please Wait a bit")

for path in Datasets:
    print("------------------------------------------------")
    workingDataset = path
    LoadData(path)
    TransposeMatrix()
    ConcatColumns()
    ApplyLevelEncoder()
    SplitTrainTest()
    Apply_SVC()
    Apply_Random_Forest()
    Apply_Gaussian_NB()
    print("------------------------------------------------")

