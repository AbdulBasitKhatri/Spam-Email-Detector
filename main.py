import pandas as pd
from sklearn.model_selection import train_test_split as tTS
from sklearn.feature_extraction.text import TfidfVectorizer as tfIdfV
from sklearn.linear_model import LogisticRegression as logRegr
from sklearn.metrics import accuracy_score

def train(showAccuracy: bool):
    dataset = pd.read_csv("emailDataset.csv")
    xTrain,xTest,yTrain,yTest = tTS(dataset["Message"], dataset["Category"], test_size=0.3, random_state=10)
    features = tfIdfV(min_df=1, stop_words='english', lowercase=True)
    xTrainFeatures = features.fit_transform(xTrain)
    xTestFeatures = features.transform(xTest)
    yTrain = yTrain.astype('int')
    yTest = yTest.astype('int')
    model = logRegr()
    model.fit(xTrainFeatures, yTrain)
    if showAccuracy:
        trainResult = model.predict(xTrainFeatures)
        trainAccuracy = accuracy_score(yTrain, trainResult)
        print(f'Training accuracy: {trainAccuracy*100:.3f}%')
        testResult = model.predict(xTestFeatures)
        testAccuracy = accuracy_score(yTest, testResult)
        print(f'Testing accuracy: {testAccuracy*100:.3f}%')
    return features,model

if __name__ == "__main__":
    features, model = train(True)
    while True:
        inpMail = [input(">> ")]
        if "exit" in inpMail:
            break
        inpFeatures = features.transform(inpMail)
        result = model.predict(inpFeatures)
        if result[0] == 0:
            print('Mail is ham')
        else:
            print('Mail is spam')
