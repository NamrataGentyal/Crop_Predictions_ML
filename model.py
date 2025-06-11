import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('Crop_recommendation.csv')

data.isnull().sum()
data.shape

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

model = RandomForestClassifier()

model.fit(x_train, y_train)

# predictions = model.predict(x_test)

pickle.dump(model, open("model.pkl", "wb"))


accuracy = model.score(x_test,y_test)

# print("Accuracy :" ,  accuracy)
# new_feature = [[78,42,42,20.130175,81.604873,7.628473,262.717340]]
# predicted_crop = model.predict(new_feature)
# print("predicted crop is:" ,  predicted_crop)