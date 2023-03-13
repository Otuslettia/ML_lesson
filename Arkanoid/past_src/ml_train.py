import pickle
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import math


path = ".\log"
allFile = os.listdir(path)
data_set = []
for file in allFile:
    with open(os.path.join(path,file),"rb") as f:
        data_set.append(pickle.load(f))

# feature
max_bricks = 100  # set a fixed size for the bricks array
max_hard_bricks = 100  # set a fixed size for the hard bricks array
X = np.zeros((1, 7 + max_bricks + max_hard_bricks))

for data in data_set:
    bricks = np.array(data['1P']['scene_info'][1]['bricks'])
    hard_bricks = np.array(data['1P']['scene_info'][1]['hard_bricks'])

    for i, scene_info in enumerate(data['1P']['scene_info'][2:]):
        ball_x = scene_info['ball'][0]
        ball_y = scene_info['ball'][1]
        speed_x = scene_info['ball'][0] - data['1P']['scene_info'][i+1]['ball'][0]
        speed_y = scene_info['ball'][1] - data['1P']['scene_info'][i+1]['ball'][1]

        # pad the bricks and hard bricks arrays with zeros if necessary
        padded_bricks = np.pad(bricks, (0, max_bricks - len(bricks)), mode='constant')
        padded_hard_bricks = np.pad(hard_bricks, (0, max_hard_bricks - len(hard_bricks)), mode='constant')

        if speed_x > 0:
            if speed_y > 0:
                direction = 0
            else:
                direction = 1
        else:
            if speed_y > 0:
                direction = 2
            else:
                direction = 3

        # stack all the features together
        features = np.hstack((ball_x, ball_y, speed_x, speed_y, direction, padded_bricks, padded_hard_bricks))
        X = np.vstack((X, features))

        # update the bricks and hard bricks arrays for the next iteration
        bricks = np.array(scene_info['bricks'])
        hard_bricks = np.array(scene_info['hard_bricks'])

X = X[1:]  # remove the first row of zeros

"""
X=np.array([0,0,0,0,0,[],[]])
for i in range(len(Ball_x)):
    X=np.vstack((X, [Ball_x[i] ,Ball_y[i] , Speed_x[i] , Speed_y[i] , Direction[i], Bricks[i], Hard_bricks[i]]))
X=X[1::]
"""

# label
Position_pred = []
platform_position_y = 400
ball_speed_y = abs(speed_y)
platform_width = 200


for i in range(len(Ball_x)):
    pred = Ball_x[i] + ((platform_position_y - Ball_y[i])//ball_speed_y) * Speed_x[i]

    section = (pred // platform_width)
    if (section % 2==0):
        pred =abs(pred - platform_width*section)
    else:
        pred =platform_width - abs(pred -platform_width*section)

    Position_pred.append(pred)

Position_pred = np.array(Position_pred)
Y=Position_pred

# training
x_train , x_test , y_train, y_test= train_test_split(X,Y,test_size=0.2)

model=DecisionTreeRegressor(criterion='squared_error',max_depth=8000,splitter='best')
model.fit(x_train,y_train)

# evaluation
y_predict=model.predict(x_test)
mse=mean_squared_error(y_test, y_predict)
print(mse)
rmse = math.sqrt(mse)
print("RMSE=%.2f" % rmse)

# save model
if not os.path.exists(os.path.dirname(__file__) + "/save"):
    os.makedirs(os.path.dirname(__file__) + "/save")
with open(os.path.join(os.path.dirname(__file__),'save',"model.pickle"),'wb') as f:
    pickle.dump(model,f)

