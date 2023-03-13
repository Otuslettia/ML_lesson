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

def find_collided_brick(x, y, s_x, s_y, bricks):
    min_distance = float('inf')
    collided_brick = None
    r = 2.5
    for brick in bricks:
        b_x, b_y = brick
        brick_width, brick_height = 25, 10
        distance = ((x-b_x)**2 + (y-b_y)**2)**0.5
        if distance <= r + brick_width/2:
            return brick
        if (x-b_x)*s_x <= 0 or (y-b_y)*s_y <= 0:
            continue
        time = min(abs((x-b_x)/s_x), abs((y-b_y)/s_y))
        collision_x = x + s_x * time
        collision_y = y + s_y * time
        dx = abs(collision_x - b_x) - brick_width/2
        dy = abs(collision_y - b_y) - brick_height/2
        if dx > r or dy > r:
            continue
        if time < min_distance:
            min_distance = time
            collided_brick = brick
    return collided_brick

# feature
Ball_x = []
Ball_y = []
Speed_x = []
Speed_y = []
Direction = []
Collided_brick_x = []
Collided_brick_y = []


for data in data_set:
    for i, sceneInfo in enumerate(data['1P']["scene_info"][2:]):
        Ball_x.append(data['1P']['scene_info'][i+1]["ball"][0])
        Ball_y.append(data['1P']['scene_info'][i+1]["ball"][1])
        Speed_x.append(data['1P']['scene_info'][i+1]["ball"][0] - data['1P']['scene_info'][i]["ball"][0])
        Speed_y.append(data['1P']['scene_info'][i+1]["ball"][1] - data['1P']['scene_info'][i]["ball"][1])
        if Speed_x[-1] > 0:
            if Speed_y[-1] > 0:
                Direction.append(0)
            else:
                Direction.append(1)
        else:
            if Speed_y[-1] > 0:
                Direction.append(2)
            else:
                Direction.append(3)
            
        bricks = data['1P']['scene_info'][i+1]['bricks']
        hard_bricks = data['1P']['scene_info'][i+1]['hard_bricks']
        bricks.extend(hard_bricks)
        
        next_brick = [0, 0]
        next_brick = find_collided_brick(data['1P']['scene_info'][i+1]["ball"][0],
                            data['1P']['scene_info'][i+1]["ball"][1], 
                            data['1P']['scene_info'][i+1]["ball"][0] - data['1P']['scene_info'][i]["ball"][0],
                            data['1P']['scene_info'][i+1]["ball"][1] - data['1P']['scene_info'][i]["ball"][1],
                            bricks)
        if next_brick is not None:
            Collided_brick_x.append(next_brick[0])
            Collided_brick_y.append(next_brick[1])
        else:
            Collided_brick_x.append(-1)
            Collided_brick_y.append(-1)

#X_labeled = np.column_stack((Ball_x, Ball_y, Speed_x, Speed_y, Direction))
X_unlabeled = np.column_stack((Ball_x, Ball_y, Speed_x, Speed_y, Direction, Collided_brick_x, Collided_brick_y))
m, n = X_unlabeled.shape
X_labeled = np.zeros((m, n))
X_labeled[:, :-2] = X_unlabeled[:, :-2]
X_labeled[:, -2:] = -1
"""
X=np.array([0,0,0,0,0])
for i in range(len(Ball_x)):
    X=np.vstack((X, [Ball_x[i] ,Ball_y[i] , Speed_x[i] , Speed_y[i] , Direction[i]]))
X=X[1::]
"""

# label
Position_pred = []
platform_position_y = 400
ball_speed_y = 7
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
Y_labeled = Position_pred

# training
x_train_l, x_test_l, y_train_l, y_test_l = train_test_split(X_labeled, Y_labeled, test_size=0.2)
model = DecisionTreeRegressor(criterion='squared_error', max_depth=8000, splitter='best')
model.fit(x_train_l, y_train_l)
print('Score on labeled data:', model.score(x_test_l, y_test_l))

"""
x_train , x_test , y_train, y_test= train_test_split(X,Y,test_size=0.2)

model=DecisionTreeRegressor(criterion='squared_error',max_depth=8000,splitter='best')
model.fit(x_train,y_train)

# evaluation
y_predict=model.predict(x_test)
mse=mean_squared_error(y_test, y_predict)
print(mse)
rmse = math.sqrt(mse)
print("RMSE=%.2f" % rmse)
"""
y_pred_unlabeled = model.predict(X_unlabeled)
Y_labeled = np.concatenate((Y_labeled, y_pred_unlabeled))
x_train, x_test, y_train, y_test = train_test_split(np.concatenate((X_labeled, X_unlabeled)), Y_labeled, test_size=0.2)
model.fit(x_train, y_train)
print('Score on labeled and unlabeled data:', model.score(x_test, y_test))
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

