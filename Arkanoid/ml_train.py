import pickle
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import math


path = "./Arkanoid 示範/cheat_data"
allFile = os.listdir(path)
data_set = []
for file in allFile:
    with open(os.path.join(path,file),"rb") as f:
        data_set.append(pickle.load(f))
for i,data in enumerate(data_set):
    if data['1P']['scene_info'][-1]['status'] == 'GAME_OVER': # 只取用過關的資料
        data_set.pop(i-1)
print(len(data_set))     
# feature
Ball_x = []
Ball_y = []
Speed_x = []
Speed_y = []
Direction = []
PlatformX = []
Brick_info = []


for data in data_set:
    for i, sceneInfo in enumerate(data['1P']["scene_info"][2:-3]):
        Ball_x.append(data['1P']['scene_info'][i+1]["ball"][0])
        Ball_y.append(data['1P']['scene_info'][i+1]["ball"][1])
        Speed_x.append(data['1P']['scene_info'][i+1]["ball"][0] - data['1P']['scene_info'][i]["ball"][0])
        Speed_y.append(data['1P']['scene_info'][i+1]["ball"][1] - data['1P']['scene_info'][i]["ball"][1])
        PlatformX.append(data['1P']['scene_info'][i+1]["platform"][0])
        # 創建1*200陣列代表磚塊地圖
        brick_form = np.zeros(280)
        for brick in data['1P']['scene_info'][i+1]['bricks']:
            x,y = brick
            row = y // 10 
            column = x // 25
            index = 8 * (row-1) + column
            brick_form[index] = 1
        for hard_brick in data['1P']['scene_info'][i+1]['hard_bricks']:
            x,y = hard_brick
            row = y// 10 
            column = x // 25
            index = 8 * (row-1) + column
            brick_form[index] = 2
        Brick_info.append(brick_form)
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

X=np.zeros(285)
for i in range(len(Ball_x)):
    X_1=np.hstack(([Ball_x[i] ,Ball_y[i] , Speed_x[i] , Speed_y[i] , Direction[i]],Brick_info[i]))
    X=np.vstack((X, X_1))
X=X[1::] # 將第一個元素刪除


# label

Position_pred = []
PlatformX.append(70)
Ball_y.append(395)
for i,ball_y in enumerate(Ball_y): # append是為了讓最後一個是395，補足最後的數字
    if ball_y == 395:
        for n in range(len(Position_pred),i):
            Position_pred.append(PlatformX[i-1])
Position_pred = np.array(Position_pred)
Y=Position_pred
print(Y.shape)

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
with open(os.path.join(os.path.dirname(__file__),'save',"model1.pickle"),'wb') as f:
    pickle.dump(model,f)

