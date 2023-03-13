import os
import pickle
import numpy as np

class MLPlay:

    def __init__(self, *args, **kwargs):
        """
        Constructor
        """
        self.ball_served = False
        self.previous_ball = (0, 0)
       

        with open(os.path.join(os.path.dirname(__file__),'save','model1.pickle'),'rb') as f:
            self.model = pickle.load(f)
    
    def update(self, scene_info, *args, **kwargs):
        if(scene_info["status"]== "GAME_OVER" or scene_info["status"]=="GAME_PASS"):
            return "RESET"
        
        if not self.ball_served:
            self.ball_served = True
            command = "SERVE_TO_RIGHT"
        else:
            Ball_x = scene_info["ball"][0]
            Ball_y = scene_info["ball"][1]
            Speed_x = scene_info["ball"][0] - self.previous_ball[0]
            Speed_y = scene_info["ball"][1] - self.previous_ball[1]
            brick_form = np.zeros(280)
            for brick in scene_info['bricks']:
                x,y = brick
                row = y // 10 
                column = x // 25
                index = 8 * (row-1) + column
                brick_form[index] = 1
            for hard_brick in scene_info['hard_bricks']:
                x,y = hard_brick
                row = y// 10 
                column = x // 25
                index = 8 * (row-1) + column
                brick_form[index] = 2
            if Speed_x > 0:
                if Speed_y > 0:
                    Direction = 0
                else:
                    Direction = 1
            else:
                if Speed_y > 0:
                    Direction = 2
                else:
                    Direction = 3
            
            x=np.array(np.hstack(([Ball_x,Ball_y,Speed_x,Speed_y,Direction],brick_form[145:]))).reshape((1,-1))
            y=self.model.predict(x)
            if scene_info["platform"][0] + 10 +5 < y:
                command = "MOVE_RIGHT"
            elif scene_info["platform"][0] + 10 -5 > y:
                command = "MOVE_LEFT"
            else:
                command = "NONE"
        
        self.previous_ball = scene_info["ball"]
        return command
    
    def reset(self):
        self.ball_served = False