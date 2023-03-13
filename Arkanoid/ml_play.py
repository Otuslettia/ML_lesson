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
       

        with open(os.path.join(os.path.dirname(__file__),'model.pickle'),'rb') as f:
            self.model = pickle.load(f)
    
    def find_collided_brick(self, x, y, s_x, s_y, bricks):
        if bricks == None:
            return None
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
            
            next_brick = [0, 0]
            next_brick = self.find_collided_brick(
                Ball_x, Ball_y, Speed_x, Speed_y, scene_info["bricks"].extend(scene_info["hard_bricks"]))
            if next_brick is not None:
                Collided_brick_x = next_brick[0]
                Collided_brick_y = next_brick[1]
            else:
                Collided_brick_x = Collided_brick_y = -1

            x = np.column_stack((Ball_x, Ball_y, Speed_x, Speed_y, Direction, Collided_brick_x, Collided_brick_y))
            y=self.model.predict(x)
            if scene_info["platform"][0] + 20 +5 < y:
                command = "MOVE_RIGHT"
            elif scene_info["platform"][0] + 20 -5 > y:
                command = "MOVE_LEFT"
            else:
                command = "NONE"
        
        self.previous_ball = scene_info["ball"]
        return command
    
    def reset(self):
        self.ball_served = False