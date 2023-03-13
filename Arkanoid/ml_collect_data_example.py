
"""
The template of the main script of the machine learning process
"""
import pickle
import os
import time
import random

class MLPlay:
    def __init__(self, *args, **kwargs):
        """
        Constructor
        """
        self.ball_served = False
        self.previous_ball = (0, 0)     
        self.pred = 100                 # Prediction of board x axis location
        self.platform_y = 400           # Position of board y axis
        self.ball_speed_y = 7           # Ball vertical speed
        self.window_width = 200         # Width of the game window 
        
        self._ml_names = ["1P"]
        game_progress = {
            "record_format_version": 2
        }
        for name in self._ml_names:
            game_progress[name] = {
                "scene_info": [],
                "command": []
            }
        self._game_progress = game_progress

    def generate_random_number(self):
        return random.randint(-20, 20)

    def update(self, scene_info, *args, **kwargs):
        
        if (scene_info["status"] == "GAME_OVER" or
                scene_info["status"] == "GAME_PASS"):
            return "RESET"

        if not self.ball_served:            
            self.ball_served = True
            self.previous_ball = scene_info["ball"]
            command = "SERVE_TO_RIGHT"      # You can change the direction to serve ball
        else:
            scene_info["status"] = "GAME_ALIVE"
            Ball_x = scene_info["ball"][0]
            Ball_y = scene_info["ball"][1]
            Speed_x = scene_info["ball"][0] - self.previous_ball[0]
            Speed_y = scene_info["ball"][1] - self.previous_ball[1]
            windows_y_length = 400
            if (Speed_y == 0): Speed_y = 1
            x_p = Ball_x + ((windows_y_length - Ball_y)//Speed_y) * Speed_x
            section = (x_p // self.window_width)
            if (section % 2 == 0):
                x_p = abs(x_p - self.window_width*section)
            else : x_p = self.window_width - abs(x_p - self.window_width*section)
            ran = self.generate_random_number()
            x_p += ran
            if (scene_info["platform"][0]) < x_p :
                command = "MOVE_RIGHT"
            elif (scene_info["platform"][0]) > x_p:
                command = "MOVE_LEFT"
            else:
                command = "NONE"
            pass
        self.previous_ball = scene_info["ball"]
        # Pass scene_info and command to generate data
        self.record(scene_info, command)
        return command

    def reset(self):
        """
        Reset the status
        """
        self.flush_to_file()
        self.ball_served = False

    def record(self, scene_info_dict: dict, cmd_dict: dict):
        """
        Record the scene information and the command
        The received scene information will be stored in a list.
        """
        for name in self._ml_names:
            self._game_progress[name]["scene_info"].append(scene_info_dict)
            self._game_progress[name]["command"].append(cmd_dict)

    def flush_to_file(self):
        """
        Flush the stored objects to the file
        """
        filename = time.strftime("%Y-%m-%d_%H-%M-%S") + ".pickle"
        if not os.path.exists(os.path.dirname(__file__) + "/log"):
            os.makedirs(os.path.dirname(__file__) + "/log")
        filepath = os.path.join(os.path.dirname(__file__),"./log/" + filename)
        # Write pickle file
        with open(filepath, "wb") as f:
            pickle.dump(self._game_progress, f)

        # Clear list
        for name in self._ml_names:
            target_slot = self._game_progress[name]
            target_slot["scene_info"].clear()
            target_slot["command"].clear()