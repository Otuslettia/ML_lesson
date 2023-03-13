
"""
The template of the main script of the machine learning process
"""
import pickle
import os
import time

class MLPlay:
    def __init__(self, *args, **kwargs):
        """
        Constructor
        """
        self.ball_served = False
        self.previous_ball = (0, 0)  
        self.platform_y = 400           # Position of board y axis
        self.ball_speed_y = 7           # Ball vertical speed
        self.window_width = 200         # Width of the game window 
        self.platform_x = 75           # 球的移動方向 False 為往上，True為往下，代表要接球
        self.platform_width = 40
        self.pred = [200]
        self.count = 0

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

    def update(self, scene_info, *args, **kwargs):
        
        if scene_info["status"] == "GAME_OVER" :
            for i in range(self.count-len(self.pred)-1):
                platformx = self.pred[-1]
                self.pred.append(platformx)
            self.pred.append(self.previous_ball[0])
            self.count = 0 # 歸零
            self.clear_list()
            return "RESET"
        elif scene_info["status"] == "GAME_PASS":
            self.count = 0
            self.record(scene_info, 'NONE')
            self.flush_to_file()
        
        if not self.ball_served:            
            self.ball_served = True
            self.previous_ball = scene_info["ball"]
            command = "SERVE_TO_LEFT"      # You can change the direction to serve ball
        else:
            if self.previous_ball[1] == 395 and scene_info['ball'][1]-self.previous_ball[1] < 0:  # 球y軸為395，且方向向上
                self.count += 1
            if self.count > len(self.pred):
                command = 'NONE'
            else:
                if self.pred[self.count-1]-18 > scene_info['platform'][0]:  # -20讓球不要在邊邊被打到
                    command = 'MOVE_RIGHT'
                elif self.pred[self.count-1]-12 < scene_info['platform'][0]:
                    command = 'MOVE_LEFT'
                else:
                    command = 'NONE'
        self.previous_ball = scene_info["ball"]

        # Pass scene_info and command to generate data
        self.record(scene_info, command)
        return command

    def reset(self):
        """
        Reset the status
        """
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
        if not os.path.exists(os.path.dirname(__file__) + "/smart_data"):
            os.makedirs(os.path.dirname(__file__) + "/smart_data")
        filepath = os.path.join(os.path.dirname(__file__),"./smart_data/" + filename)
        # Write pickle file
        with open(filepath, "wb") as f:
            pickle.dump(self._game_progress, f)
        self.clear_list()

    def clear_list(self):
        # Clear list
        for name in self._ml_names:
            target_slot = self._game_progress[name]
            target_slot["scene_info"].clear()
            target_slot["command"].clear()