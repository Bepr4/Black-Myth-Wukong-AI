import pyautogui
import cv2
import time
import matplotlib.pyplot as plt
import utils.directkeys as directkeys
import numpy as np
from screen_key_grab.grabscreen import grab_screen
from screen_key_grab.getkeys import key_check
from utils.restart import restart
# from utils.death_reset_controller import DeathResetController


class Wukong(object):
    def __init__(self, observation_w, observation_h, action_dim):
        super().__init__()

        self.observation_dim = observation_w * observation_h
        self.width = observation_w
        self.height = observation_h
        self.death_cnt = 0
        self.action_dim = action_dim
        self.obs_window = (336,135,1395,795)
        self.boss_blood_window = (597, 894, 1104, 906)
        self.self_blood_window = (180, 954, 430, 970)
        self.boss_stamina_window = (345, 78, 690, 81)  # 如果后期有格挡条的boss可以用，刀郎没有
        self.self_stamina_window = (1473, 938, 1510, 1008)  # 棍势条
        self.boss_blood = 0
        self.self_blood = 0
        self.boss_stamina = 0
        self.self_stamina = 0
        self.stop = 0
        self.emergence_break = 0

        # self.low_hp_threshold = 70
        # self.low_hp_cooldown = 0
        # self.death_reset = DeathResetController(
        #     obs_window=self.obs_window,
        #     teleport_key="l",
        # )
        
    # 用canny边缘检测实现血量识别，不是特别准确，但打刀郎由于血条有特效只能用这个
    # def self_blood_count(self, obs_gray): 
    #     blurred_img = cv2.GaussianBlur(obs_gray, (3, 3), 0) 
    #     canny_edges = cv2.Canny(blurred_img, 10, 100)
    #     value = canny_edges.argmax(axis=-1)
    #     return np.max(value)
    
    # def self_blood_count(self, obs_gray):
    #     blurred = cv2.GaussianBlur(obs_gray, (3, 3), 0)
    #     edges = cv2.Canny(blurred, 10, 100)

    #     h, w = edges.shape
    #     sample_rows = edges[h // 2 - 1:h // 2 + 2]   # 只取中间 3 行
    #     positions = []

    #     for row in sample_rows:
    #         xs = np.where(row > 0)[0]
    #         if len(xs) > 0:
    #             positions.append(xs.max())  # 取最右侧边缘

    #     if not positions:
    #         return 0

    #     return int(np.median(positions))

    def self_blood_count(self, self_blood_img):
        hsv = cv2.cvtColor(self_blood_img, cv2.COLOR_BGR2HSV)

        # 取中间主体，避开最上面的高光和最下面的边框，但不要切得太薄
        roi = hsv[3:11, 4:-4]

        # 放宽阈值，避免运动/受击时白色变灰白就整帧漏掉
        lower_white = np.array([0, 0, 165], dtype=np.uint8)
        upper_white = np.array([180, 80, 255], dtype=np.uint8)
        mask = cv2.inRange(roi, lower_white, upper_white)

        # 让白条更连续，填掉 1-2 像素的小断裂
        kernel = np.ones((2, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        col_sum = (mask > 0).sum(axis=0)

        # 一列里至少有 3 个白像素才算命中
        white = (col_sum >= 3).astype(np.uint8)

        # 从左往右找“连续白条”的末端，允许少量小缺口
        end = 0
        gap = 0
        started = False

        for i, v in enumerate(white):
            if v:
                started = True
                end = i
                gap = 0
            elif started:
                gap += 1
                if gap >= 3:
                    break

        return int(end)



    def boss_blood_count(self, boss_blood_hsv_img): # 用HSV颜色空间识别血量，统计亮度在180-220之间的像素数量，作为血量值
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([360, 30, 220])
        mask = cv2.inRange(boss_blood_hsv_img, lower_white, upper_white)
        white_pixel_count = cv2.countNonZero(mask)
        return white_pixel_count

    def self_stamina_count(self, self_stamina_hsv_img):
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([360, 30, 220])
        mask = cv2.inRange(self_stamina_hsv_img, lower_white, upper_white)
        white_pixel_count = cv2.countNonZero(mask)
        return white_pixel_count

    def boss_stamina_count(self, boss_stamina_hsv_img): # 目前刀郎没有架势条（格挡条）
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([360, 30, 220])
        mask = cv2.inRange(boss_stamina_hsv_img, lower_white, upper_white)
        white_pixel_count = cv2.countNonZero(mask)
        return white_pixel_count
    
    def self_power_count(self,self_power_hsv_img): # 天命人棍势条（斜的棍势条，不包含棍势点）
        lower_white = np.array([0, 0, 220])
        upper_white = np.array([360, 45, 256])
        mask = cv2.inRange(self_power_hsv_img, lower_white, upper_white)
        white_pixel_count = cv2.countNonZero(mask)
        return white_pixel_count

    def self_endurance_count(self,obs_gray): # 天命人气力条
        blurred_img = cv2.GaussianBlur(obs_gray, (3,3), 0)
        canny_edges = cv2.Canny(blurred_img, 10, 100)
        value = canny_edges.argmax(axis=-1)
        return np.max(value)

    def take_action(self, action):
        if action == 0:  # j
            directkeys.light_attack()
        elif action == 1:  # m
            directkeys.left_dodge()
        elif action == 2:
            directkeys.sanlian()
        elif action == 3:
            directkeys.right_dodge()
        elif action == 4:
            directkeys.hard_attack()
        elif action == 5:
            directkeys.stay_still()
        elif action == 6:
            directkeys.ding_shen_gong_ji()
        elif action == 7:
            directkeys.kan_po()  # 轻棍+识破



    def get_reward(self, boss_blood, next_boss_blood, self_blood, next_self_blood,
                   boss_stamina, next_boss_stamina, self_stamina, next_self_stamina,
                   stop, emergence_break, action, boss_attack):
        print(next_self_blood, boss_blood)
        if next_self_blood < 70:     # self dead 用hsv识别则量值大约在400，用canny大约在40
            print("dead")
            print("快死了，当前血量：",self_blood,"马上血量：",next_self_blood)
            reward = -1  # 快死了，惩罚
            done = 0  # 还没有死，等真正死了再重置
            stop = 0
            emergence_break += 1
            # if self.death_cnt <= 2:
            #     self.death_cnt += 1
            print("后跳并喝血")
            pyautogui.keyDown('S')
            directkeys.dodge()
            directkeys.dodge()
            directkeys.dodge()
            time.sleep(0.2)
            pyautogui.press('R')
            time.sleep(1)
            pyautogui.press('R')
            pyautogui.press('R')
            pyautogui.keyUp('S')
            time.sleep(1)
            #else:
            #    pass
            
            # 用风灵月影增加训练效率
            # pyautogui.keyDown('num2')
            # pyautogui.keyDown('num2')
            # pyautogui.keyDown('num2') 
            # time.sleep(1)
            # pyautogui.keyUp('num2') 
            return reward, done, stop, emergence_break

        else:
            reward = 0
            self_blood_reward = 0
            boss_blood_reward = 0
            self_stamina_reward = 0
            if next_self_blood - self_blood < -5:  # 如果掉血
                self_blood_reward = (next_self_blood - self_blood) // 10  # 惩罚，掉血越多惩罚越大
                print("掉血惩罚")
                time.sleep(0.05)  
                # 防止连续取帧时一直计算掉血
            if next_boss_blood - boss_blood <= -18: # 如果boss掉血了，奖励，打掉越多奖励越大
                print("打掉boss血而奖励")
                boss_blood_reward = (boss_blood - next_boss_blood) // 5
                boss_blood_reward = min(boss_blood_reward, 20)

            if (action == 1 or action == 3) and boss_attack == True and next_self_stamina - self_stamina >= 7 and next_self_blood-self_blood == 0:
                print("完美闪避奖励")
                self_stamina_reward += 2
            elif (action == 1 or action == 3) and boss_attack == True and next_self_blood-self_blood == 0:
                print("成功闪避")
                self_stamina_reward += 0.5
            reward = reward + self_blood_reward * 0.8 + \
                boss_blood_reward * 1.2 + self_stamina_reward * 1.0  # 最终加权，boss掉血奖励权重大一些，闪避奖励权重小一些
            done = 0  # 还没有死
            emergence_break = 0 
            return reward, done, stop, emergence_break

    def step(self, action, boss_attack):
        if (action == 0):
            print("一连")
        elif (action == 1):
            print("左闪避")
        elif (action == 2):
            print("三连")
        elif action == 3:
            print("右闪避")
        elif action == 4:
            print("重棍")
        elif action == 5:
            print("气力不足，歇脚一歇")
        elif action == 6:
            print("定！五连绝世！")
        elif action == 7:
            print("轻棍+识破")
        self.take_action(action) # 执行动作

        obs_screen = grab_screen(self.obs_window) # 截图
        obs_resize = cv2.resize(obs_screen, (self.width, self.height)) # 统一尺寸
        obs = np.array(obs_resize).reshape(-1, self.height, self.width, 4)[0] # 这里的obs是下一帧的图像，作为状态输入给DQN
        # 血量统计
        self_blood_img = grab_screen(self.self_blood_window)
        #self_blood_hsv_img = cv2.cvtColor(self_blood_img, cv2.COLOR_BGR2HSV) #转HSV，方便统计亮度
        # 如果是有血量上的特效的，用self_blood_count，否则boss_blood_count更准确
        # self_blood_gray_img = cv2.cvtColor(self_blood_img, cv2.COLOR_BGR2GRAY)
        next_self_blood = self.self_blood_count(self_blood_img) # 已修改                      # 这里Boss的统计方法和自身是一致的
        # boss血量统计
        boss_blood_img = grab_screen(self.boss_blood_window)
        boss_blood_hsv_img = cv2.cvtColor(boss_blood_img, cv2.COLOR_BGR2HSV)
        next_boss_blood = self.boss_blood_count(boss_blood_hsv_img)
        # 棍势统计
        self_stamina_img = grab_screen(self.self_stamina_window)
        self_stamina_hsv_img = cv2.cvtColor(
            self_stamina_img, cv2.COLOR_BGR2HSV)
        next_self_stamina = self.self_stamina_count(self_stamina_hsv_img)
        # boss架势条统计 ？？？
        boss_stamina_img = grab_screen(self.boss_stamina_window)
        boss_stamina_hsv_img = cv2.cvtColor(
            boss_stamina_img, cv2.COLOR_BGR2HSV)
        next_boss_stamina = self.self_stamina_count(boss_stamina_hsv_img)
        reward, done, stop, emergence_break = self.get_reward(self.boss_blood, next_boss_blood, self.self_blood, next_self_blood,
                                                              self.boss_stamina, next_boss_stamina, self.self_stamina, next_self_stamina,
                                                              self.stop, self.emergence_break, action, boss_attack)
        self.self_blood = next_self_blood
        self.boss_blood = next_boss_blood
        self.self_stamina = next_self_stamina
        self.boss_stamina = next_boss_stamina
        return (obs, reward, done, stop, emergence_break)

    def pause_game(self, paused): # 用于训练中暂停
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('start game')
                time.sleep(1)
            else:
                paused = True
                print('pause game')
                time.sleep(1)
        if paused:
            print('paused')
            while True:
                keys = key_check()
                if 'T' in keys:
                    if paused:
                        paused = False
                        print('start game')
                        time.sleep(1)
                        break
                    else:
                        paused = True
                        time.sleep(1)
        return paused

    def reset(self, initial=False): 
        restart(initial) # 重置环境，死了就重置，初始状态不需要重置
        obs_screen = grab_screen(self.obs_window)
        obs_resize = cv2.resize(obs_screen, (self.width, self.height))
        obs = np.array(obs_resize).reshape(-1, self.height, self.width, 4)[0] # 这里的obs是重置后第一帧的图像，作为状态输入给DQN
        return obs
