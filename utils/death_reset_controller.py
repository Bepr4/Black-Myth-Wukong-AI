import time

import cv2
import pyautogui

from screen_key_grab.grabscreen import grab_screen


class DeathResetController:
    def __init__(
        self,
        obs_window,
        death_dark_threshold=28.0,
        death_dark_ratio_threshold=0.65,
        death_confirm_frames=2,
        self_blood_dead_threshold=20,
        zero_blood_confirm_frames=2,
        respawn_bright_threshold=30.0,
        respawn_confirm_frames=3,
        respawn_timeout=25.0,
        post_respawn_delay=2.0,
        teleport_key="l",
        post_teleport_delay=0.5,

        lock_on_key="0",
        post_lock_on_delay=0.8,
        delay=10
    ):
        # 主观察窗口，用来判断当前画面是黑屏死亡还是已经重新出生
        self.obs_window = obs_window

        # 真死判定参数：
        # mean_brightness <= death_dark_threshold 说明整体画面足够黑
        # dark_ratio >= death_dark_ratio_threshold 说明大部分像素都很黑
        self.death_dark_threshold = death_dark_threshold
        self.death_dark_ratio_threshold = death_dark_ratio_threshold

        # 连续多少帧都满足“黑屏 + 血条消失”，才认为真的死了
        self.death_confirm_frames = death_confirm_frames

        # 自身血条低于这个值，才允许参与“真死”判定
        # 这样可以避免只是过场变暗时被误判成死亡
        self.self_blood_dead_threshold = self_blood_dead_threshold

        # 重生检测参数：
        # 重新出生后画面会明显变亮，连续几帧都亮起来后再执行传送
        self.respawn_bright_threshold = respawn_bright_threshold
        self.respawn_confirm_frames = respawn_confirm_frames

        # 最长等待多久，避免卡死在等待循环里
        self.respawn_timeout = respawn_timeout

        # 检测到已经出生后，再额外等一会儿，确保角色已经可操作
        self.post_respawn_delay = post_respawn_delay

        # 传送 mod 绑定的按键，默认是 L
        self.teleport_key = teleport_key

        # 按下传送键之后，给游戏一点时间完成传送
        self.post_teleport_delay = post_teleport_delay
        self.zero_blood_confirm_frames = zero_blood_confirm_frames
        # # 连续黑屏计数器，用来避免单帧误判
        # self.dark_frame_streak = 0
        # self.zero_blood_streak = 0  # 连续血条为零的计数器，辅助判断是否真的死了
        
        # 连续满足“自己和 boss 都空血”的计数器，用来避免单帧误判
        self.both_zero_blood_streak = 0

        self.lock_on_key = lock_on_key  # 锁敌按键，默认是数字键 0
        self.post_lock_on_delay = post_lock_on_delay

        self.delay = delay

    def reset_episode_state(self):
        # 每次新一轮开始时，把连续黑屏计数清零
        # self.dark_frame_streak = 0
        # self.zero_blood_streak = 0
        self.both_zero_blood_streak = 0

    def _frame_stats(self, obs_screen):
        # 只取前三个通道，忽略 grabscreen 返回的 alpha 通道
        gray = cv2.cvtColor(obs_screen[:, :, :3], cv2.COLOR_BGR2GRAY)

        # 整体平均亮度：越低说明画面越黑
        mean_brightness = float(gray.mean())

        # 画面中“很黑”的像素占比
        # 这里用 gray < 25 作为“几乎黑屏”的经验阈值
        dark_ratio = float((gray < 25).mean())

        return mean_brightness, dark_ratio

    
    # def is_really_dead(self, obs_screen, next_self_blood):
    #     mean_brightness, dark_ratio = self._frame_stats(obs_screen)

    #     dark_screen = (
    #         mean_brightness <= self.death_dark_threshold
    #         and dark_ratio >= self.death_dark_ratio_threshold
    #     )
    #     self_blood_gone = next_self_blood <= self.self_blood_dead_threshold
    #     zero_blood = next_self_blood == 0

    #     # 通道 A：黑屏 + 血条消失
    #     if dark_screen and self_blood_gone:
    #         self.dark_frame_streak += 1
    #     else:
    #         self.dark_frame_streak = 0

    #     # 通道 B：连续多帧零血
    #     if zero_blood:
    #         self.zero_blood_streak += 1
    #     else:
    #         self.zero_blood_streak = 0

    #     dark_death = self.dark_frame_streak >= self.death_confirm_frames
    #     zero_blood_death = self.zero_blood_streak >= self.zero_blood_confirm_frames

    #     return dark_death or zero_blood_death
    def is_really_dead(self, obs_screen, next_self_blood, next_boss_blood):
        both_zero_blood = next_self_blood == 0 and next_boss_blood == 0

        if both_zero_blood:
            self.both_zero_blood_streak += 1
        else:
            self.both_zero_blood_streak = 0

        return self.both_zero_blood_streak >= self.zero_blood_confirm_frames



    def restart_after_death(self, initial=False):
        # 每次进入 reset 流程，先把内部状态清掉
        self.reset_episode_state()

        # 初始进入环境时不需要等死亡复活
        if initial:
            return

        print("死, restart")
        print("等待重生")

        time.sleep(self.delay) 

        # 最多等一段时间，避免因为识别失败一直卡住
        deadline = time.time() + self.respawn_timeout
        bright_streak = 0

        while time.time() < deadline:
            obs_screen = grab_screen(self.obs_window)
            mean_brightness, _ = self._frame_stats(obs_screen)

            # 画面重新亮起来，说明大概率已经从黑屏死亡进入出生后的场景
            if mean_brightness >= self.respawn_bright_threshold:
                bright_streak += 1
                if bright_streak >= self.respawn_confirm_frames:
                    break
            else:
                bright_streak = 0

            time.sleep(0.25)
        print("确认重生")

        # 再额外等一会，确保角色已可操作
        time.sleep(self.post_respawn_delay)

        # 触发传送 mod，直接飞到 boss 点
        pyautogui.press(self.teleport_key)

        # 给传送动画/加载一点时间
        time.sleep(self.post_teleport_delay)

        # 到 boss 点后手动索敌，让 boss 血条出现
        print("锁敌")
        pyautogui.press(self.lock_on_key)
        time.sleep(self.post_lock_on_delay)