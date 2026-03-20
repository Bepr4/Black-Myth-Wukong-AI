import os
import time
import cv2

from env_wukong import Wukong
from screen_key_grab.grabscreen import grab_screen


def to_bgr(img):
    # 截图可能带有 Alpha 通道，这里统一转换成 BGR，便于后续 OpenCV 处理。
    if img.shape[-1] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def crop(img, region):
    # 按 (left, top, right, bottom) 裁剪指定区域。
    left, top, right, bottom = region
    return img[top:bottom + 1, left:right + 1]


def draw_region(img, region, color, label):
    # 在整张截图上画出区域框，方便肉眼核对各个检测窗口的位置。
    left, top, right, bottom = region
    cv2.rectangle(img, (left, top), (right, bottom), color, 2)
    cv2.putText(
        img,
        label,
        (left, max(top - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def save_debug_snapshot():
    # 初始化环境对象，复用环境里已经定义好的检测窗口与计数逻辑。
    env = Wukong(observation_w=175, observation_h=200, action_dim=4)
    full_screen = to_bgr(grab_screen())

    # 每次调试输出都写到独立的时间戳目录，避免历史结果被覆盖。
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("debug_output", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # 汇总所有需要检查的屏幕区域。前几个窗口来自环境配置，后几个是额外硬编码的调试区域。
    regions = {
        "obs_window": env.obs_window,
        "boss_blood_window": env.boss_blood_window,
        "self_blood_window": env.self_blood_window,
        "boss_stamina_window": env.boss_stamina_window,
        "self_stamina_window": env.self_stamina_window,
        "self_power_window": (1566, 971, 1599, 1008),
        "self_endurance_window": (186, 987, 311, 995),
        "ding_shen_window": (1458, 851, 1459, 852),
    }

    # 为不同区域指定不同颜色，叠加到总截图里时更容易区分。
    colors = {
        "obs_window": (255, 200, 0),
        "boss_blood_window": (0, 0, 255),
        "self_blood_window": (0, 255, 0),
        "boss_stamina_window": (255, 0, 255),
        "self_stamina_window": (255, 255, 0),
        "self_power_window": (0, 255, 255),
        "self_endurance_window": (255, 128, 0),
        "ding_shen_window": (128, 255, 128),
    }

    # 保存总览图和每个裁剪区域的原始截图，方便逐一核对。
    overlay = full_screen.copy()
    for name, region in regions.items():
        draw_region(overlay, region, colors[name], name)
        cv2.imwrite(os.path.join(output_dir, f"{name}.png"), crop(full_screen, region))

    # 角色血条：先裁剪，再转 HSV，最后调用环境内的计数方法。
    self_blood_img = crop(full_screen, env.self_blood_window)
    self_blood_hsv = cv2.cvtColor(self_blood_img, cv2.COLOR_BGR2HSV)
    self_blood_value = env.self_blood_count(self_blood_img)

    # Boss 血条检测流程与角色血条一致。
    boss_blood_img = crop(full_screen, env.boss_blood_window)
    boss_blood_hsv = cv2.cvtColor(boss_blood_img, cv2.COLOR_BGR2HSV)
    boss_blood_value = env.boss_blood_count(boss_blood_hsv)

    # 角色体力条检测。
    self_stamina_img = crop(full_screen, env.self_stamina_window)
    self_stamina_hsv = cv2.cvtColor(self_stamina_img, cv2.COLOR_BGR2HSV)
    self_stamina_value = env.self_stamina_count(self_stamina_hsv)

    # Boss 韧性/体力条检测。
    boss_stamina_img = crop(full_screen, env.boss_stamina_window)
    boss_stamina_hsv = cv2.cvtColor(boss_stamina_img, cv2.COLOR_BGR2HSV)
    boss_stamina_value = env.boss_stamina_count(boss_stamina_hsv)

    # 棍势/能量条检测。
    self_power_img = crop(full_screen, regions["self_power_window"])
    self_power_hsv = cv2.cvtColor(self_power_img, cv2.COLOR_BGR2HSV)
    self_power_value = env.self_power_count(self_power_hsv)

    # 耐力条这里走灰度逻辑，因此先转灰度图再计数。
    self_endurance_img = crop(full_screen, regions["self_endurance_window"])
    self_endurance_gray = cv2.cvtColor(self_endurance_img, cv2.COLOR_BGR2GRAY)
    self_endurance_value = env.self_endurance_count(self_endurance_gray)

    # 额外把血条白色掩码保存出来，便于判断 HSV 阈值是否合理。
    lower_white = (0, 0, 180)
    upper_white = (179, 30, 220)
    self_blood_mask = cv2.inRange(self_blood_hsv, lower_white, upper_white)
    boss_blood_mask = cv2.inRange(boss_blood_hsv, lower_white, upper_white)
    cv2.imwrite(os.path.join(output_dir, "self_blood_mask.png"), self_blood_mask)
    cv2.imwrite(os.path.join(output_dir, "boss_blood_mask.png"), boss_blood_mask)
    cv2.imwrite(os.path.join(output_dir, "screen_overlay.png"), overlay)

    # 整理一份文本报告，既方便终端查看，也方便后续回溯。
    lines = [
        f"output_dir: {os.path.abspath(output_dir)}",
        f"self_blood_count: {self_blood_value}",
        f"boss_blood_count: {boss_blood_value}",
        f"self_stamina_count: {self_stamina_value}",
        f"boss_stamina_count: {boss_stamina_value}",
        f"self_power_count: {self_power_value}",
        f"self_endurance_count: {self_endurance_value}",
        "self_dead_threshold: < 400",
    ]

    # 根据关键计数结果给出简单诊断建议，帮助快速定位问题。
    advice = []
    if self_blood_value < 400:
        advice.append("self_blood_count already below death threshold; self blood crop or HSV threshold is likely wrong.")
    if boss_blood_value == 0:
        advice.append("boss_blood_count is 0; boss bar crop may be off-screen or hidden in the current frame.")
    if not advice:
        advice.append("Current snapshot looks numerically valid; if training still resets early, the issue is likely intermittent flashes or restart frames.")

    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines + [""] + advice) + "\n")

    # 同步输出到控制台，调试时不用打开文件也能立刻看到结果。
    print("\n".join(lines))
    print("")
    for item in advice:
        print(item)


if __name__ == "__main__":
    save_debug_snapshot()
