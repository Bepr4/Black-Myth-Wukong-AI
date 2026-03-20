import argparse

import cv2
import torch

import utils.directkeys as directkeys
from env_wukong import Wukong
from nets.ResNet_boss_model import ResNet50_boss
from nets.dqn_net import Q_construct
from screen_key_grab.grabscreen import grab_screen


INDEX_TO_LABEL = {
    0: "冲刺砍",
    1: "旋转飞",
    2: "扔刀",
    3: "飞雷神",
    4: "锄地",
    5: "锄地起飞",
    6: "受到攻击",
    7: "普攻",
    8: "观察",
    9: "大荒星陨",
}


def build_obs_tensor(obs):
    chw = obs.transpose(2, 0, 1)
    tensor = torch.from_numpy(chw).unsqueeze(0).float()
    return tensor[:, :3, 20:180, 5:165]


def choose_action(q_net, boss_net, obs_tensor):
    with torch.no_grad():
        output_boss, intermediate = boss_net(obs_tensor)
        _, indices_boss = torch.max(output_boss, dim=1)
        q_values = q_net(intermediate).cpu()
        action = int(q_values.argmax(dim=1).item())
    return action, int(indices_boss.item())


def apply_action_constraints(env, action, boss_state):
    self_power_window = (1566, 971, 1599, 1008)
    self_endurance_window = (186, 987, 311, 995)
    ding_shen_window = (1458, 851, 1459, 852)

    self_power_img = grab_screen(self_power_window)
    self_power_hsv = cv2.cvtColor(self_power_img, cv2.COLOR_BGR2HSV)
    self_power = env.self_power_count(self_power_hsv)

    self_endurance_img = grab_screen(self_endurance_window)
    endurance_gray = cv2.cvtColor(self_endurance_img, cv2.COLOR_BGR2GRAY)
    self_endurance = env.self_endurance_count(endurance_gray)

    ding_shen_img = grab_screen(ding_shen_window)
    hsv_img = cv2.cvtColor(ding_shen_img, cv2.COLOR_BGR2HSV)
    ding_shen_available = hsv_img[0, 0][2] >= 130

    if boss_state in (8, 6) and self_power > 100:
        action = 4

    if action not in (1, 3) and self_endurance < 30 and self_endurance != 0:
        action = 5

    if ding_shen_available:
        action = 6

    return action


def main():
    parser = argparse.ArgumentParser(description="Run a trained Wukong model without training")
    parser.add_argument(
        "--model",
        type=str,
        default=r"D:\HCC\Black-Myth-Wukong-AI-main\models\wukong_0904_1_2500.pth",
        help="Path to the trained Q-network checkpoint",
    )
    parser.add_argument(
        "--boss-model",
        type=str,
        default=r"D:\HCC\Black-Myth-Wukong-AI-main\boss_model.pkl",
        help="Path to the boss state classifier",
    )
    parser.add_argument("--episodes", type=int, default=0, help="0 means run forever")
    parser.add_argument("--initial-steal", type=int, default=1, help="1 to keep the opening hard attack")
    args = parser.parse_args()

    env = Wukong(observation_w=175, observation_h=200, action_dim=4)

    boss_net = ResNet50_boss(num_classes=10)
    boss_net.load_state_dict(torch.load(args.boss_model, map_location="cpu"))
    boss_net.eval()

    q_net = Q_construct(input_dim=256, num_actions=env.action_dim).float()
    q_net.load_state_dict(torch.load(args.model, map_location="cpu"))
    q_net.eval()

    env.pause_game(True)
    obs = env.reset(initial=True)
    episode_idx = 0
    episode_reward = 0.0
    initial_steal = args.initial_steal == 1

    print(f"loaded model: {args.model}")
    print("按 T 开始/暂停，Ctrl+C 退出")

    while args.episodes == 0 or episode_idx < args.episodes:
        obs_tensor = build_obs_tensor(obs)
        action, boss_state = choose_action(q_net, boss_net, obs_tensor)
        boss_attack = boss_state not in (6, 8)

        if initial_steal:
            print("开局偷一棍")
            directkeys.hard_attack_long()
            initial_steal = False

        action = apply_action_constraints(env, action, boss_state)
        obs, reward, done, _, _ = env.step(action, boss_attack)
        episode_reward += float(reward)

        print(
            f"[Play] ep={episode_idx:04d} boss={INDEX_TO_LABEL.get(boss_state, boss_state)} "
            f"action={action} reward={float(reward):+.3f} ep_reward={episode_reward:+.3f}"
        )

        env.pause_game(False)

        if done:
            print(f"[Episode End] ep={episode_idx:04d} reward={episode_reward:+.3f}")
            obs = env.reset()
            episode_idx += 1
            episode_reward = 0.0
            initial_steal = args.initial_steal == 1


if __name__ == "__main__":
    main()
