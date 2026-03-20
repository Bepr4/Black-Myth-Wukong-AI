import itertools
import sys
import torch
import os
import cv2
import utils.directkeys as directkeys
import time
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from nets.dqn_net import Q_construct
from utils.schedules import *
from replay_buffer import *
from collections import namedtuple
from nets.ResNet_boss_model import ResNet50_boss
from screen_key_grab.grabscreen import grab_screen
from torch.utils.tensorboard import SummaryWriter 

import csv
import atexit
from datetime import datetime
from collections import namedtuple, deque

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])  # 优化器规范，包含优化器构造函数和参数

# 此处指定在cpu上跑
dtype = torch.FloatTensor
dlongtype = torch.LongTensor
device = "cpu"                  # cpu上跑，因为真正的瓶颈是环境交互和图像处理，而不是模型训练，网络本身也不大
paused = True          # 游戏初始状态为暂停，等待训练函数中调用env.pause_game(False)后开始训练
writer = SummaryWriter()   # 用于TensorBoard记录训练过程中的数据，如损失和奖励等
# # Set the logger
# logger = Logger('./logs')

# 状态对应--刀郎
index_to_label = {
    0: '冲刺砍',
    1: '旋转飞',
    2: '扔刀',
    3: '飞雷神',
    4: '锄地',
    5: '锄地起飞',
    6: '受到攻击',
    7: '普攻',
    8: '观察',
    9: '大荒星陨'
}




def dqn_learning(env,
                 optimizer_spec,
                 exploration=LinearSchedule(1000, 0.1), 
                 stopping_criterion=None,
                 replay_buffer_size=1000,
                 batch_size=32,
                 gamma=0.99,
                 learning_starts=50,
                 learning_freq=4,
                 frame_history_len=4,
                 target_update_freq=10,
                 double_dqn=False,
                 checkpoint=0):

    ################
    #  BUILD MODEL #
    ################
    paused = env.pause_game(True)  


    num_actions = env.action_dim
    # 初始boss模型
    model_resnet_boss = ResNet50_boss(num_classes=10) # 刀郎或其他boss的模型
    model_resnet_boss.load_state_dict(torch.load(
        'D:/HCC/Black-Myth-Wukong-AI-main/boss_model.pkl'))  # 这里是广智的视觉模型（resnet）
    model_resnet_boss.to(device)
    model_resnet_boss.eval() # 设置为推理模式
    
    # 初始自身模型
    # model_resnet_malo = ResNet50_boss(num_classes=2) # 用于判断自身是否倒地
    # model_resnet_malo.load_state_dict(torch.load(
    #     'D:/dqn_wukong/RL-ARPG-Agent-1/malo_model.pkl'))
    # model_resnet_malo.to(device)
    # model_resnet_malo.eval()


    # 控制冻结和更新的参数
    for param in model_resnet_boss.parameters():  # 冻结视觉模型的参数，不参与训练
        param.requires_grad = False
        
    # Q网络初始化
    Q = Q_construct(input_dim=256, num_actions=num_actions).type(dtype)  
    Q_target = Q_construct(input_dim=256, num_actions=num_actions).type(dtype) # 目标网络，结构和Q网络一样，参数初始化为Q网络的参数，在训练过程中定期更新为Q网络的参数

    # load checkpoint
    if checkpoint != 0:
        checkpoint_path = "models/wukong_0825_2_1200.pth"
        Q.load_state_dict(torch.load(checkpoint_path))
        Q_target.load_state_dict(torch.load(checkpoint_path))
        print('load model success')

    # initialize optimizer
    optimizer = optimizer_spec.constructor(
        Q.parameters(), **optimizer_spec.kwargs)

    # create replay buffer  经验回放池
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

        # reward monitor: TensorBoard + CSV + console
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_dir = os.path.join("runs", f"reward_live_{run_id}")
    writer = SummaryWriter(log_dir=tb_log_dir)

    os.makedirs("logs", exist_ok=True)
    csv_path = os.path.join("logs", f"reward_trace_{run_id}.csv")
    csv_fp = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_fp)
    csv_writer.writerow([
        "step",
        "episode",
        "reward_step",
        "reward_step_ma200",
        "episode_reward_running",
        "epsilon",
        "done",
        "action",
    ])

    STEP_REWARD_LOG_INTERVAL = 10
    step_reward_window = deque(maxlen=200)

    atexit.register(csv_fp.close)
    atexit.register(writer.close)

    print(f"[RewardMonitor] TensorBoard dir: {tb_log_dir}")
    print(f"[RewardMonitor] CSV path: {csv_path}")


    ###########
    # RUN ENV #
    ###########

    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset(initial=True)
    LOG_EVERY_N_STEPS = 10
    SAVE_MODEL_EVERY_N_STEPS = 100
    episode_rewards = []
    episode_reward = 0  # 当前episode的奖励总和
    episode_cnt = 0
    loss_fn = nn.MSELoss() # 损失函数
    loss_cnt = 0
    #reward_cnt = 0 # 统计reward次数，到一定次数统计一次reward
    #reward_10 = 0  # 用来画reward曲线
    boss_attack = False # 表征boss是否处于攻击状态
    initial_steal = True # 第一次上来先偷一棍
    for t in itertools.count(start=checkpoint):   # 无限循环，直到满足stopping_criterion条件才会break
        # t += 5500
        # Check stopping criterion 可自定义
        if stopping_criterion is not None and stopping_criterion(env, t): #没步检查一次暂停，现在还没写暂停的逻辑
            break
        # Step the env and store the transition
        # store last frame, return idx used later
        last_stored_frame_idx = replay_buffer.store_frame(last_obs)  # 存储当前帧图像到回放池，返回该帧在回放池中的索引，后续存储动作、奖励等信息时会用到这个索引
        # get observatitrons to input to Q network (need to append prev frames)
        observations = replay_buffer.encode_recent_observation()  # 神秘单帧observation？神人代码
        # print(observations.shape)
        
        if initial_steal:
            print("偷一棍")
            directkeys.hard_attack_long() # 黑神话特色偷一刀
            initial_steal = False
        obs = torch.from_numpy(observations).unsqueeze(  # 把observation转换为tensor，并添加一个batch维度，变成1, C, H, W的形状，符合PyTorch的输入要求
            0).type(dtype)  
        obs = obs[:, :3, 20:180, 5:165]  
        
        # 视觉模型判断boss状态，更新boss_attack
        output_boss, intermediate_results_boss = model_resnet_boss(obs) # 输入当前帧图像，输出boss状态的预测结果和中间层结果（作为Q网络的输入）
        max_values_boss, indices_boss = torch.max(output_boss, dim=1)    # output_boss是10维的log概率分布，表示每个boss状态的概率，取最大值的索引作为预测的boss状态
        print("当前帧判断的boss状态:", index_to_label[indices_boss.item()])
        if indices_boss.item() != 6 and indices_boss.item() != 8: # 如果不是收击和观察状态，就认为boss在攻击
            boss_attack = True
        else:
            boss_attack = False

        # 选择动作    
        # before learning starts, choose actions randomly
        if t < learning_starts:
            action = np.random.randint(num_actions) # 随机出招
        else:
            # epsilon greedy exploration epsilon-greedy 策略是一种在强化学习中使用的决策策略
            sample = random.random()
            threshold = exploration.value(t)
            if sample > threshold: # 走greedy策略，选择Q值最大的动作
                q_value_all_actions = Q(intermediate_results_boss) # 输入视觉模型的中间结果，输出每个动作的Q值
                q_value_all_actions = q_value_all_actions.cpu() 
                action = ((q_value_all_actions).data.max(1)[1])[0] # 选择Q值最大的动作
            else: # 走exploration策略，随机选择动作
                action = torch.IntTensor(
                    [[np.random.randint(num_actions)]])[0][0]
                

        '''---------------自身状态提取--------------'''
        
        self_power_window = (1566,971,1599,1008) # 棍势点
        self_power_img = grab_screen(self_power_window)
        self_power_hsv = cv2.cvtColor(self_power_img, cv2.COLOR_BGR2HSV)
        self_power = env.self_power_count(self_power_hsv)  # >50一段 >100第二段
        
        self_endurance_window = (186,987,311,995) # 耐力条
        self_endurance_img = grab_screen(self_endurance_window)
        endurance_gray = cv2.cvtColor(self_endurance_img,cv2.COLOR_BGR2GRAY)
        self_endurance = env.self_endurance_count(endurance_gray) # 为0是满或空 中间是准确的
        
        ding_shen_window = (1458,851,1459,852) #技能
        ding_shen_img = grab_screen(ding_shen_window)
        hsv_img = cv2.cvtColor(ding_shen_img, cv2.COLOR_BGR2HSV)
        hsv_value = hsv_img[0,0]
        
        ding_shen_available = False
        if hsv_value[2] >= 130:   # 如果技能图标亮了，说明定身技能可用
            ding_shen_available = True
        
        self_window = (548,770,1100,1035) # 倒地位置
        self_img = grab_screen(self_window)
        screen_reshape = cv2.resize(self_img,(175,200))[20:180,5:165,:3]
        screen_reshape = screen_reshape.transpose(2,0,1)
        screen_reshape = screen_reshape.reshape(1,3,160,160)
        tensor_malo = torch.from_numpy(screen_reshape).type(dtype = torch.float32)

        '''-----------------------------------------------'''
        
        '''--------------------手动约束部分-----------------'''
        selected_num = random.choice([1, 3]) # 1,3分别是左翻滚和右翻滚
        # if indices_boss.item() == 4 or indices_boss.item() == 9:  # 锄地
        #     action = torch.tensor([selected_num])
        # elif indices_boss.item() == 5 or indices_boss.item() == 1 or indices_boss.item() == 2:  # 锄地起飞
        #     action = torch.tensor([selected_num])
        # elif indices_boss.item() == 7:  # 普攻
        #     action = torch.tensor([selected_num])
        # elif indices_boss.item() == 6:  # 受到攻击
        #     action = torch.tensor([2])
        # elif indices_boss.item() == 3: # 飞雷神
        #     action = torch.tensor([selected_num])
        # elif indices_boss.item() == 0:  # 冲刺砍或扔刀
        #     if self_power > 100:
        #         action = torch.tensor([4])

        if indices_boss.item() == 8 or indices_boss.item() ==6 :  # 观察
            if self_power > 100:
                action = torch.tensor([4])
        
        if action != 3 and action != 1 and self_endurance < 30 and self_endurance != 0: # 攻击但是没有耐力了
             action = torch.tensor([5])  # 强制改成站桩回气
        # if indices_boss.item() == 1 and self_power > 50: # 可识破
        #     action = torch.tensor([7])
        # for state in state_list:
        #     if state != 6 and state != 8:
        #         action = torch.tensor([selected_num])
        if ding_shen_available == True: # 定身技能可用，优先使用定身技能
            action = torch.tensor([6])
            
        # # 额外判断是否倒地，倒地则必须翻滚
        # res,embed = model_resnet_malo(tensor_malo)
        # max_values_boss, indices_self = torch.max(res, dim=1)
        # if indices_self.item() == 0: # 猴倒地
        #     print("倒地了，翻滚")
        #     action = torch.tensor([selected_num])
        
        '''----------------约束结束----------------------'''
        # state_list.append(indices_boss.item()) # 维护boss状态
        # print(state_list)
        
        obs, reward, done, stop, emergence_break = env.step(
            action, boss_attack)  #  最重要的一行！step执行动作，返回这次动作的结果，包括下一帧图像obs、奖励reward、是否死亡done、是否需要紧急中断emergence_break等信息
        
        if action == 4:  # 把重棍处理成三连棍
            action = 2
        elif action == 5: # 把歇脚回气力处理成轻棍
            action = 0
        elif action == 6: # 定身处理成重棍        把扩展动作压回基础动作空间，避免 Q 学习目标过散
            action = 2
        elif action == 7: # 识破处理成重棍         实际还是学前5个动作？？action_dim=4
            action = 2
        
        reward = float(reward)
        episode_reward += reward

        step_reward_window.append(reward)
        step_reward_ma = float(np.mean(step_reward_window))

        action_to_log = int(action.item()) if isinstance(action, torch.Tensor) else int(action)

        writer.add_scalar("reward/step", reward, t)
        writer.add_scalar("reward/step_ma200", step_reward_ma, t)
        writer.add_scalar("reward/episode_running", episode_reward, t)

        csv_writer.writerow([
            t,
            episode_cnt,
            reward,
            step_reward_ma,
            episode_reward,
            float(exploration.value(t)),
            int(done),
            action_to_log,
        ])

        if t % STEP_REWARD_LOG_INTERVAL == 0:
            csv_fp.flush()
            writer.flush()
            print(
                f"[Reward] step={t:06d} ep={episode_cnt:04d} "
                f"r={reward:+7.3f} ma200={step_reward_ma:+7.3f} "
                f"ep_total={episode_reward:+8.3f} eps={exploration.value(t):.4f}"
            )

        
        # 放回环境交互的结果到回放池中
        replay_buffer.store_effect(last_stored_frame_idx, action, reward, done)  # 把执行动作后的结果存储到回放池中，idx是之前store_frame返回的索引，action、reward、done分别是执行动作后的动作、奖励和是否死亡
        
        if done:
            obs = env.reset()   # 重开
            initial_steal = True  # 新的一条命开始时，再偷一棍

            # episode_rewards.append(episode_reward)
            # writer.add_scalar("reward_episode", episode_reward, episode_cnt)
            # episode_cnt += 1
            # print("current episode reward %d" % episode_reward)
            # episode_reward = 0    # 还是需要写好done的逻辑，因为会记录reward曲线
            episode_rewards.append(episode_reward)
            episode_ma10 = float(np.mean(episode_rewards[-10:]))

            writer.add_scalar("reward/episode", episode_reward, episode_cnt)
            writer.add_scalar("reward/episode_ma10", episode_ma10, episode_cnt)

            print(
                f"[Episode] #{episode_cnt:04d} reward={episode_reward:+.3f} "
                f"ma10={episode_ma10:+.3f}"
            )

            episode_cnt += 1
            episode_reward = 0.0

        last_obs = obs
        env.pause_game(False)  # 检查是否需要暂停
        # Perform experience replay and train the network
        # if the replay buffer contains enough samples..
        last_time = time.time()  
        if (t > learning_starts and 
                t % learning_freq == 0 and  # 每learning_freq步训练一次
                replay_buffer.can_sample(batch_size)): # 回访池里样本必须够抽一个batch
            last_time = time.time()
            # sample transition batch from replay memory
            # done_mask = 1 if next state is end of episode
            obs_t, act_t, rew_t, obs_tp1, done_mask = replay_buffer.sample(
                batch_size)      
            obs_t = torch.tensor(obs_t, dtype=torch.float32)
            obs_t = obs_t[:, :3, 20:180, 5:165]
            obs_t = obs_t.to(device)       # 采样 batch 并预处理
            act_t = torch.tensor(act_t, dtype=torch.long).to(device)
            rew_t = torch.tensor(rew_t, dtype=torch.float32).to(device)
            obs_tp1 = torch.tensor(obs_tp1, dtype=torch.float32)
            obs_tp1 = obs_tp1[:, :3, 20:180, 5:165]
            obs_tp1 = obs_tp1.to(device)
            done_mask = torch.tensor(done_mask, dtype=torch.float32).to(device)
            # input batches to networks
            # get the Q values for current observations (Q(s,a, theta_i))
            output_boss_, intermediate_results_boss = model_resnet_boss(obs_t)
            output_boss_, intermediate_results_boss_tp1 = model_resnet_boss(
                obs_tp1)
            q_values = Q(intermediate_results_boss)
            q_s_a = q_values.gather(1, act_t.unsqueeze(1))
            q_s_a = q_s_a.squeeze()
            if (double_dqn):
                # ------------
                # double DQN
                # ------------
                # get Q values for best actions in obs_tp1
                # based off the current Q network
                # max(Q(s',a',theta_i)) wrt a'
                q_tp1_values = Q(intermediate_results_boss_tp1)
                q_tp1_values = q_tp1_values.detach()
                _, a_prime = q_tp1_values.max(1)
                # get Q values from frozen network for next state and chosen action
                # Q(s', argmax(Q(s',a',theta_i), theta_i_frozen)) (argmax wrt a')
                q_target_tp1_values = Q_target(intermediate_results_boss_tp1)
                q_target_tp1_values = q_target_tp1_values.detach()
                q_target_s_a_prime = q_target_tp1_values.gather(
                    1, a_prime.unsqueeze(1))
                q_target_s_a_prime = q_target_s_a_prime.squeeze()
                # if current state is end of episode, then there is no next Q value
                q_target_s_a_prime = (1 - done_mask) * q_target_s_a_prime
                expected_q = rew_t + gamma * q_target_s_a_prime
            else:
                # -------------
                # regular DQN
                # -------------
                # get Q values for best actions in obs_tp1
                # based off frozen Q network
                # max(Q(s',a',theta_i_frozen)) wrt a'
                q_tp1_values = Q_target(intermediate_results_boss_tp1)
                q_tp1_values = q_tp1_values.detach()
                q_s_a_prime, a_prime = q_tp1_values.max(1)
                # if current state is end of episode, then there is no next Q value
                q_s_a_prime = (1 - done_mask) * q_s_a_prime
                # Compute Bellman error
                # r + gamma * Q(s', a', theta_i_frozen) - Q(s, a, theta_i)
                expected_q = rew_t + gamma * q_s_a_prime
            time_before_optimization = time.time()
            # 计算loss
            loss = loss_fn(expected_q, q_s_a)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar("loss_dqn", loss.item(), loss_cnt)
            loss_cnt += 1
            num_param_updates += 1
            print('optimization took {} seconds'.format(
                time.time()-time_before_optimization))
            # update target Q network weights with current Q network weights
            if num_param_updates % target_update_freq == 0:
                Q_target.load_state_dict(Q.state_dict())
            print('loop took {} seconds'.format(time.time()-last_time))
            env.pause_game(False)
        # 4. Log progress
        if t % SAVE_MODEL_EVERY_N_STEPS == 0:
            if not os.path.exists("models"):
                os.makedirs("models")
            if not os.path.exists("models_res"):
                os.makedirs("models_res")
            model_save_path = "models/wukong_0904_1_%d.pth" % (t)
            torch.save(Q.state_dict(), model_save_path)
        # episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-10:])
            best_mean_episode_reward = max(
                best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0:
            print("---------------------------------")
            print("Timestep %d" % (t,))
            print("learning started? %d" % (t > learning_starts))
            print("mean reward (10 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.kwargs['lr'])
            sys.stdout.flush()