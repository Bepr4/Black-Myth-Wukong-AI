import numpy as np
import random

def sample_n_unique(sampling_f, n):
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class ReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        super().__init__()

        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx = 0
        self.num_in_buffer = 0

        self.obs = None
        self.action = None
        self.reward = None
        self.done = None

    def can_sample(self, batch_size):
        return batch_size + 1 <= self.num_in_buffer

    
    def _encode_sample(self, idxes):
        obs_batch = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)
    
    def encode_recent_observation(self): # 这个函数是用来获取当前帧图像以及之前的历史帧图像，作为输入给Q网络的observation

        assert self.num_in_buffer > 0  # 确保回放池中至少有一条数据
        return self._encode_observation((self.next_idx - 1) % self.size) # 

    def _encode_observation(self, idx): # 这个函数是用来获取当前帧图像以及之前的历史帧图像，作为输入给Q网络的observation
        end_idx = idx + 1 
        start_idx = end_idx - self.frame_history_len

        if len(self.obs.shape)==2:
            return self.obs[end_idx-1]

        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx-1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)

        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 0) 
        else:
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w)

    def store_frame(self, frame):  # 存储当前帧图像到回放池，返回该帧在回放池中的索引，后续存储动作、奖励等信息时会用到这个索引

        # if observation is an image...
        if len(frame.shape) > 1:
            # transpose image frame into c, h, w instead of h, w, c
            #print(frame.shape)
            #print(frame.type)
            frame = frame.transpose(2, 0, 1) # 将图像从HWC格式转换为CHW格式，符合PyTorch的输入要求，C是通道数，H是高度，W是宽度
            #print(frame.shape)

        if self.obs is None: # 如果没初始化先初始化
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.uint8) # 初始化obs数组，大小为回放池大小乘以每帧图像的形状，数据类型为uint8
            self.action   = np.empty([self.size],                     dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.bool_)
        self.obs[self.next_idx] = frame 

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done): # 把执行动作后的结果存储到回放池中，idx是之前store_frame返回的索引，action、reward、done分别是执行动作后的动作、奖励和是否死亡
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done