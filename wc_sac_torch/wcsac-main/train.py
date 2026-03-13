#!/usr/bin/env python3
import torch
import os
import time
from datetime import datetime

from logger import Logger
from replay_buffer import ReplayBuffer
import utils

import hydra
from omegaconf import OmegaConf


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg

        run_name = self._make_run_name()
        self.run_dir = os.path.join(self.work_dir, "data", run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        OmegaConf.save(config=cfg, f=os.path.join(self.run_dir, "config.yaml"))

        self.logger = Logger(
            self.run_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name,
        )
        print(f"run_dir: {self.run_dir}")

        assert 1 >= cfg.risk_level >= 0, f"risk_level must be between 0 and 1 (inclusive), got: {cfg.risk_level}"
        assert cfg.seed != -1, f"seed must be provided, got default seed: {cfg.seed}"
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = utils.make_safety_env(cfg)

        cfg.agent.agent.obs_dim = int(self.env.observation_space.shape[0])
        cfg.agent.agent.action_dim = int(self.env.action_space.shape[0])
        cfg.agent.agent.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]
        self.agent = hydra.utils.instantiate(cfg.agent.agent, _recursive_=False)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
        )

        self.step = 0
        self.max_episode_steps = self._resolve_max_episode_steps()
        if cfg.restart_path != "dummy":
            self.agent.load(cfg.restart_path)

        self.model_dir = os.path.join(self.run_dir, "model")
        self.model_weights_dir = os.path.join(self.model_dir, "model_weights")
        os.makedirs(self.model_weights_dir, exist_ok=True)

    def _make_run_name(self):
        ts = datetime.now().strftime("%Y-%m-%d")
        agent_name = str(self.cfg.agent.name)
        seed = int(self.cfg.seed)
        base = f"{ts}_{agent_name}_s{seed}"
        data_dir = os.path.join(self.work_dir, "data")
        run_name = base
        idx = 1
        while os.path.exists(os.path.join(data_dir, run_name)):
            run_name = f"{base}_{idx}"
            idx += 1
        return run_name

    def _resolve_max_episode_steps(self):
        env_spec = getattr(self.env, "spec", None)
        if env_spec is not None and getattr(env_spec, "max_episode_steps", None) is not None:
            return int(env_spec.max_episode_steps)
        if hasattr(self.cfg, "horizon"):
            return int(self.cfg.horizon)
        return int(self.cfg.agent.agent.max_episode_len)

    def evaluate(self):
        mean_reward = 0
        mean_cost = 0
        mean_cost_rate = 0
        cost_limit_violations = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs, _ = self.env.reset()
            self.agent.reset()
            done, truncated = False, False
            ep_reward = 0
            ep_cost = 0
            ep_len = 0

            while not done and not truncated:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, truncated, info = self.env.step(action)
                ep_reward += reward
                ep_cost += info.get("cost", 0)
                ep_len += 1

            mean_reward += ep_reward
            mean_cost += ep_cost
            ep_cost_rate = ep_cost / max(ep_len, 1)
            mean_cost_rate += ep_cost_rate
            cost_limit_violations += 1 if (ep_cost_rate > float(self.cfg.agent.agent.cost_limit)) else 0

        mean_reward /= self.cfg.num_eval_episodes
        mean_cost /= self.cfg.num_eval_episodes
        mean_cost_rate /= self.cfg.num_eval_episodes
        self.logger.log("eval/mean_reward", mean_reward, self.step)
        self.logger.log("eval/mean_cost", mean_cost, self.step)
        self.logger.log("eval/mean_cost_rate", mean_cost_rate, self.step)
        self.logger.log("eval/cost_limit_violations", cost_limit_violations, self.step)
        self.logger.dump(self.step)
        self.agent.save(self.model_dir)
        self.agent.save_actor(self.model_weights_dir, self.step)

    def run(self):
        episode, ep_reward, ep_cost, total_cost, done, truncated = 0, 0, 0, 0, True, True
        start_time = time.time()
        update_freq = int(getattr(self.cfg, "update_freq", 1))
        update_after = int(getattr(self.cfg, "update_after", self.cfg.num_seed_steps))
        reward_scale = float(getattr(self.cfg, "reward_scale", 1.0))
        while self.step < self.cfg.num_train_steps:
            if done or truncated:
                if self.step > 0:
                    self.logger.log("train/duration", time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if (self.step > 0 and self.step % self.cfg.eval_frequency == 0):
                    self.logger.log("eval/episode", episode, self.step)
                    self.evaluate()

                self.logger.log("train/episode_reward", ep_reward, self.step)
                self.logger.log("train/episode_cost", ep_cost, self.step)
                if self.step > 0:
                    self.logger.log("train/cost_rate", total_cost / self.step, self.step)

                obs, _ = self.env.reset()
                self.agent.reset()
                done, truncated = False, False
                ep_reward = 0
                ep_cost = 0
                ep_step = 0
                episode += 1

                self.logger.log("train/episode", episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            next_obs, reward, done, truncated,info = self.env.step(action)
            cost = info.get("cost", 0)
            reward = float(reward) * reward_scale
            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if ep_step + 1 == self.max_episode_steps else done
            ep_reward += reward
            ep_cost += cost
            total_cost += cost
            self.replay_buffer.add(obs, action, reward, cost, next_obs, done, done_no_max)

            if (
                self.step >= update_after
                and (self.step % update_freq) == 0
                and len(self.replay_buffer) >= int(self.cfg.agent.agent.batch_size)
            ):
                for _ in range(update_freq):
                    self.agent.update(self.replay_buffer, self.logger, self.step)

            obs = next_obs
            ep_step += 1
            self.step += 1
        self.agent.save(self.model_dir)
        self.logger.log("eval/episode", episode, self.step)
        self.evaluate()


@hydra.main(config_path='config', config_name='train_pricing', version_base=None)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
