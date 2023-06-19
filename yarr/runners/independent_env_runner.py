import numpy as np
import torch
from typing import List
from typing import Union

from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.replay_buffer.replay_buffer import ReplayBuffer
from yarr.runners._independent_env_runner import _IndependentEnvRunner
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.stat_accumulator import StatAccumulator, SimpleAccumulator
from yarr.agents.agent import Summary
from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv

from yarr.runners.env_runner import EnvRunner
import logging

class IndependentEnvRunner(EnvRunner):

    def __init__(self,
                 train_env: Env,
                 agent: Agent,
                 train_replay_buffer: Union[ReplayBuffer, List[ReplayBuffer]],
                 num_train_envs: int,
                 num_eval_envs: int,
                 rollout_episodes: int,
                 eval_episodes: int,
                 training_iterations: int,
                 eval_from_eps_number: int,
                 episode_length: int,
                 eval_env: Union[Env, None] = None,
                 eval_replay_buffer: Union[ReplayBuffer, List[ReplayBuffer], None] = None,
                 stat_accumulator: Union[StatAccumulator, None] = None,
                 rollout_generator: RolloutGenerator = None,
                 weightsdir: str = None,
                 logdir: str = None,
                 max_fails: int = 10,
                 num_eval_runs: int = 1,
                 env_device: torch.device = None,
                 multi_task: bool = False,
                 classifier = None,
                 l2a = None):
            self._classifier = classifier
            self._l2a = l2a
            super().__init__(train_env, agent, train_replay_buffer, num_train_envs, num_eval_envs,
                            rollout_episodes, eval_episodes, training_iterations, eval_from_eps_number,
                            episode_length, eval_env, eval_replay_buffer, stat_accumulator,
                            rollout_generator, weightsdir, logdir, max_fails, num_eval_runs,
                            env_device, multi_task)

    def summaries(self) -> List[Summary]:
        summaries = []
        if self._stat_accumulator is not None:
            summaries.extend(self._stat_accumulator.pop())
        self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
        summaries.extend(self._agent_summaries)

        # add current task_name to eval summaries .... argh this should be inside a helper function
        if hasattr(self._eval_env, '_task_class'):
            eval_task_name = change_case(self._eval_env._task_class.__name__)
        elif hasattr(self._eval_env, '_task_classes'):
            if self._current_task_id != -1:
                task_id = (self._current_task_id) % len(self._eval_env._task_classes)
                eval_task_name = change_case(self._eval_env._task_classes[task_id].__name__)
            else:
                eval_task_name = ''
        else:
            raise Exception('Neither task_class nor task_classes found in eval env')

        # multi-task summaries
        if eval_task_name and self._multi_task:
            for s in summaries:
                if 'eval' in s.name:
                    s.name = '%s/%s' % (s.name, eval_task_name)

        return summaries

    # serialized evaluator for individual tasks
    def start(self, weight,
              save_load_lock, writer_lock,
              env_config,
              device_idx,
              save_metrics,
              cinematic_recorder_cfg, interactive=False):
        multi_task = isinstance(env_config[0], list)
        if multi_task:
            eval_env = CustomMultiTaskRLBenchEnv(
                task_classes=env_config[0],
                observation_config=env_config[1],
                action_mode=env_config[2],
                dataset_root=env_config[3],
                episode_length=env_config[4],
                headless=env_config[5],
                swap_task_every=env_config[6],
                include_lang_goal_in_obs=env_config[7],
                time_in_state=env_config[8],
                record_every_n=env_config[9])
        else:
            eval_env = CustomRLBenchEnv(
                task_class=env_config[0],
                observation_config=env_config[1],
                action_mode=env_config[2],
                dataset_root=env_config[3],
                episode_length=env_config[4],
                headless=env_config[5],
                include_lang_goal_in_obs=env_config[6],
                time_in_state=env_config[7],
                record_every_n=env_config[8])

        self._internal_env_runner = _IndependentEnvRunner(
            self._train_env, eval_env, self._agent, self._timesteps, self._train_envs,
            self._eval_envs, self._rollout_episodes, self._eval_episodes,
            self._training_iterations, self._eval_from_eps_number, self._episode_length, self._kill_signal,
            self._step_signal, self._num_eval_episodes_signal,
            self._eval_epochs_signal, self._eval_report_signal,
            self.log_freq, self._rollout_generator, None,
            self.current_replay_ratio, self.target_replay_ratio,
            self._weightsdir, self._logdir,
            self._env_device, self._previous_loaded_weight_folder,
            num_eval_runs=self._num_eval_runs)

        stat_accumulator = SimpleAccumulator(eval_video_fps=30)
        if not interactive:
            self._internal_env_runner._run_eval_independent('eval_env',
                                                            stat_accumulator,
                                                            weight,
                                                            writer_lock,
                                                            True,
                                                            device_idx,
                                                            save_metrics,
                                                            cinematic_recorder_cfg)
        else:
            self._run_eval_interactive('eval_env',
                                        weight,
                                        True,
                                        device_idx)
            
    def _run_eval_interactive(self, name: str,
                              weight,
                              eval=True,
                              device_idx=0):

        self._name = name
        self._is_test_set = type(weight) == dict

        device = torch.device('cuda:%d' % device_idx) if torch.cuda.device_count() > 1 else torch.device('cuda:0')
        self._agent.build(training=False, device=device)

        logging.info('%s: Launching env.' % name)
        np.random.seed()

        logging.info('Agent information:')
        logging.info(self._agent)

        env = self._eval_env
        env.eval = eval
        env.launch()

        if not os.path.exists(self._weightsdir):
            raise Exception('No weights directory found.')

        # one weight for all tasks (used for validation)
        if type(weight) == int:
            logging.info('Evaluating weight %s' % weight)
            weight_path = os.path.join(self._weightsdir, str(weight))
            seed_path = self._weightsdir.replace('/weights', '')
            self._agent.load_weights(weight_path)
            weight_name = str(weight)

        new_transitions = {'train_envs': 0, 'eval_envs': 0}
        total_transitions = {'train_envs': 0, 'eval_envs': 0}
        current_task_id = -1

        # reset the task
        variation = 0
        eval_demo_seed = 1000 # TODO
        obs = env.reset_to_seed(variation, eval_demo_seed)
        # replace the language goal with user input
        command = ''
        while command != 'quit':
            command = input("Enter a command: ")
            if command == 'reset':
                eval_demo_seed += 1
                obs = env.reset_to_seed(variation, eval_demo_seed)
                continue
            # tokenize the command
            env._lang_goal = command
            tokens = tokenize([command])[0].numpy()
            # send the tokens to the classifier
            command_class = self._classifier.predict(tokens)
            # if command class is 1, use voxel transformer
            if command_class == 1:
                obs['lang_goal_tokens'] = tokens
                self._agent.reset()
                timesteps = 1
                obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
                prepped_data = {k:torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}

                act_result = self._agent.act(self._step_signal.value, prepped_data,
                                        deterministic=eval)
                transition = env.step(act_result)
            else:
                
            # double step updates rendered views?
            transition = env.step(act_result)
            obs = dict(transition.observation)