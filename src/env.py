import numpy as np
from src.models import Observation, Action, StepResult, TaskConfig

class GPUClusterEnv:
    def __init__(self):
        self.config = None
        self.state = None
        self.total_reward = 0.0

    def reset(self, config: TaskConfig) -> Observation:
        self.config = config
        self.total_reward = 0.0
        self.state = Observation(
            time_step=0,
            active_gpus=1,
            queue_size=0,
            current_budget=config.initial_budget,
            incoming_jobs=0
        )
        return self.state

    def step(self, action: Action) -> StepResult:
        if self.state is None:
            raise ValueError("Environment must be reset before calling step.")

        # 1. Apply Action (Scale infrastructure)
        self.state.active_gpus = max(0, self.state.active_gpus + action.gpus_to_provision)
        
        # 2. Simulate incoming workloads
        new_jobs = np.random.poisson(self.config.job_arrival_rate)
        self.state.incoming_jobs = new_jobs
        self.state.queue_size += new_jobs

        # 3. Process jobs (1 GPU processes 1 job per step)
        jobs_processed = min(self.state.active_gpus, self.state.queue_size)
        self.state.queue_size -= jobs_processed

        # 4. Calculate Costs & Rewards
        gpu_cost = self.state.active_gpus * 2.5  # $2.50 per step per GPU
        sla_penalty = self.state.queue_size * 1.0 # $1 penalty per waiting job
        
        self.state.current_budget -= gpu_cost
        
        # Reward shaping
        reward = (jobs_processed * 5.0) - gpu_cost - sla_penalty
        self.total_reward += reward
        self.state.time_step += 1

        # 5. Terminal Conditions
        done = self.state.time_step >= self.config.max_steps or self.state.current_budget <= 0

        return StepResult(
            observation=self.state,
            reward=reward,
            done=done,
            info={"jobs_processed": jobs_processed, "total_reward": self.total_reward}
        )
