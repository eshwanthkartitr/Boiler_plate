from pydantic import BaseModel
from typing import List, Dict

class Observation(BaseModel):
    time_step: int
    active_gpus: int
    queue_size: int
    current_budget: float
    incoming_jobs: int

class Action(BaseModel):
    gpus_to_provision: int  # Can be positive (spin up) or negative (spin down)

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict

class TaskConfig(BaseModel):
    task_id: str
    difficulty: str
    max_steps: int
    initial_budget: float
    job_arrival_rate: float # Lambda for Poisson distribution
