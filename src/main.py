from fastapi import FastAPI, HTTPException
from src.models import Action, TaskConfig
from src.env import GPUClusterEnv
from src.tasks import TASKS
import subprocess

app = FastAPI(title="GPU Cluster OpenEnv")
env = GPUClusterEnv()

@app.get("/")
def health_check():
    return {"status": "ok", "message": "GPUClusterEnv is running"}

@app.post("/reset")
def reset_env(task_id: str = "easy"):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    obs = env.reset(TASKS[task_id])
    return {"observation": obs.dict()}

@app.post("/step")
def step_env(action: Action):
    try:
        result = env.step(action)
        return result.dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def get_state():
    if env.state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized")
    return {"observation": env.state.dict()}

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": list(TASKS.keys()),
        "action_schema": Action.schema()
    }

@app.get("/grader")
def grader():
    # Normalizes total reward to a 0.0 - 1.0 score based on max possible baseline
    if env.state is None:
        return {"score": 0.0}
    max_expected_reward = env.config.max_steps * 10 # Arbitrary max for example
    score = max(0.0, min(1.0, env.total_reward / max_expected_reward))
    return {"score": score}

@app.post("/baseline")
def run_baseline():
    # Trigger the baseline script and return results
    result = subprocess.run(["python", "src/baseline.py"], capture_output=True, text=True)
    return {"output": result.stdout}
