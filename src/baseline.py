import requests

BASE_URL = "http://localhost:7860" # Default HF space port

def evaluate_baseline(task_id):
    requests.post(f"{BASE_URL}/reset?task_id={task_id}")
    done = False
    
    while not done:
        state = requests.get(f"{BASE_URL}/state").json()["observation"]
        
        # Simple policy: If queue is larger than active GPUs, provision more.
        gpus_needed = state["queue_size"] - state["active_gpus"]
        action = {"gpus_to_provision": max(-1, min(2, gpus_needed))} # Throttle scaling
        
        step_res = requests.post(f"{BASE_URL}/step", json=action).json()
        done = step_res["done"]
        
    score = requests.get(f"{BASE_URL}/grader").json()["score"]
    print(f"Task: {task_id} | Final Score: {score:.3f}")

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        evaluate_baseline(task)
