---
title: GPUClusterEnv
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# GPU Cluster Resource Management Environment (GPUClusterEnv)

A real-world cloud infrastructure environment where agents manage GPU provisioning to handle ML training workloads under strict budget constraints. 

Managing compute resources for incoming ML training jobs requires balancing strict budgets against Service Level Agreement (SLA) penalties for long queue times. This environment challenges agents to dynamically scale GPU resources to match fluctuating job arrival rates while maximizing overall reward.

## 🚀 Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GPUClusterEnv
   cd GPUClusterEnv
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI server:
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 7860
   ```

## 🧠 Environment Design

### Observation Space

The observation space is represented as a structured dictionary containing the current state of the GPU cluster:

| Feature | Description | Type |
| :--- | :--- | :--- |
| `time_step` | Current step in the episode. | `int` |
| `active_gpus` | Number of currently provisioned GPUs. | `int` |
| `queue_size` | Number of jobs waiting to be processed. | `int` |
| `current_budget` | Remaining budget for the episode. | `float` |
| `incoming_jobs` | Number of new jobs that arrived in the last step. | `int` |

### Action Space

The agent controls the scaling of the infrastructure by specifying how many GPUs to provision or de-provision:

| Feature | Description | Type | Notes |
| :--- | :--- | :--- | :--- |
| `gpus_to_provision` | Number of GPUs to spin up (positive) or spin down (negative). | `int` | Infrastructure scaling |

### Reward Function

Instead of a sparse reward, the environment uses a shaped reward function that continuously evaluates the agent's performance based on processing jobs while minimizing costs and SLA penalties:

$$Reward = (JobsProcessed \times 5.0) - (ActiveGPUs \times CostPerGPU) - (QueueSize \times Penalty)$$

*   **CostPerGPU**: $2.50 per step per active GPU.
*   **Penalty**: $1.00 SLA penalty per step for each waiting job in the queue.

### Terminal Conditions

An episode ends when:
1. The maximum number of `time_steps` for the task is reached.
2. The `current_budget` drops to $0 or below.

## 📋 Tasks

The environment provides 3 graded tasks with escalating difficulty:

1. **Easy** (`task_id: "easy"`): Low job arrival rate, generous budget. (Max Steps: 50)
2. **Medium** (`task_id: "medium"`): Moderate job arrival rate, standard budget. (Max Steps: 100)
3. **Hard** (`task_id: "hard"`): High, erratic job arrival rate, tight budget. (Max Steps: 200)

## 🤖 Baseline Agent

To evaluate the baseline agent performance:
```bash
python src/baseline.py
```
