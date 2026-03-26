from src.models import TaskConfig

TASKS = {
    "easy": TaskConfig(task_id="easy", difficulty="easy", max_steps=50, initial_budget=1000.0, job_arrival_rate=2.0),
    "medium": TaskConfig(task_id="medium", difficulty="medium", max_steps=100, initial_budget=800.0, job_arrival_rate=5.0),
    "hard": TaskConfig(task_id="hard", difficulty="hard", max_steps=200, initial_budget=500.0, job_arrival_rate=12.0),
}
