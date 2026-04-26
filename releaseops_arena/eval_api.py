import subprocess
import os
import json
from fastapi import APIRouter, BackgroundTasks

router = APIRouter()

@router.post("/run-eval")
def run_eval(background_tasks: BackgroundTasks, limit: int = 3, model_id: str = "hiitsesh/releaseops-grpo-1.7"):
    """
    Triggers an evaluation run. Check HF space logs (stdout) for progress.
    """
    def run_script(limit_val, model_val):
        print(f"=== Starting Evaluation for {model_val} with limit {limit_val} ===", flush=True)
        # We run the existing script as a subprocess so it doesn't block the API and uses the Torch pipeline logic intact
        cmd = [
            "python", "training/evaluate_llm_baseline.py",
            "--backend", "torch",
            "--torch-model", model_val,
            "--limit", str(limit_val),
            "--output-json", "outputs/eval_api_result.json"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("=== Evaluation completed ===", flush=True)
            print("STDOUT:", result.stdout, flush=True)
        except subprocess.CalledProcessError as e:
            print("=== Evaluation failed ===", flush=True)
            print("STDERR:", e.stderr, flush=True)
            
    background_tasks.add_task(run_script, limit, model_id)
    return {"message": f"Evaluation started in background for {model_id} (limit={limit}). Check the HF Space Logs for output."}

@router.get("/get-eval-results")
def get_eval_results():
    if not os.path.exists("outputs/eval_api_result.json"):
        return {"status": "pending_or_missing", "message": "Result file not found yet."}
    with open("outputs/eval_api_result.json", "r") as f:
        return json.load(f)
