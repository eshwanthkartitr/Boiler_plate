# ReleaseOps Arena

Konichiwa

Hey This is Eshwanth and as a techie We know that companies are constanly pushing codes to customers for meeting their quartley goals or to honor the commits they had made. This leads to dependency to Ai agents for automated tasks like checking the ci/cd pipeline and pinging the person who can resolve it based on their exp level or some sort of security research agent which scans for an defective packages dependencies and many more essentially these are context files for a particular task which is domain specific and needs to be heavily customizable to acoommadate the users needs so when people put their trust on these powerful agents they have tons and tons of way they could go haywire like hallucinating or writing hardcoded content to pass test cases so the devloper needs to do extensive check on the code changes made by these special skilled agents

That is where Realeaseops comes in where it's trying to learn the underlying org's policy where instead of checking every agent basically doing a exhaustive search we go try to learn the policy which governs a particular org's systems and workflows instead of having a large instruction book which needs to be maintained constanly and can easily break in production where we need as much as reliability as we want. Using Rl i have methodically proven that a slm which is less than quarter the param as another model which has been finetuned for agentic coding tasks that which is from salesforce model even with clear instruction the system tends to break a due to the sheer amount of interaction between agents 

In this project i have replicated different tools , agents which will give us the workflow

---

### 🔗 Important Links (Hackathon Requirements)

* **GitHub (public):** [github.com/eshwanthkartitr/RL](https://github.com/eshwanthkartitr/RL)
* **Google Colab (Open in Colab for this walkthrough):** [Open `ReleaseOps_final_walkthrough.ipynb` in Colab](https://colab.research.google.com/github/eshwanthkartitr/RL/blob/main/notebooks/ReleaseOps_final_walkthrough.ipynb)  
  That URL only works if `notebooks/ReleaseOps_final_walkthrough.ipynb` is on the `main` branch—if the folder is missing on GitHub, add it locally, commit, and push.
* **Hugging Face Space (live environment):** [hiitsesh/New_gpu_space](https://huggingface.co/spaces/hiitsesh/New_gpu_space)
* **2-Minute demo / pitch video:** `[INSERT YOUR YOUTUBE URL HERE]`
* Local path to the notebook: [notebooks/ReleaseOps_final_walkthrough.ipynb](notebooks/ReleaseOps_final_walkthrough.ipynb) (compulsory checklist in the first cell).

---

A little bit abt env 



## 📈 Training Evidence & Results

To prove the SLM can learn the organizational policy without exhaustive searching, the supervisor model was trained using Group Relative Policy Optimization (GRPO) over 100 steps. 

![GRPO Training Results](./images/Training.png)

* **Adaptation & Policy Learning:** The reward plot (top) shows an initial exploration dip at step 15, followed by a steady climb to a stable reward of ~3.09. The model successfully learned the environment's hard rules. 
* **Tool Reliability:** The tool usage plot (bottom) proves the agent mastered the OpenEnv tool schema. While tool call frequency stabilizes, the `tool_failure_freq` remains practically at zero, showing it does not hallucinate commands.

---

## 💻 Getting Started / Installation

ReleaseOps Arena is built on standard OpenEnv mechanics. You can use the **hosted** Space over HTTPS, or run a **local** copy with Docker. Those are two options—not two steps for the same thing.

**1. Install OpenEnv Core**
```bash
pip install openenv-core
```

**2. Call the live Space (no Docker required)**  
Use the public `https://…hf.space` app URL. That hits the same server HF already runs; you do **not** need `docker run` for this (install deps with `pip install -r requirements.txt` in your project, not a container).

```python
from releaseops_arena.client import ReleaseOpsEnvClient
from releaseops_arena.models import ReleaseOpsAction

# Not the huggingface.co/spaces/… *page* URL — use the *.hf.space host
BASE = "https://hiitsesh-new-gpu-space.hf.space"
client = ReleaseOpsEnvClient(BASE)
obs = client.reset()
print(obs.model_dump() if hasattr(obs, "model_dump") else obs.dict())
# next: client.step(ReleaseOpsAction(...))
```

Open **`/docs`** on the same `BASE` for interactive API details.

**3. Run the same API locally (optional Docker)**  
Only if you want the container on your machine: pull the **Space** image from Hugging Face’s registry and map **7860** (see `Dockerfile` / `PORT`). Then point the client at `http://127.0.0.1:7860`. Copy the exact `docker pull` / image name from your Space: **Settings → Add space to Docker Desktop** (or the registry snippet on the Space repo).

```bash
# Example — use the image reference from the Space (Settings / Docker). Port 7860 matches this repo’s Dockerfile.
docker run -d -p 7860:7860 -e PORT=7860 registry.hf.space/<org-or-user>/<space-name>:latest
```

```python
# After docker run, use local base:
BASE = "http://127.0.0.1:7860"
```

If the registry path 404s, use HF’s UI for the authoritative name (namespace and slug differ from the `*.hf.space` hostname in subtle ways).