import json
import os
import threading
import uuid

from fastapi import FastAPI, HTTPException

from releaseops_arena.tool_env import ReleaseOpsToolEnv

SUPPORTS_CONCURRENT_SESSIONS: bool = True
MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "64"))

app = FastAPI(title="ReleaseOps Arena Env")

_env_sessions: dict[str, ReleaseOpsToolEnv] = {}
_env_lock = threading.Lock()


@app.get("/health")
def health():
    return {
        "ok": True,
        "active_sessions": len(_env_sessions),
        "max_concurrent_envs": MAX_CONCURRENT_ENVS,
    }


@app.post("/reset")
def reset(params: dict):
    with _env_lock:
        if len(_env_sessions) >= MAX_CONCURRENT_ENVS:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Maximum concurrent environments reached: {MAX_CONCURRENT_ENVS}. "
                    "Close an environment before creating a new one."
                ),
            )

        env = ReleaseOpsToolEnv()
        observation = env.reset(**params)
        env_id = str(uuid.uuid4())
        _env_sessions[env_id] = env

    return {
        "env_id": env_id,
        "observation": json.loads(observation),
        "reward": env.reward,
        "done": env.done,
    }


@app.post("/step")
def step(action: dict):
    env_id = action.get("env_id")
    tool = action.get("tool")
    arguments = action.get("arguments", {})

    if not env_id:
        raise HTTPException(status_code=400, detail="Missing required field: env_id")
    if not tool:
        raise HTTPException(status_code=400, detail="Missing required field: tool")
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=400, detail="Field 'arguments' must be an object")

    with _env_lock:
        env = _env_sessions.get(env_id)

    if env is None:
        raise HTTPException(status_code=404, detail=f"Unknown env_id: {env_id}")

    if tool.startswith("_") or not hasattr(env, tool):
        raise HTTPException(status_code=400, detail=f"Unknown tool: {tool}")

    method = getattr(env, tool)
    if not callable(method):
        raise HTTPException(status_code=400, detail=f"Tool is not callable: {tool}")

    try:
        result = method(**arguments)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid arguments for {tool}: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    parsed_result = result
    if isinstance(result, str):
        try:
            parsed_result = json.loads(result)
        except json.JSONDecodeError:
            parsed_result = result

    response = {
        "env_id": env_id,
        "result": parsed_result,
        "observation": json.loads(env.render_observation()),
        "reward": env.reward,
        "done": env.done,
        "terminal_reason": env.state.get("terminal_reason") if env.state else None,
    }

    if env.done:
        with _env_lock:
            _env_sessions.pop(env_id, None)

    return response


@app.post("/close")
def close(payload: dict):
    env_id = payload.get("env_id")
    if not env_id:
        raise HTTPException(status_code=400, detail="Missing required field: env_id")

    with _env_lock:
        removed = _env_sessions.pop(env_id, None)

    if removed is None:
        raise HTTPException(status_code=404, detail=f"Unknown env_id: {env_id}")

    return {"env_id": env_id, "closed": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("releaseops_arena.server:app", host="0.0.0.0", port=8000)
