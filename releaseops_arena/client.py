from typing import Any, Dict, Optional, Union

import requests

from releaseops_arena.models import ReleaseOpsAction, ReleaseOpsObservation


class ReleaseOpsEnvClient:
    """Small HTTP client for the hosted ReleaseOps Arena environment."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self.env_id: Optional[str] = None

    def reset(self, **kwargs) -> ReleaseOpsObservation:
        response = requests.post(f"{self.base_url}/reset", json=kwargs, timeout=30)
        response.raise_for_status()
        payload = response.json()
        self.env_id = payload["env_id"]
        return ReleaseOpsObservation(**payload)

    def step(self, action: Union[ReleaseOpsAction, Dict[str, Any]]) -> ReleaseOpsObservation:
        if self.env_id is None:
            raise RuntimeError("Call reset() before step().")

        if isinstance(action, ReleaseOpsAction):
            payload = action.model_dump() if hasattr(action, "model_dump") else action.dict()
        else:
            payload = dict(action)

        payload["env_id"] = self.env_id
        response = requests.post(f"{self.base_url}/step", json=payload, timeout=30)
        response.raise_for_status()
        return ReleaseOpsObservation(**response.json())

    def close(self) -> Dict[str, Any]:
        if self.env_id is None:
            return {"closed": False, "reason": "no active session"}

        response = requests.post(
            f"{self.base_url}/close",
            json={"env_id": self.env_id},
            timeout=30,
        )
        response.raise_for_status()
        self.env_id = None
        return response.json()
