import os, uuid, json
from .schemas import RunState
def new_state(artifacts_dir:str, dataset_ref:str, target:str|None, time_col:str|None)->RunState:
  os.makedirs(artifacts_dir, exist_ok=True)
  return RunState(run_id=uuid.uuid4().hex[:12], artifacts_dir=artifacts_dir, dataset_ref=dataset_ref, target=target, time_col=time_col)
def save_state(state:RunState)->str:
  p=os.path.join(state.artifacts_dir,f"runstate_{state.run_id}.json")
  with open(p,"w") as f: f.write(state.model_dump_json(indent=2)); return p
