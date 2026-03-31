import json
from pathlib import Path

from src.llm_client import LLMClient
from src.experiment_runner import ExperimentRunner

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "extension_cognitive_modes.json"
OUT_PATH = ROOT / "data" / "full_run_cognitive_extension_results.jsonl"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

tasks_path = Path(cfg["tasks_path"])
if not tasks_path.is_absolute():
    tasks_path = ROOT / tasks_path

print(f"Running cognitive extension from config: {CONFIG_PATH}")
print(f"Output path: {OUT_PATH}")

llm = LLMClient(model="gpt-4o-mini")
runner = ExperimentRunner(llm=llm, seed=cfg.get("seeds", [42])[0])

runner.run_multiple(
    dataset_paths=[str(tasks_path)],
    buffer_sizes=cfg["buffer_sizes"],
    policies=cfg["policies"],
    summarisation_levels=cfg["summarisation_levels"],
    summarisation_window_sizes=cfg["summarisation_window_sizes"],
    out_path=str(OUT_PATH),
    max_steps=cfg.get("max_steps", 12),
    min_steps_before_done=3,
    include_verbose=False,
)

print(f"Done. Results at: {OUT_PATH}")