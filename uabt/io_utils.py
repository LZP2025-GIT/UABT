from __future__ import annotations
import json
import csv
from pathlib import Path
from typing import Dict, Any
import yaml

def load_params(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    if p.suffix.lower() == ".json":
        return json.loads(p.read_text(encoding="utf-8"))

    if p.suffix.lower() in (".yml", ".yaml"):
        return yaml.safe_load(p.read_text(encoding="utf-8"))

    if p.suffix.lower() == ".csv":
        rows = list(csv.DictReader(p.read_text(encoding="utf-8").splitlines()))
        return csv_to_params(rows)

    raise ValueError(f"Unsupported config format: {p.suffix}")

def csv_to_params(rows):
    # Convert a simple (section, key, value) CSV into nested dict
    out = {"metadata": {}, "prior": {}, "domains": {}, "filters": {}, "verification": {}, "controls": {}, "correlation": {}}
    for r in rows:
        sec, key, val = r["section"], r["key"], r["value"]
        if sec.startswith("S") and sec[1:].isdigit():
            out.setdefault("domains", {}).setdefault(sec, {})
            # cast numerics if possible
            try:
                val_cast = float(val)
            except Exception:
                val_cast = val
            out["domains"][sec][key] = val_cast
        elif sec == "metadata":
            out["metadata"][key] = val if not val.replace(".","",1).isdigit() else (float(val) if "." in val else int(val))
        elif sec == "prior":
            out["prior"][key] = float(val)
        elif sec == "filters":
            out["filters"][key] = float(val)
        elif sec == "verification":
            out["verification"][key] = float(val)
        elif sec == "controls":
            out["controls"][key] = float(val)
        elif sec == "correlation":
            out["correlation"][key] = float(val)
    # Normalize controls.omega list if provided as omega1..omega5
    if "omega" not in out["controls"]:
        omegas = [out["controls"].get(f"omega{i}") for i in range(1, 10) if f"omega{i}" in out["controls"]]
        out["controls"]["omega"] = [float(x) for x in omegas] if omegas else [0.95,0.90,0.85,0.90,0.85]
    return out
