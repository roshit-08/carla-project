import json
from pathlib import Path

NB_PATH = Path("notebooks/xgboost_harsh_driving_v2.ipynb")
with open(NB_PATH) as f:
    nb = json.load(f)

# Keep cells 1 to 12 and the final markdown cell (cell 19)
cells = nb["cells"][:12] + [nb["cells"][-1]]

nb["cells"] = cells

for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"✓ Trimmed notebook to {len(nb['cells'])} clean cells.")
