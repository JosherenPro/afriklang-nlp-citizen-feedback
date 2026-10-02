# Point d'entrée Streamlit Community Cloud. Ce dossier porte son propre requirements.txt
# (torch CPU, sans l'outillage du notebook) : celui de la racine tire torch CUDA (~2,5 Go).
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
