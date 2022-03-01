# setup script
echo "creating virtual environment"
python3 -m venv .venv
source .venv/bin/activate
echo "installing convex_nn."
python -m pip install -e .
