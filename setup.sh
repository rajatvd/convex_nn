# setup script
echo "creating virtual environment"
python3 -m venv .venv
source .venv/bin/activate
echo "pulling for submodules."
git pull --recurse-submodules
git submodule update --init lab
echo "installing lab."
python -m pip install -e lab
echo "installing cvx_nn."
python -m pip install -e .
echo "making directories."
mkdir -p figures data results
echo "installing dependencies"
python -m pip install -r requirements.txt
