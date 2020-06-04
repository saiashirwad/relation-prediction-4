nvidia-smi
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html --quiet
pip install neptune-client --quiet
pip install optuna --quiet
pip install neptunecontrib
cd openke
rm -rf release
bash make.sh
cd /content/relation-prediction-4
python run.py
