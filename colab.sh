nvidia-smi
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html --quiet
pip install neptune-client --quiet
pip install optuna --quiet
pip install neptune-contrib --quiet
pip install livelossplot --quiet
cd openke
rm -rf release
bash make.sh
cd /content/drive/My Drive/code/relation-prediction-4