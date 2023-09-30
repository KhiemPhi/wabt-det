conda create --name wabt-det python=3.8
conda activate wabt-det
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pytorch-lightning==1.5.0
pip install transformers
pip install scikit-learn
pip install optuna