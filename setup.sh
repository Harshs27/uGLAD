# Update the conda package. (optional)
conda update -n base conda
# Create conda environment.
conda create -n uglad python -y;
conda activate uglad;
conda install -c conda-forge notebook -y;
python -m ipykernel install --user --name uglad;

# Install pytorch
conda install numpy -y;
conda install pytorch torchvision -c pytorch -y;

# Install packages from conda-forge.
conda install -c conda-forge matplotlib -y;

# Install packages from anaconda.
conda install -c anaconda pandas networkx scipy -y;

# Install pip packages
pip install -U scikit-learn

# Pyvis installation
pip install pyvis
