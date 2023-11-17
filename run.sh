# !/bin/bash

if [ -d /app/model/GroundingDINO/ ]; then
    echo "Files for GroundingDINO exists. Installing GroundingDINO"
    cd /app/model/GroundingDINO/
    pip install -e .
    cd /app
else
    echo "Files for GroundingDINO does not exists. Downloading GroundingDINO"
    cd /app/model
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    cd GroundingDINO
    echo "Installing GroundingDINO"
    pip install -e .
    cd /app
fi

if ! [ -d /app/model/GroundingDINO/weights ]; then
    echo "Downloading weights for GroundingDINO"
    cd /app/model/GroundingDINO/
    mkdir weights
    cd weights
    wget wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    cd /app
fi

flask run --host=0.0.0.0
