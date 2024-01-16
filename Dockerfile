FROM nvcr.io/nvidia/pytorch:23.12-py3

RUN git clone https://github.com/boschresearch/DroidCalib.git
WORKDIR DroidCalib
RUN git submodule update --init --recursive
RUN pip install --no-cache-dir \
  torch-scatter \
  tensorboard \
  scipy \
  opencv-python==4.8.0.74 \
  tqdm \
  matplotlib \
  pyyaml \
  evo
RUN python3 setup.py install
  
