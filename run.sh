python setup.py develop
pip install h5py

python pytorch3dunet/train.py --config resources/3DUnet_confocal_boundary/train_config.yml
