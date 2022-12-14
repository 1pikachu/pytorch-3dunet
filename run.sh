python setup.py develop
pip install h5py

ln -s /home2/pytorch-broad-models/gpuoob/Unet/3dunet_datasets 3dunet_datasets
ln -s /home2/pytorch-broad-models/gpuoob/Unet/2dunet_datasets 2dunet_datasets

python pytorch3dunet/train.py --config resources/3DUnet_confocal_boundary/train_config.yml
