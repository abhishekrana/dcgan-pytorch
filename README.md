## Requirements
pytorch >= 0.4.1

### Setup Environment
```sh
# Dependencies
pip install matplotlib
pip install git+https://github.com/wookayin/tensorflow-plot.git@master
```

## Dataset
Download small NORB dataset
```sh
cd scripts
./download_norb_small.sh
```

## Train
```sh
./run.sh
```

## Results
![alt text](https://raw.githubusercontent.com/abhishekrana/dcgan-pytorch/master/results/fake_aeroplane.png "Fake aeroplane")
![alt text](https://raw.githubusercontent.com/abhishekrana/dcgan-pytorch/master/results/fake_human.png "Fake human")

## Acknowledgment
https://github.com/ndrplz/small_norb

