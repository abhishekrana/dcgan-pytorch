git clone https://github.com/ndrplz/small_norb
cd smallnorb

mkdir -p smallnorb
cd smallnorb
wget https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz
wget https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz
wget https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz
wget https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz
wget https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz
wget https://cs.nyu.edu/%7Eylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz
# k: keep original .gz
gunzip -k *.gz
cd ..

python main.py

cd smallnorb_export

cd train
mkdir airplane; ls -1 | grep airplane | xargs mv -t airplane
mkdir car; ls -1 | grep car | xargs mv -t car
mkdir human; ls -1 | grep human | xargs mv -t human
mkdir truck; ls -1 | grep truck | xargs mv -t truck
mkdir animal; ls -1 | grep animal | xargs mv -t animal
cd ..

cd test
mkdir airplane; ls -1 | grep airplane | xargs mv -t airplane
mkdir car; ls -1 | grep car | xargs mv -t car
mkdir human; ls -1 | grep human | xargs mv -t human
mkdir truck; ls -1 | grep truck | xargs mv -t truck
mkdir animal; ls -1 | grep animal | xargs mv -t animal
cd ..
