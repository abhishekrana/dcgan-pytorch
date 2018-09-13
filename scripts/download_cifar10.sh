
#wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

wget http://pjreddie.com/media/files/cifar.tgz
tar -xzvf cifar.tgz

cd cifar/train
for class_name in airplane automobile bird cat deer dog frog horse ship truck
do
	mkdir -p $class_name
	mv *$class_name*.png $class_name
done

cd ../test
for class_name in airplane automobile bird cat deer dog frog horse ship truck
do
	mkdir -p $class_name
	mv *$class_name*.png $class_name
done

cd ../..
mv cifar ../datasets/cifar10
