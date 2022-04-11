mkdir ../data
cp get_cifarfs.py ../data/
cd ../data
wget https://www.dropbox.com/s/wuxb1wlahado3nq/cifar-fs-splits.zip?dl=0
mv cifar-fs-splits.zip?dl=0 cifar-fs-splits.zip
unzip cifar-fs-splits.zip
rm cifar-fs-splits.zip

python get_cifarfs.py
mv cifar-fs-splits/val1000* cifar-fs/

