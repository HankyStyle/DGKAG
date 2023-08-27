# download ConceptNet
mkdir -p data/
mkdir -p data/cpnet/
wget -nc -P data/cpnet/ https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
cd data/cpnet/
yes n | gzip -d conceptnet-assertions-5.6.0.csv.gz
# download ConceptNet entity embedding
wget https://csr.s3-us-west-1.amazonaws.com/tzw.ent.npy
cd ../../


# download MCQ dataset (from https://github.com/DRSY/DGen)
mkdir -p data/mcq/
wget -nc -P data/mcq/ https://raw.githubusercontent.com/DRSY/DGen/main/Layer1/dataset/total_new_cleaned_train.json
wget -nc -P data/mcq/ https://raw.githubusercontent.com/DRSY/DGen/main/Layer1/dataset/total_new_cleaned_test.json

# create output folders
mkdir -p data/mcq/grounded/
mkdir -p data/mcq/modeling/



# download Sciq dataset (from https://allenai.org/data/sciq)
mkdir -p data/sciq/
wget -nc -P data/sciq/ https://ai2-public-datasets.s3.amazonaws.com/sciq/SciQ.zip
unzip -n data/sciq/SciQ.zip -d data/sciq/
mv "data/sciq/SciQ dataset-2 3/train.json" data/sciq/
mv "data/sciq/SciQ dataset-2 3/test.json" data/sciq/
mv "data/sciq/SciQ dataset-2 3/valid.json" data/sciq/
rm -r "data/sciq/SciQ dataset-2 3"
rm -r "data/sciq/__MACOSX"

# create output folders
mkdir -p data/sciq/grounded/
mkdir -p data/sciq/modeling/