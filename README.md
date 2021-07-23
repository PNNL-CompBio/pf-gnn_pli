#Requirements
python                    3.6.1
rdkit                     2020.03.3.0
biopython                 1.78
openbabel                 2.4.1
numpy                     1.19.2
scipy                     1.5.2
torchvision               0.7.0

Conda enviroment is highly recommended for this implementation
#Data Preparation
Data preperation requires the ligand and protein to be in a mol format readable by rdkit
.mol, .mol2, and .pdb are readily handled by rdkit
.sdf is easily handled with openbabel conversion, made convenient with the pybel wrapper

Both files can then be fed into extractM2.py where the cropping window can be adjusted on line 29
For easy model integration it is best to store the m2 protein window produced by the
extract script along with the original protein ex: pickle.dump((m1,m2), file)

Once cropped complexes are stored, their numpy featurization files can be created.
Files for the different models are labeled in the Data_Prep directory

#Training
Below is an example of the training command. Additional options can be added to the 
argument parser here (learning rate, layer amount and dimension, etc). Defaults are
in place for undeclared parameters including a save directory. 

python -W ignore -u train.py --dropout_rate=0.3 --epoch=500 --ngpu=1 --batch_size=32 --num_workers=0  --train_keys=[your_training_keys.pkl]  --test_keys=[your_test_keys.pkl]

The save directory stores each epoch as a .pt allowing the best model inatance to be loaded
later on
Training and test metrics such as loss and ROC are stored in the same directory for each GPU
used. Ex 3 GPUS: log-rank1.csv, log-rank2.csv, and log-rank3.csv
