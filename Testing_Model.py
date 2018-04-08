"""-----------------------------------------------"""

"""---Launching the prediction from ./src folder----"""

"""-----------------------------------------------"""

import os, sys
import configparser


#Reading Configuration file (STRUCTURE.txt)
config = configparser.RawConfigParser()
config.read_file(open(r'./STRUCTURE.txt'))

#Experiment Name
exp_name = config.get('experiment name', 'name')

#output on logfile
nohup = config.getboolean('testing settings', 'nohup')

execute_sys = '' if sys.platform == 'darwin' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

#creating results
result_dir = exp_name
print("\nA. Creating Results' directory")
if os.path.exists(result_dir):
    pass
elif sys.platform=='darwin':
    os.system('md ' + result_dir)
else:
    os.system('mkdir -p ' + result_dir)


#running prediction
if nohup:
    print("\nB. Running prediction on the System")
    os.system(execute_sys +' nohup python -u ./src/retinaNN_predict.py > ' +'./'+exp_name+'/'+exp_name+'_prediction.nohup')
else:
    print("\nB. Running prediction on the System (no nohup)......")
    print("\nPlease wait..")
    print("\nPredicting.........")
    os.system(execute_sys +' python ./src/retinaNN_predict.py')
