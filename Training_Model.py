"""-----------------------------------------------"""

"""---Launching the training from ./src folder----"""

"""-----------------------------------------------"""

import os, sys
import configparser


#Reading Configuration file (STRUCTURE.txt)
config = configparser.RawConfigParser()
config.read_file(open(r'./STRUCTURE.txt'))


#Experiment Name
exp_name = config.get('experiment name', 'name')

#output on logfile
nohup = config.getboolean('training settings', 'nohup')

execute_sys = '' if sys.platform == 'darwin' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

#creating results
loc_result = exp_name
print("\nA. Creating Results' directory")
if os.path.exists(loc_result):
    print("Directory exists already")
elif sys.platform=='darwin':
    os.system('mkdir ' + loc_result)
else:
    os.system('mkdir -p ' +loc_result)

print("\nN. copying the Structure file in the results folder")
if sys.platform=='darwin':
    os.system('copy STRUCTURE.txt .\\' +exp_name+'\\'+exp_name+'_STRUCTURE.txt')
else:
    os.system('cp STRUCTURE.txt ./' +exp_name+'/'+exp_name+'_STRUCTURE.txt')

# Experiment Execution
if nohup:
    print("\nB. Running training on the System")
    os.system(execute_sys +' nohup python -u ./src/retinaNN_training.py > ' +'./'+exp_name+'/'+exp_name+'_training.nohup')
else:
    print("\nB. Running training on System (no nohup)")
    os.system(execute_sys +' python ./src/retinaNN_training.py')
