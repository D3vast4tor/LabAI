from distutils.command import build
from keras import backend as K
import json, keras.layers
import tensorflow as tf
from models import *
def main():  
    fitness = 0.
    sufficent = False
    generation = 0
    models = []
    best = None
    while not sufficent:
        nmodels = []
        os.mkdir("./generation_"+str(generation))
        
        if generation == 0:
            for i in range(0,11):
                models.append(Model())
        else:
            for _ in models:
                nmodels.append(Model(b.dna,Model().dna))
            
            
        for _ in models:
            lfitness = _.evaluate()
            _.save_model(lfitness)
        b = getBestIndividual("generation_" + str(generation))
        if b["fitness"] > 0.5:
            print(b["name"])
            sufficent = True
        else:
            for _ in models:
                _.mutate()
                
        
        
        
        
    
    
if __name__ == "__main__":
    main()