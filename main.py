from distutils.command import build
from keras import backend as K
import json, keras.layers
import tensorflow as tf
from models import *
def main():
    tf.device("GPU:1")     
    fitness = 0.
    sufficent = False
    print("\n\n\t\t\tEVALUATING PARENT FITNESS\n\n")
    '''
    while True:
        i = Model()
        j = Model()
        ifitness = i.evaluate()
        K.clear_session()
        jfitness = j.evaluate()
        K.clear_session()
        if fitness < ifitness:
            fitness = ifitness
        if fitness < jfitness:
            fitness = jfitness
        i.save_model(ifitness)
        j.save_model(jfitness)    
        
        K.clear_session()
        
        k = Model(i.dna,j.dna)
        l = Model(j.dna,i.dna)
        kfitness = k.evaluate()
        K.clear_session()
        lfitness = l.evaluate()
        K.clear_session()
        if fitness < kfitness:
            fitness = kfitness
        if fitness < lfitness:
            fitness = lfitness
        
        k.save_model(kfitness)
        l.save_model(lfitness)
        
        K.clear_session()
    
    '''
    while True:
        model = keras.models.load_model("./weights/Robert")
        Robert = Model()
        Robert.model = model
        print(Robert.evaluate())
        Robert.model.save("./weights/Robert")
    
if __name__ == "__main__":
    main()