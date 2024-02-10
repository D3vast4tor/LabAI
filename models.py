from typing import Any
import keras, random, time, math, names, json, os
from matplotlib.font_manager import json_dump
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
class Model:
    
    def __init__(self, dna1: dict | None = None, dna2: dict | None = None, config: Sequential | None = None):
        super()
        self.dna = dict()
        self.fitness = 0.
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.cifar10.load_data()
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 32, 32, 3)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 32, 32, 3)
        self.input_shape = (32,32,3)
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255
        
        if dna1 != None and dna2 != None:
            i1 = list(dna1).index("Flatten")
            n1_dna = list(dna1.items())
            n1_dna = n1_dna[:i1]
            i2 = list(dna2).index("Flatten")
            n2_dna = list(dna2.items())
            n2_dna = n2_dna[i2:]
            new_dna = n1_dna+n2_dna
            self.dna = dict(new_dna)
            self.model = Sequential()
            for _ in self.dna:
                self.model.add(self.dna[_])
            
            
            self.model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
            
            
        elif dna1 != None:
            self.dna = dna1.copy()
            
            self.model = Sequential()
            for _ in self.dna:
                self.model.add(self.dna[_])
            
            
            self.model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
            
        elif dna1 == None and dna2 == None and config == None:    
            
            self.randomize()
            
            self.model = Sequential()
            for _ in self.dna:
                self.model.add(self.dna[_])
            
            
            self.model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
             
        else:
            self.model = config
            assert self.model != None
            self.model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    def randomize(self):
        
        u_start = 256
        k_start = 64
        r = random.randint(3,6)
        if((r%2) == 0):
            r -= 1
        for i in range(1,r):
            self.dna["Conv2D_"+str(i)] = Conv2D(k_start,(3,3), input_shape = self.input_shape, padding = "same", activation=tf.nn.relu)
            k_start *= 2
            if((i%2) == 0):
                self.dna["MaxPooling2D_"+str(i)] = MaxPooling2D(pool_size = (2,2))
        self.dna["Dropout_1"] = Dropout(0.1)
        self.dna["Flatten"] = Flatten()
        r = random.randint(3,6)
        if((r%2) == 0):
            r -= 1
        for j in range(1,r//2):
            self.dna["Dense_"+str(j)] = Dense(u_start, activation = tf.nn.relu)
            u_start = (u_start//2)
        self.dna["Dropout_2"] = Dropout(0.1)
        self.dna["Dense"] = Dense(u_start//2, activation = tf.nn.softmax)

    def evaluate(self):
        assert self.model != None
        start_t = time.time()
        self.model.fit(x=self.x_train,y=self.y_train, epochs=2)
        end_t = time.time()
        train_duration = end_t - start_t
        res = self.model.evaluate(self.x_test, self.y_test,verbose = "1")
        self.fitness = res[1]/math.log(train_duration,10)
        return self.fitness

    def mutate(self):
        assert self.model != None
        nmodel = Sequential()
        ndna = {}
        i,j,k,l = 1,1,1,1
        for _ in self.model.layers:
            fbase = 1 << random.randrange(2,7)
            nbase = 1 << random.randrange(2,5)
            dropout_percent = 0.0
            
            while not  0.1 <= dropout_percent <= 0.5:
                dropout_percent = random.random()

            
            if isinstance(_,Conv2D) and i == 1:
                ndna["Conv2D_" + str(i)] = Conv2D(fbase, (3,3), input_shape = self.input_shape, padding = "same", activation = tf.nn.relu)
                nmodel.add(Conv2D(fbase, (3,3), input_shape = self.input_shape, padding = "same", activation = tf.nn.relu))
                i += 1
            elif isinstance(_,Conv2D) and random.randint(0,1):
                ndna["Conv2D_" + str(i)] = Conv2D(fbase, (3,3), activation = tf.nn.relu)
                nmodel.add(Conv2D(fbase, (3,3), input_shape = self.input_shape, padding = "same", activation = tf.nn.relu))
                i += 1
            if isinstance(_,MaxPooling2D):
                ndna["MaxPooling2D_"+str(l)] = MaxPooling2D(pool_size = (2,2))
                nmodel.add(MaxPooling2D(pool_size = (2,2)))
                l += 1
            if isinstance(_,Flatten):
                ndna["Flatten"] = Flatten()
                nmodel.add(Flatten())
            if isinstance(_,Dropout) and random.randint(0,1):
                ndna["Dropout_" + str(j)] = Dropout(dropout_percent)
                nmodel.add(Dropout(dropout_percent))
                j += 1
                
            if isinstance(_,Dense):
                ndna["Dense_"+str(k)] = Dense(nbase, tf.nn.relu)
                nmodel.add(Dense(nbase, tf.nn.relu))            
            ndna["Dense"] = Dense(10, tf.nn.softmax)
        
        nmodel.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy'])
        
        self.model = nmodel
        self.dna = ndna        
            
                

    def save_model(self, fitness: float):
        name = names.get_first_name(gender = "male")
        assert self.model != None
        config = self.model.get_config()
        config['fitness'] = fitness
        while(name in os.listdir("./models")):
            name = names.get_first_name(gender = "male")
        self.model._name = name
        with open("./models/" + name, "x") as f:
            json.dump(config, f)
            f.close()
            
    
def getBestIndividual(generation_path: str) -> dict:
    bmodel = ""
    bfitness = 0.
    for model in os.listdir(generation_path):
        with open(generation_path + model,"r+") as file:
            configs = json.load(file)
            if configs['fitness'] > bfitness:
                bfitness = configs['fitness']
                bmodel = str(model)
    return Model(config = configs)   
    
def load_model(model_name : str, generation: int) -> Sequential:
    return keras.Sequential.from_config(json.load(open("./generation_"+str(generation)+"/"+model_name)))