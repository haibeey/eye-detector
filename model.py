import tensorflow as tf
from math import pi
import numpy as np
from random import randint
import cv2
keras=tf.keras
from keras import models,layers


def toPixel(rgb):
    return (255&255)<<24|(rgb[0]&255)<<16|(rgb[1]&255)<<8|rgb[2]&255
class dataset:
    def __init__(self):
        self.image=[]
        self.label=[]
        self.label_for_sk=[]
        self.index=0
        self.epoch=1
        self.batch=20
        with open("C:\Python35\eye detection\skin pixel dataset.txt") as f:
            for line in f:
                line_split=[rgb for rgb in map(int,line.strip().split("\t"))]
                rgb=[rgb for rgb in line_split[:3]]
                self.image.append(rgb)
                label_value=[0,0]
                label_value[line_split[-1]%2]=1
                self.label.append(label_value)
                self.label_for_sk.append([line_split[-1]%2])
        
    def add_pixel(self,pixel,label):
        self.image.append(pixel)
        self.label.append(label)
    def set_batch(self,batch):
        self.batch=batch
    def set_epoch(self,epoch):
        self.epoch=epoch
    def shuffle(self):
        len_image=len(self.image)
        for index in range(len_image):
            index_random=randint(0,len_image-1)
            self.image[index],self.image[index_random]=self.image[index_random],self.image[index]
            self.label[index],self.label[index_random]=self.label[index_random],self.label[index]
            self.label_for_sk[index],self.label_for_sk[index_random]=self.label_for_sk[index_random],self.label_for_sk[index]

    def get_next_batch(self):
        if self.index>=len(self.image):
            self.index=0
            if self.epoch<=0:
                return False
            self.epoch-=1
            self.shuffle()
        a=self.image[self.index:self.index+self.batch]
        b=self.label_for_sk[self.index:self.index+self.batch]
        self.index+=self.batch
        return a,b

class  model(object):
    def __init__(self,dataset,is_training=True):
        self.graph_after_train=None
        if not is_training:
            return

        self.dataset=dataset
        self.c=tf.placeholder(tf.float32,[None,3])
    
        #initializer = tf.contrib.layers.xavier_initializer()
        self.layer1 = tf.layers.dense(tf.layers.dense(self.c, 10, activation=tf.nn.relu),10,activation=tf.nn.relu)
    
        self.out = tf.layers.dense(self.layer1, 1, activation=tf.nn.sigmoid)
        self.label=tf.placeholder(tf.float32,[None,1])
        
        print(self.label.shape,self.out.shape,self.layer1.shape)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,logits=self.out))
        self.train = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(self.loss)

        self.md=models.Sequential()

    def keras_train(self):
        self.md.add(layers.Dense(10,activation="relu",input_shape=(3,),name="input"))
        self.md.add(layers.Dense(5,activation="relu"))
        self.md.add(layers.Dense(1,activation="sigmoid",name="out"))

        self.md.compile(optimizer="rmsprop",
                    loss="binary_crossentropy",
                    metrics=["accuracy"])
        self.md.summary()

        history=self.md.fit(np.asarray(self.dataset.image),
                  np.asarray(self.dataset.label_for_sk),
                  epochs=2,
                  batch_size=10,
                  validation_data=(np.asarray(self.dataset.image),
                  np.asarray(self.dataset.label_for_sk)))
        self.md.save("model.h5")
        print(self.md.predict(np.asarray(self.dataset.image)))

    def predict_keras(self,pred):
        return self.md.predict(pred)
    def freeze_keras(self,sess):
        from  tensorflow.python.framework.graph_util import convert_variables_to_constants 
        graph=sess.graph
        with graph.as_default():
            freeze_var_name=list(v.op.name for v in tf.global_variables())
            output_name=[out.op.name for out in self.md.outputs]
            input_graph_def=graph.as_graph_def()
            frozen_graph=convert_variables_to_constants(sess,input_graph_def,output_name,freeze_var_name)
            return frozen_graph
    def keras_to_pb(self):
        from keras import backend as K
        fg=self.freeze_keras(K.get_session())
        tf.train.write_graph(fg,"C:\\Python35\\eye detection","model_pb.pb",as_text=False)
    def train_skin(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            next_batch=self.dataset.get_next_batch()
            print(next_batch)
            while next_batch:
                print(sess.run([self.train,self.loss],feed_dict={self.c:next_batch[0],self.label:next_batch[1]}))
                next_batch=self.dataset.get_next_batch()

            correct_prediction = tf.equal(np.round(sess.run(self.out, feed_dict={self.c: self.dataset.image,
                                      self.label: self.dataset.label_for_sk})), self.dataset.label_for_sk)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run([correct_prediction,accuracy]))#np.asarray(self.dataset.image),np.asarray(self.dataset.label_for_sk))
            

    def predict_skin(self,rgb):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            return sess.run(self.out,feed_dict={self.c:rgb})


    def load_graph(self,frozen_graph_filename):
        # We load the protobuf file from the disk and parse it to retrieve the 
        # unserialized graph_def
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it 
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="prefix")
        return graph

    def predict_loaded(self,frozen_graph_filename,rgb):
        if self.graph_after_train is None:
            self.graph_after_train=self.load_graph(frozen_graph_filename)
        
        x = self.graph_after_train.get_tensor_by_name("prefix/input_input:0")
        y = self.graph_after_train.get_tensor_by_name("prefix/out/Sigmoid:0")
        
        with tf.Session(graph=self.graph_after_train) as sess:
            return sess.run(y,feed_dict={x:rgb})
    

    def predict_skin_(self,frozen_graph_filename,array_to_pre):
        graph=self.load_graph(frozen_graph_filename)
        # for op in graph.get_operations():
        #     print(op.name)
    
        # We launch a Session
        with tf.Session(graph=graph) as sess:
           pass

if __name__=="__main__":
    d=dataset()
    d.shuffle()
    m=model(d)
    m.keras_train()
    m.keras_to_pb()

    image_path="C:\\Python35\\eye detection\\a.jpg"
    getImage=lambda x:np.array(cv2.imread(x),dtype=np.float32)
    image=getImage(image_path)
    image_h=[]

    for row in image:
        pred=m.predict_keras(row)
        this_row=[]
        for i in pred:
            if i[0]<=0.5:
                this_row.append([0,0,0])
            else:
                this_row.append([255,255,255])
        #print(this_row,pred)
        image_h.append(this_row)
    image_h=np.asarray(image_h)
    cv2.imwrite('color_img.jpg', image_h)
    cv2.imshow("color_img.jpg", image_h)
    # #cv2.waitKey()
    print(m.predict_keras(np.asarray([[102,143,192],[164,162,114],[255,255,255]])))