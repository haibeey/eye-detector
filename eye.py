import cv2
import numpy as np
from model import model

#trained model to perform pixel classification
m=model(None,is_training=None)

class eye(object):
    def __init__(self,image):
        self.image=image# image is image tensor contaning only black and white region
        self.eyes_finder=UnionAndRank(self.image)
        self.elipse_region={}
        self.weed_out()
        
    def weed_out(self):
        index=0
        eyes_region={}
        for region in self.eyes_finder.parent:
            x,y=self.eyes_finder.get_x_y(region)
            index_x,index_y=self.eyes_finder.get_x_y(index)
            
            if self.eyes_finder.is_black_pixel(self.image[x][y]):
                if not region in eyes_region:
                    eyes_region[region]=[(x,y)]
                eyes_region[region].append((index_x,index_y))
            index+=1
        
        for region in eyes_region:
            self.elipse_region[region]=elipse(eyes_region[region])
            self.elipse_region[region].fit()
            print(self.elipse_region[region])

    def get_eyes(self):
        eyes=[]
        print(len(self.elipse_region))
        for region_x in self.elipse_region:
            eye1=self.elipse_region[region_x]
            if eye1.a<=0 or eye1.b<=0 or not 1<=eye1.b/eye1.a <=3:continue
            for region_y in self.elipse_region:
                if region_x!=region_y:
                    
                    eye2=self.elipse_region[region_y]
                    if eye1.a<=0 or eye1.b<=0 or eye2.a<=0 or eye2.b<=0:continue
                    if 1<=eye2.b/eye2.a <=3:
                        if eye1.calc_orientation(eye2) <=20:
                            if (eye1.b+eye2.b)/2< eye.dist((eye1.y,eye1.x),(eye2.y,eye2.x))<3*(eye1.b+eye2.b)/2:
                                eyes.extend([eye1,eye2])
                                print(self.elipse_region[region_x],self.elipse_region[region_y]," i found its")
        return eyes

    @classmethod
    def dist(cls,x,y):
        return ((x[0]-y[0])**2+(x[1]-y[1])**2)**0.5
                         

class elipse:
    def __init__(self,region):
        self.region=region
        self.a=0 #minor axis
        self.b=0 #major axis
        self.orientation=0
        self.x=0
        self.y=0
    def fit(self):
        self.region.sort()
        self.x,self.y=self.region[len(self.region)//2]
        self.a=abs(self.region[0][0]-self.region[-1][0])
        self.region.sort(key=lambda x:x[1])
        self.b=abs(self.region[0][1]-self.region[-1][1])
        self.orientation=abs(self.region[0][0]-self.region[-1][0])
    def calc_orientation(self,other):
        hypoteneus=eye.dist((self.x,self.y),(other.x,other.y))
        adjacent=abs(self.y-other.y)
        return (hypoteneus**2-adjacent**2)**0.5

    def __repr__(self):
        return str((str(len(self.region)),self.orientation,self.a,self.b,self.y,self.x))

class UnionAndRank:
    def __init__(self,image):
        self.image=image
        self.len_x=len(self.image)
        self.len_y=len(self.image[0])
        print(self.len_x,self.len_y)
        self.rank=[1]*(self.len_x*self.len_y)
        self.parent=[0]*(self.len_x*self.len_y)
        self.first_white=0
        self.fill_parent()
        self.get_potential_eyes()
        
    def fill_parent(self):
        for i in range(self.len_x):
            for j in range(self.len_y):
                if not self.is_black_pixel(self.image[i][j]):
                    self.first_white=UnionAndRank.get_index_value(i,j)
                self.parent[UnionAndRank.get_index_value(i,j)]=UnionAndRank.get_index_value(i,j)

    @classmethod
    def is_black_pixel(self,rgb):
        return all(rgb[i]<10 for i in range(3))
    
    def get_potential_eyes(self):
        
        dx=[-1,0,-1,-1,0,1,1,1]
        dy=[0,-1,-1,1,1,1,0,-1]
        for i in range(self.len_x):
            for j in range(self.len_y):
                if self.is_black_pixel(self.image[i][j]):
                    for k in range(8):
                        if 0<=i+dx[k]<self.len_x and 0<=j+dy[k]<self.len_y:
                            if self.is_black_pixel(self.image[i+dx[k]][j+dy[k]]):
                            #print(self.image[i][j],i,j,i+dx[k],j+dy[k],"nigga")
                                
                                self.union(self.get_index_value(i,j,x_len=self.len_x),\
                                            self.get_index_value(i+dx[k],j+dy[k],x_len=self.len_x))
                                break
   
                            else:
                                self.union(self.first_white,self.get_index_value(i+dx[k],j+dy[k],x_len=self.len_x))

    @staticmethod
    def get_index_value(x,y,x_len=200):
        return x_len*y+x
    @staticmethod
    def get_x_y(value,x_len=200):
        return value%x_len,value//x_len

    def find(self,x): 
        if x!=self.parent[x]:
            self.parent[x]=self.find(self.parent[x])
        return self.parent[x]

    def union(self,x,y):
        a=self.find(x)
        b=self.find(y)
        #print(a,b)

        if a==b:return
        
        if self.rank[a]>self.rank[b]:
            self.parent[b]=self.parent[a]
        else:
            self.parent[a]=self.parent[b]
            if self.rank[a]==self.rank[b]:
                self.rank[a]+=1


class AugmentEyes:

    getImage=lambda s,x:np.array(cv2.imread(x),dtype=np.float32)

    def __init__(self,image_or_image_path):
        if isinstance(image_or_image_path,str):
            self.image=self.getImage(image_or_image_path)
        elif isinstance(image_or_image_path,np.ndarray):
            if not image_or_image_path.shape[-1]==3:
                raise ValueError("no an image type")
            self.image=image_or_image_path
        elif isinstance(image_or_image_path,list):
            try:
                true=len(image_or_image_path[-1][0])==3
                if true:
                    self.image=image_or_image_path
                else:
                    raise ValueError("invalid image type")
            except IndexError:
                raise ValueError("invalid image type")
        else:
            raise ValueError("invalid image type")
        self.eye=eye#empty at first
        self.image_bw=self.pred()
        cv2.imwrite("ima1.jpg",self.image_bw)

    def pred(self):
        new_image=[]
        for row in self.image:
            pred=m.predict_loaded("C:\\Python35\\eye detection\\model_pb.pb",row)
            this_row=[]
            for i in pred:
                if i[0]<0.5:
                    this_row.append([0,0,0])
                else:
                    this_row.append([255,255,255])
            #print(this_row,pred)
            new_image.append(this_row)
        new_image=np.asarray(new_image)
        self.eye=eye(new_image)
        return new_image

    def stylize(self,color,action="stylize"):
        if action=="stylize":
            eyes=self.eye.get_eyes()
            for r in eyes:
                self.style(r.region,color)
            return self.image
        else:
            raise ValueError("not supported")
    def style(self,region,color):
        for r in region:
            self.image[r[0]][r[1]]=color
    
    def show_image(self):
        cv2.imshow("image", self.image)
        cv2.imwrite("sample_out.jpg",self.image)
        cv2.waitKey()
    
if __name__=="__main__":
    image_path="C:\\Python35\\eye detection\\sample_input.jpg"
    getImage=lambda x:np.array(cv2.imread(x),dtype=np.float32)
    image=getImage(image_path)
    image=cv2.resize(image,(200,200))
    ae=AugmentEyes(image)
    ae.stylize([0,0,100])
    ae.show_image()