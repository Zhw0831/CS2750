import pandas as pd 
import numpy as np
import pickle

class BBN():
    def __init__(self):
        # the columns: fever, paleness, cough, wbc
        self.attributes = ['Fever', 'Paleness', 'Cough', 'HighWBC']
        self.ml = []
        self.df = pd.read_csv(r'C:\Users\Zhen Wu\Desktop\CS2750\hw7\HW-7-Data\Pneumonia.csv')
        self.pneumonia_t = self.df[self.df['Pneumonia']==1]
        print('frequency of pneumonia true is %f' %(len(self.pneumonia_t.values)/len(self.df.values)))
        self.pneumonia_f = self.df[self.df['Pneumonia']==0]
        print('frequency of pneumonia false is %f' %(len(self.pneumonia_f.values)/len(self.df.values)))

    def ml_calc(self,i,j,flag):
        estimate = []
        # if flag = 1, i.e., pneumonia is true
        if(flag==1):
            # retrieve the instances of parameter i given pneumonia is true
            para_t = self.pneumonia_t[self.pneumonia_t[self.attributes[i]]==1]
            para_f = self.pneumonia_t[self.pneumonia_t[self.attributes[i]]==0]
            # calculate the ml estimate
            # if estimate true
            if(j==1):
                ml_true = len(para_t)/len(self.pneumonia_t)
                return ml_true
            # if estimate false
            elif(j==0):
                ml_false = len(para_f)/len(self.pneumonia_t)
                return ml_false

        # if flag = 0, i.e., pneumonia is true
        elif(flag==0):
            # retrieve the instances of parameter i given pneumonia is false
            para_t = self.pneumonia_f[self.pneumonia_f[self.attributes[i]]==1]
            para_f = self.pneumonia_f[self.pneumonia_f[self.attributes[i]]==0]
            # calculate the ml estimate
            # if estimate true
            if(j==1):
                ml_true = len(para_t)/len(self.pneumonia_f)
                return ml_true
            # if estimate false
            elif(j==0):
                ml_false = len(para_f)/len(self.pneumonia_f)
                return ml_false;
    
    def estimate_para(self):
        # the number of rows and columns in the ml estimate matrix
        rows = len(self.attributes)*2
        # columns takes t/f of pneumonia
        cols = 2
        matrix = np.zeros(shape=(rows, cols), dtype=float)

        # the matrix is organized like this:
        #               pneumonia_false      pneumonia_true
        #  fever_false
        #  fever_true
        #  similar for every attribute
        counter = 0
        for i in range(len(self.attributes)):
            matrix[counter,0] = self.ml_calc(i,0,0)
            matrix[counter+1,0] = self.ml_calc(i,1,0)
            matrix[counter,1] = self.ml_calc(i,0,1)
            matrix[counter+1,1] = self.ml_calc(i,1,1)
            counter = counter + 2
        
        print(matrix)
        self.ml = matrix


a = BBN()
a.estimate_para()

pickle.dump(a,open(r'C:\Users\Zhen Wu\Desktop\CS2750\hw7\HW-7-Data\bbn','wb'))


