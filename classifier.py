import numpy as np
import csv
import reader1
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

data1_dir = 'C:/Users/Dongjie/Desktop/4502 project/player_data.csv'
data1 = reader1.read_Data(data1_dir)

data2_dir = 'C:/Users/Dongjie/Desktop/4502 project/Seasons_Stats.csv'
data2 = reader1.read_Data(data2_dir)


a= np.stack([data1[:,0],data1[:,3],data1[:,4],data1[:,5],data1[:,1]],axis=1)
print(a)
dict1={}
n = 0   
for i in range(data1.shape[0]):
    if a[i][1] in dict1.keys():
        a[i][1] = dict1[a[i][1]]
    else:
        n = n + 1
        dict1[a[i][1]] = n
        a[i][1] = dict1[a[i][1]]

    
for i in range(data1.shape[0]):
    h1 = int(a[i][2][0])
    h2 = int(a[i][2][2:])    
    a[i][2] = h1*30.5+h2*2.54

c = np.zeros(data1.shape[0],dtype=np.dtype((str, 16)))
for i in range(data1.shape[0]):
    for j in range(data2.shape[0]):
        if a[i][4] == data2[j][1] and a[i][0] == data2[j][2]:
            c[i] = data2[j][5]
            


a = np.stack([a[:,1],a[:,2],a[:,3],c],axis=1)
dict2={}
n = 0
for i in range(data1.shape[0]):
    if a[i][3] in dict2.keys():
        a[i][3] = dict2[a[i][3]]
    else:
        n = n + 1
        dict2[a[i][3]] = n
        a[i][3] = dict2[a[i][3]]


print(a)


a = Imputer().fit_transform(a)

'''
for i in range(a.shape[0]):
    if a[i][1] < 0
'''




def logistic_Regression_Classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model

def random_Forest_Classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model

def decision_Tree_Classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model



def cal_result(pre,lab):
    rig_num = 1.0;
    wro_num = 1.0;
    for i in range(len(pre)):
        if pre[i] == lab[i]:
            rig_num += 1
        else:
            wro_num += 1
    print("the right riatio is:",rig_num/(rig_num+wro_num))

def F1(pre,lab):
    Tp = np.ones((1,12))
    Fp = np.ones((1,12))
    Fn = np.ones((1,12))

    p = np.ones((1,12))
    r = np.ones((1,12))
    f = np.zeros((1,12))

    F = 0
    ratio = np.zeros((1,12))
    for i in range(len(pre)):
        for j in range(12):
            if lab[i] == j+1 and pre[i] ==j+1:
                Tp[0,j] += 1
            if lab[i] != j+1 and pre[i] ==j+1:
                Fp[0,j] += 1
            if lab[i] == j+1 and pre[i] != j+1:
                Fn[0,j] +=1
            if lab[i] == j+1:
                ratio[0,j] += 1

    
    for j in range(12):
        p[0,j] = 1.0*Tp[0,j]/(Tp[0,j]+Fp[0,j])
        r[0,j] = 1.0*Tp[0,j]/(Tp[0,j]+Fn[0,j])
        f[0,j] = 2.0*p[0,j]*r[0,j]/(p[0,j]+r[0,j])
        F += ratio[0,j]*f[0,j]
    F = 1.0*F/sum(ratio[0,:])
    #print("the class ratio:",ratio/1.0*sum(ratio[0,:]))
    print("the F1 sorce is: ",F)
                

def classifier_Machine(dir,test_cho,pre_cho,user_list,section,train_x,train_label,predict_x,predict_label,test_data):
    if 1 in section:
        print("Random Forest Classifier")
        filename = 'pre_RF.csv'
        model = random_Forest_Classifier(train_x,train_label)
        if test_cho == True:
            cal_result(model.predict(predict_x),predict_label)
            F1(model.predict(predict_x),predict_label)
    if 2 in section:
        print("Logistic Regression Classifier")
        filename = 'pre_LR.csv'
        model = logistic_Regression_Classifier(train_x,train_label)
        if test_cho == True:
            cal_result(model.predict(predict_x),predict_label)
            F1(model.predict(predict_x),predict_label)
    if 3 in section:
        print("Decision Tree Classifier")
        filename = 'pre_DT.csv'
        model = decision_Tree_Classifier(train_x,train_label)
        if test_cho == True:
            cal_result(model.predict(predict_x),predict_label)
            F1(model.predict(predict_x),predict_label)

'''
    if pre_cho == True:
        test1 = reader1.read_Data('E:/csv/test.csv')
        
        predict_label1 = np.zeros((892816,2))
        predict_label1[:,0] = test1[:,0]
        predict_label1[:,1] = model.predict(test1[:,1:])[:,1]
        reader1.write_result('E:/csv/'+filename,predict_label1) 
'''

def get_Classifier():
    names = dict(RF = 1,LR = 2,DT = 3)
    return names
print("prepare the train_data")

y = a[:,-1].astype(np.float64)
train_data = a[:,0:-1].astype(np.float64)



X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size=0.8, random_state=42)

claName =get_Classifier()
section= [claName['RF']]   #在这里设置分类器，参数分别是RF，LR，DT

dir = 'D:/yizhi/competetion/test_features/ter/'
classifier_Machine(dir,\
                   test_cho = True,\
                   pre_cho = False,\
                   user_list = 0 ,\
                   section = section,\
                   train_x = X_train,\
                   train_label = y_train,\
                   predict_x = X_test,\
                   predict_label = y_test,\
                   test_data = 0)



        
    



