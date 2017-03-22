import h5py
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.svm import LinearSVC,NuSVC
from sklearn.metrics import hamming_loss, accuracy_score
from sklearn.metrics import matthews_corrcoef, hamming_loss, zero_one_loss, coverage_error, label_ranking_average_precision_score, label_ranking_loss
from sklearn.preprocessing import scale
import numpy as np
from sklearn.externals import joblib
f = h5py.File("dataset_294.h5")
# f = h5py.File("../vgg_features/dataset_features.h5")
x = f['x'].value
y = f['y'].value
f.close()


x = scale(x)
# x = np.add(x,np.random.normal(size=4096))
# x =  x + 2 * np.random.normal(size=(2000,4096))

x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)
# tmp_x = x_train[0:400]
# tmp_y = y_train[0:400]
# batch_size = 200

def get_result(y_true,y_pred):
    total_correctly_predicted = len([i for i in range(len(y_true)) if (y_true[i]==y_pred[i]).sum() == 5])
    print("Fully correct output")
    print(total_correctly_predicted)
    print(total_correctly_predicted/400.)
    print("hamming loss")
    print(hamming_loss(y_true,y_pred))

# model = OneVsRestClassifier(NuSVC(kernel='rbf',gamma=0.0019,verbose=2, probability=True),n_jobs=-1)
# model = OneVsRestClassifier(LinearSVC(C = 0.000000000000001),n_jobs=-1)
model = OneVsRestClassifier(SVC(kernel='rbf',gamma=0.0020,C=5., probability=True ),n_jobs=-1)
# model = OneVsRestClassifier(RandomForestClassifier(n_estimators=1000,verbose=2,n_jobs=-1),n_jobs=-1)
# model = OneVsRestClassifier(LogisticRegression(),n_jobs=-1)
# model = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear',verbose=2 )),n_jobs=-1)
batch_x, batch_y = x_train[0:100],y_train[0:100]
model.fit(batch_x, batch_y)
y_pred = model.predict(x_test)
print("*"*100)
print("Trained on dataset:"+str(batch_x.shape))
get_result(y_test,y_pred)
batch = range(100,1600,100)
for i in batch:
    next_batch_x = x_train[i:i+100]
    next_batch_y = y_train[i:i+100]
    scores = np.abs(model.decision_function(next_batch_x))
    tmp_y = next_batch_y
    index = [i for i,Sum in enumerate(np.sum(scores<0.2,axis=1)) if Sum!=0]
    print("*"*100)
    print("Found "+str(len(index))+" uncertain examples")
    batch_x = np.vstack((batch_x,next_batch_x[index]))
    batch_y = np.vstack((batch_y,next_batch_y[index]))
    print("New Dataset shape"+str(batch_x.shape))
    model.fit(batch_x,batch_y)
    y_pred = model.predict(x_test)
    get_result(y_test,y_pred)   
    
    
    
    
    
