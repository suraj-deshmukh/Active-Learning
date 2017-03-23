import h5py
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss, accuracy_score
import numpy as np
f = h5py.File("dataset_features.h5")
x = f['x'].value
y = f['y'].value
f.close()



x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)


def get_result(y_true,y_pred):
    total_correctly_predicted = len([i for i in range(len(y_true)) if (y_true[i]==y_pred[i]).sum() == 5])
    print("Fully correct output")
    print(total_correctly_predicted)
    print(total_correctly_predicted/400.)
    print("hamming loss")
    print(hamming_loss(y_true,y_pred))


model = OneVsRestClassifier(SVC(),n_jobs=-1)

batch_x, batch_y = x_train[0:40],y_train[0:40]
model.fit(batch_x, batch_y)
y_pred = model.predict(x_test)
print("*"*100)
print("Trained on dataset:"+str(batch_x.shape))
get_result(y_test,y_pred)
batch = range(40,1600,60)
for i in batch:
    next_batch_x = x_train[i:i+60]
    next_batch_y = y_train[i:i+60]
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
