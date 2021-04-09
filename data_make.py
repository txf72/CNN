import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()

def main(size):
    

    train_data = np.load(path)
    train_label = np.load(path)
    
    
    #dataをランダムに取り出すためにindexをシャッフル
    xsize = train_data.shape[0]
    idx = np.arange(xsize)
    np.random.shuffle(idx)
    

    print("学習データ総数＝",size)
    
    it = 2
    train_data = train_data[it*size:(it+1)*size]
    train_label = train_label[it*size:(it+1)*size]
    print("train_data",train_data.shape)
    print("train_label",train_label.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_label,
                                                        test_size=0.3, random_state=1234,
                                                        shuffle=True)
    X_train = X_train/255
    X_test = X_test/255
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)
    
    y_train_ = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
    x = X_train
    t = y_train
    
    test = X_test
    test_labels = y_test
    
    print("train",x.shape)
    print("test",test.shape)

    return x, t, test, test_labels