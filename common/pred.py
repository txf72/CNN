from common.layers_pred import BatchNormalization_pred


class pred:
    
    def __init__(self):
        
        # レイヤの生成
        self.layers = OrderedDict()
        self.layers["conv1"] =  Convolution(grad['W1'], grad['b1'],1,1)
        self.layers["batch1"] = BatchNormalization_pred(grad["gamma_1"], grad["beta_1"], move["move_mean_1"], move["move_var_1"])
        self.layers["relu1"] = ReLU()
        self.layers["pool1"] = MaxPooling(pool_h=2, pool_w=2, stride=2, pad=1)

        self.layers["conv2"] =  Convolution(grad['W2'], grad['b2'],1,1)
        self.layers["batch2"] = BatchNormalization(grad["gamma_2"], grad["beta_2"], move["move_mean_2"], move["move_var_2"])
        self.layers["relu2"] = ReLU()
        self.layers["pool2"] = MaxPooling(pool_h=2, pool_w=2, stride=2, pad=1)

        self.layers["conv3"] =  Convolution(grad['W3'], grad['b3'],1,1)
        self.layers["batch3"] = BatchNormalization_pred(grad["gamma_3"], grad["beta_3"], move["move_mean_3"], move["move_var_3"])
        self.layers["relu3"] = ReLU()
        self.layers["pool3"] = MaxPooling(pool_h=2, pool_w=2, stride=2, pad=1)
        
        self.layers["conv4"] =  Convolution(grad['W4'], grad['b4'],1,1)
        self.layers["batch4"] = BatchNormalization_pred(grad["gamma_4"], grad["beta_4"], move["move_mean_4"], move["move_var_4"])
        self.layers["relu4"] = ReLU()

        
        self.layers["conv5"] =  Convolution(grad['W5'], grad['b5'],1,1)
        self.layers["batch5"] = BatchNormalization_pred(grad["gamma_5"], grad["beta_5"], move["move_mean_5"], move["move_var_5"])
        self.layers["relu5"] = ReLU()

        self.layers["conv6"] =  Convolution(grad['W6'], grad['b6'],1,1)
        self.layers["batch6"] = BatchNormalization_pred(grad["gamma_6"], grad["beta_6"], move["move_mean_6"], move["move_var_6"])
        self.layers["relu6"] = ReLU()
        
        self.layers["affine1"] = Affine(grad['W7'], grad['b7'])
        self.layers["batch7"] = BatchNormalization_pred(grad["gamma_7"], grad["beta_7"], move["move_mean_7"], move["move_var_7"])
        self.layers["relu7"] = ReLU()
        
        self.layers["affine2"] = Affine(grad['W8'], grad['b8'])
        self.layers["batch8"] = BatchNormalization_pred(grad["gamma_8"], grad["beta_8"], move["move_mean_8"], move["move_var_8"])
        self.layers["relu8"] = ReLU()

        self.layers["affine3"] = Affine(grad['W9'], grad['b9'])
        self.last_layer = SoftmaxWithLoss()
        
        
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = t.argmax(axis=1)
        acc = np.sum(y == t) / x.shape[0]
        
        return acc

    def loss(self, x, t):
        """
        損失関数
        x : 入力データ
        t : 教師データ
        """
        y = self.predict(x)
        
        return self.last_layer.forward(y, t)
    
