from common.optimizer import Adam
from common.layers import Convolution, MaxPooling, ReLU, Affine, SoftmaxWithLoss, BatchNormalization
from collections import OrderedDict
import numpy as np


class CustomConvNet:
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param_1={'filter_num':20, 'filter_size':3,'pad':2, 'stride':1},
                 conv_param_2={'filter_size':2},
                 conv_param_3={'filter_size':1},
                 pool_param={'pool_size':2, 'pad':2, 'stride':2},
                 hidden_size_1=100, hidden_size_2=45,output_size=15, weight_init_std=0.01):
        """
        input_size : tuple, 入力の配列形状(チャンネル数、画像の高さ、画像の幅)
        conv_param : dict, 畳み込みの条件
        pool_param : dict, プーリングの条件
        hidden_size : int, 隠れ層のノード数
        output_size : int, 出力層のノード数
        weight_init_std ： float, 重みWを初期化する際に用いる標準偏差
        """
                
        filter_num = conv_param_1['filter_num']
        filter_size_1 = conv_param_1['filter_size']
        filter_size_2 = conv_param_2['filter_size']
        filter_size_3 = conv_param_3['filter_size']
        filter_pad = conv_param_1['pad']
        filter_stride = conv_param_1['stride']
        
        pool_size = pool_param['pool_size']
        pool_pad = pool_param['pad']
        pool_stride = pool_param['stride']
        
        input_size = input_dim[1]
        conv_output_size_1 = (input_size + 2*filter_pad - filter_size_1) // filter_stride + 1 # 畳み込み後のサイズ(H,W共通)
        pool_output_size_1 = (conv_output_size_1 + 2*pool_pad - pool_size) // pool_stride + 1 # プーリング後のサイズ(H,W共通)
        conv_output_size_2 = (pool_output_size_1 + 2*filter_pad - filter_size_2) // filter_stride + 1 # 畳み込み後のサイズ(H,W共通)
        pool_output_size_2 = (conv_output_size_2 + 2*pool_pad - pool_size) // pool_stride + 1 # プーリング後のサイズ(H,W共通)
        conv_output_size_3 = (pool_output_size_2 + 2*filter_pad - filter_size_2) // filter_stride + 1 # 畳み込み後のサイズ(H,W共通)
        pool_output_size_3 = (conv_output_size_3 + 2*pool_pad - pool_size) // pool_stride + 1 # プーリング後のサイズ(H,W共通)
        conv_output_size_4 = (pool_output_size_3 + 2*filter_pad - filter_size_2) // filter_stride + 1 # 畳み込み後のサイズ(H,W共通)
        conv_output_size_5 = (conv_output_size_4 + 2*filter_pad - filter_size_3) // filter_stride + 1 # 畳み込み後のサイズ(H,W共通)
        conv_output_size_6 = (conv_output_size_5 + 2*filter_pad - filter_size_3) // filter_stride + 1 # 畳み込み後のサイズ(H,W共通)
        conv_output_pixel_6 = filter_num * conv_output_size_6 ** 2
        

            
        
        # 重みの初期化
        self.params = {}
        std = weight_init_std
        self.params['W1'] = std * np.random.randn(filter_num, input_dim[0], filter_size_1, filter_size_1) # W1は畳み込みフィルターの重みになる
        self.params['b1'] = np.zeros(filter_num)
        self.params["gamma_1"] = np.ones(filter_num)
        self.params["beta_1"] = np.zeros(filter_num)

        
        
        self.params['W2'] = std * np.random.randn(filter_num, filter_num, filter_size_2, filter_size_2) # 畳み込みフィルターの重みになる
        self.params['b2'] = np.zeros(filter_num)
        self.params["gamma_2"] = np.ones(filter_num)
        self.params["beta_2"] = np.zeros(filter_num)
   
        
        self.params['W3'] = std * np.random.randn(filter_num, filter_num, filter_size_2, filter_size_2) # 畳み込みフィルターの重みになる
        self.params['b3'] = np.zeros(filter_num)
        self.params['b3'] = np.zeros(filter_num)
        self.params["gamma_3"] = np.ones(filter_num)
        self.params["beta_3"] = np.zeros(filter_num)

        
        self.params['W4'] = std * np.random.randn(filter_num, filter_num, filter_size_2, filter_size_2) # 畳み込みフィルターの重みになる
        self.params['b4'] = np.zeros(filter_num)
        self.params["gamma_4"] = np.ones(filter_num)
        self.params["beta_4"] = np.zeros(filter_num)
       
        
        self.params['W5'] = std * np.random.randn(filter_num, filter_num, filter_size_3, filter_size_3) # 畳み込みフィルターの重みになる
        self.params['b5'] = np.zeros(filter_num)
        self.params["gamma_5"] = np.ones(filter_num)
        self.params["beta_5"] = np.zeros(filter_num)

        
        self.params['W6'] = std * np.random.randn(filter_num, filter_num, filter_size_3, filter_size_3) # 畳み込みフィルターの重みになる
        self.params['b6'] = np.zeros(filter_num)
        self.params["gamma_6"] = np.ones(filter_num)
        self.params["beta_6"] = np.zeros(filter_num)

        
        self.params['W7'] = std *  np.random.randn(conv_output_pixel_6, hidden_size_1)
        self.params['b7'] = np.zeros(hidden_size_1)
        self.params["gamma_7"] = np.ones(hidden_size_1)
        self.params["beta_7"] = np.zeros(hidden_size_1)

        
        self.params['W8'] = std *  np.random.randn(hidden_size_1, hidden_size_2)
        self.params['b8'] = np.zeros(hidden_size_2)
        self.params["gamma_8"] = np.ones(hidden_size_2)
        self.params["beta_8"] = np.zeros(hidden_size_2)

        
        self.params['W9'] = std *  np.random.randn(hidden_size_2, output_size)
        self.params['b9'] = np.zeros(output_size)
        
        

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param_1['stride'], conv_param_1['pad']) # W1が畳み込みフィルターの重み, b1が畳み込みフィルターのバイアスになる
        self.layers["BatchNormalization1"] = BatchNormalization(self.params["gamma_1"], self.params["beta_1"])                                                               
        self.layers['ReLU1'] = ReLU()
        self.layers['Pool1'] = MaxPooling(pool_h=pool_size, pool_w=pool_size, stride=pool_stride, pad=pool_pad)
        
        
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           conv_param_1['stride'], conv_param_1['pad']) # W1が畳み込みフィルターの重み, b1が畳み込みフィルターのバイアスになる
        self.layers["BatchNormalization2"] = BatchNormalization(self.params["gamma_2"], self.params["beta_2"])                                                               
        self.layers['ReLU2'] = ReLU()
        self.layers['Pool2'] = MaxPooling(pool_h=pool_size, pool_w=pool_size, stride=pool_stride, pad=pool_pad)
        
        
        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'],
                                           conv_param_1['stride'], conv_param_1['pad']) # W1が畳み込みフィルターの重み, b1が畳み込みフィルターのバイアスになる
        self.layers["BatchNormalization3"] = BatchNormalization(self.params["gamma_3"], self.params["beta_3"])                                                               
        self.layers['ReLU3'] = ReLU()
        self.layers['Pool3'] = MaxPooling(pool_h=pool_size, pool_w=pool_size, stride=pool_stride, pad=pool_pad)
        
        
        self.layers['Conv4'] = Convolution(self.params['W4'], self.params['b4'],
                                           conv_param_1['stride'], conv_param_1['pad']) # W1が畳み込みフィルターの重み, b1が畳み込みフィルターのバイアスになる
        self.layers["BatchNormalization4"] = BatchNormalization(self.params["gamma_4"], self.params["beta_4"])                                                               
        self.layers['ReLU4'] = ReLU()      
        
        self.layers['Conv5'] = Convolution(self.params['W5'], self.params['b5'],
                                           conv_param_1['stride'], conv_param_1['pad']) # W1が畳み込みフィルターの重み, b1が畳み込みフィルターのバイアスになる
        self.layers["BatchNormalization5"] = BatchNormalization(self.params["gamma_5"], self.params["beta_5"])                                                              
        self.layers['ReLU5'] = ReLU()
        
        self.layers['Conv6'] = Convolution(self.params['W6'], self.params['b6'],
                                           conv_param_1['stride'], conv_param_1['pad']) # W1が畳み込みフィルターの重み, b1が畳み込みフィルターのバイアスになる
        self.layers["BatchNormalization6"] = BatchNormalization(self.params["gamma_6"], self.params["beta_6"])                                                   
        self.layers['ReLU6'] = ReLU()        
               
        
        self.layers['Affine1'] = Affine(self.params['W7'], self.params['b7'])
        self.layers["BatchNormalization7"] = BatchNormalization(self.params["gamma_7"], self.params["beta_7"])
        self.layers['ReLU7'] = ReLU()
        
        self.layers['Affine2'] = Affine(self.params['W8'], self.params['b8'])
        self.layers["BatchNormalization8"] = BatchNormalization(self.params["gamma_8"], self.params["beta_8"])
        self.layers['ReLU8'] = ReLU()
        
        self.layers['Affine3'] = Affine(self.params['W9'], self.params['b9'])
        self.last_layer = SoftmaxWithLoss()

        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """
        損失関数
        x : 入力データ
        t : 教師データ
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=500):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def gradient(self, x, t):
        """勾配を求める（誤差逆伝播法）
        Parameters
        ----------
        x : 入力データ
        t : 教師データ
        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Conv3'].dW, self.layers['Conv3'].db
        grads['W4'], grads['b4'] = self.layers['Conv4'].dW, self.layers['Conv4'].db
        grads['W5'], grads['b5'] = self.layers['Conv5'].dW, self.layers['Conv5'].db
        grads['W6'], grads['b6'] = self.layers['Conv6'].dW, self.layers['Conv6'].db
        grads['W7'], grads['b7'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W8'], grads['b8'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W9'], grads['b9'] = self.layers['Affine3'].dW, self.layers['Affine3'].db
        
        
        grads["gamma_1"],grads["beta_1"] = self.layers["BatchNormalization1"].dgamma, self.layers["BatchNormalization1"].dbeta
        grads["gamma_2"],grads["beta_2"] = self.layers["BatchNormalization2"].dgamma, self.layers["BatchNormalization2"].dbeta
        grads["gamma_3"],grads["beta_3"] = self.layers["BatchNormalization3"].dgamma, self.layers["BatchNormalization3"].dbeta
        grads["gamma_4"],grads["beta_4"] = self.layers["BatchNormalization4"].dgamma, self.layers["BatchNormalization4"].dbeta
        grads["gamma_5"],grads["beta_5"] = self.layers["BatchNormalization5"].dgamma, self.layers["BatchNormalization5"].dbeta
        grads["gamma_6"],grads["beta_6"] = self.layers["BatchNormalization6"].dgamma, self.layers["BatchNormalization6"].dbeta
        grads["gamma_7"],grads["beta_7"] = self.layers["BatchNormalization7"].dgamma, self.layers["BatchNormalization7"].dbeta
        grads["gamma_8"],grads["beta_8"] = self.layers["BatchNormalization8"].dgamma, self.layers["BatchNormalization8"].dbeta
        
        
        
        #移動平均 
        move = {}
        move["move_mean_1"],move["move_var_1"] = self.layers["BatchNormalization1"].moving_mean, self.layers["BatchNormalization1"].moving_var
        move["move_mean_2"],move["move_var_2"] = self.layers["BatchNormalization2"].moving_mean, self.layers["BatchNormalization2"].moving_var
        move["move_mean_3"],move["move_var_3"] = self.layers["BatchNormalization3"].moving_mean, self.layers["BatchNormalization3"].moving_var
        move["move_mean_4"],move["move_var_4"] = self.layers["BatchNormalization4"].moving_mean, self.layers["BatchNormalization4"].moving_var
        move["move_mean_5"],move["move_var_5"] = self.layers["BatchNormalization5"].moving_mean, self.layers["BatchNormalization5"].moving_var
        move["move_mean_6"],move["move_var_6"] = self.layers["BatchNormalization6"].moving_mean, self.layers["BatchNormalization6"].moving_var
        move["move_mean_7"],move["move_var_7"] = self.layers["BatchNormalization7"].moving_mean, self.layers["BatchNormalization7"].moving_var
        move["move_mean_8"],move["move_var_8"] = self.layers["BatchNormalization8"].moving_mean, self.layers["BatchNormalization8"].moving_var
       
        
        return grads,move