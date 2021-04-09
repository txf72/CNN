import numpy as np
from common.model import CustomConvNet





class training:
    def __init__(self, model, epochs, batch_size, optimizer,grad=None):
        
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.grad = self.grad
        
    def train(self,x,t,test,test_labels):
        


        """
        学習する
        """
        # 繰り返し回数
        xsize = x.shape[0]
        iter_num = np.ceil(xsize / self.batch_size).astype(np.int)
        
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []

        for epoch in range(self.epochs):
            print()
            print("    epoch=%s"%epoch)

            # シャッフル
            idx = np.arange(xsize)
            np.random.shuffle(idx)

            for it in range(iter_num):
                """
                ランダムなミニバッチを順番に取り出す
                """

                mask = idx[self.batch_size*it : self.batch_size*(it+1)]

                 # ミニバッチの生成
                x_train = x[mask]
                t_train = t[mask]

                  # 勾配の計算 (誤差逆伝播法を用いる) 
                grads,move = snet.gradient(x_train, t_train)

                 # 更新
                self.optimizer.update(model.params, grads)

            ## 学習経過の記録

            # 訓練データにおけるloss
            # print("calculating train_loss")    
            train_loss.append(self.model.loss(x,t))

            # print("calculating test_loss")
            # テストデータにおけるloss
            test_loss.append(self.model.loss(test,test_labels))


            # 訓練データにて精度を確認
            train_acc = self.model.accuracy(x,t)
            train_accuracy.append(train_acc)
            train_acc = print("    train_accuracy",train_acc)
            

            # テストデータにて精度を算出
            test_acc = self.model.accuracy(test,test_labels)
            test_accuracy.append(test_acc)
            test_acc = print("    test_accuracy",test_acc)
    
        return train_loss, test_loss, train_accuracy, test_accuracy

