
from keras.callbacks import Callback

class KerasMetricsSplitter(Callback):

    def __init__(self, train_tb=None, test_tb=None):
        """
        Construct a KerasMetricsSplitter
        
        :param train_tb: a keras.callbacks.TensorBoard object to recieve training metrics
        :param test_tb: a keras.callbacks.TensorBoard object to recieve test/validation metrics
        """
        super(KerasMetricsSplitter, self).__init__()
        self.test_tb = test_tb
        self.train_tb = train_tb

    def set_model(self, model):
        """
        Implement Callback.set_model
        
        :param model: 
        """
        if self.test_tb:
            self.test_tb.set_model(model)
        if self.train_tb:
            self.train_tb.set_model(model)

    def isTestMetric(self,metricName):
        """
        Determine whether a keras metric is computed on test/validation or training data
        
        At the moment simply look for the val prefix
        
        :param metricName: name of the metric
        :return: True iff the metric is determined to be computed on test/validation data
        """
        return metricName.find("val")==0 # metrics starting with val are computed on validation/test data

    def on_epoch_end(self, epoch, logs=None):
        """
        Implement Callback.set_model

        :param model: 
        """
        logs = logs or {}
        train_logs = {}
        test_logs = {}
        # divide metrics up into test and train and...
        for metric in logs.keys():
            if self.isTestMetric(metric):
                test_logs[metric] = logs[metric]
            else:
                train_logs[metric] = logs[metric]
        # ... route to the appropriate TensorBoard instance
        if self.test_tb:
            self.test_tb.on_epoch_end(epoch,test_logs)
        if self.train_tb:
            self.train_tb.on_epoch_end(epoch,train_logs)


    def on_train_end(self, x):
        if self.test_tb:
            self.test_tb.on_train_end(x)
        if self.train_tb:
            self.train_tb.on_train_end(x)
