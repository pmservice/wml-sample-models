# encoding=utf-8
import json
import time
import os

class EMetrics(object):
    """
    Manage the logging of metrics on behalf of a python client training with a deep learning framework
    
    Metrics recorded be passed to Watson Machine Learning and made available to WML clients
    
    This will also output TEST_GROUP metrics to file val_dict_list.json to be read by the Hyper-Parameter-Optimization (HPO) algorithm
    
    Example Usage:
    
    from emetrics import EMetrics
    
    with EMetrics.open("1") as metrics:
        metrics.record(EMetrics.TEST_GROUP, 1, {"accuracy": 0.6}) # record TEST metric accuracy=0.6 after step 1
        metrics.record(EMetrics.TEST_GROUP, 2, {"accuracy": 0.5}) # record TEST metric accuracy=0.5 after step 2
        metrics.record(EMetrics.TEST_GROUP, 3, {"accuracy": 0.9}) # record TEST metric accuracy=0.9 after step 3
    """

    TEST_GROUP = "test"   # standard group name for metrics collected on test dataset (also referred to as holdout or validation dataset)
    TRAIN_GROUP = "train" # standard group name for metrics collected on training dataset

    def __init__(self,subId,f):
        if "TRAINING_ID" in os.environ:
            self.trainingId = os.environ["TRAINING_ID"]
        else:
            self.trainingId = ""
        self.rIndex = 1
        self.subId = subId
        self.f = f
        self.test_history = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    @staticmethod
    def open(subId=None):
        """
        Open and return an EMetrics object
        
        :param subId: optional, the string identifier of an HPO sub-execution (only used in HPO, caller can get the subId from the SUBID environment variable) 
        :return: EMetrics object
        """
        if "LOG_DIR" in os.environ:
            folder = os.environ["LOG_DIR"]
        elif "JOB_STATE_DIR" in os.environ:
            folder = os.path.join(os.environ["JOB_STATE_DIR"],"logs")
        else:
            folder = "/tmp"

        if subId is not None:
            folder = os.path.join(folder, subId)

        if not os.path.exists(folder):
            os.makedirs(folder)

        f = open(os.path.join(folder, "evaluation-metrics.txt"), "a")
        return EMetrics(subId,f)

    def __encode(self,value):
        if isinstance(value,int):
            return { "type":2, "value": str(value) }
        if isinstance(value,float):
            return {"type": 3, "value": str(value) }
        return { "value": str(value) }

    def record(self,group,iteration,values):
        """
        Record a set of metrics for a particular group and iteration
        
        :param group: a string identifying how the metrics were computed.  Use EMetrics.TEST_GROUP for validation/test data metrics.
        :param iteration: an integer indicating the iteration/step/epoch at which the metrics were computed
        :param values: a dict containing one or more named metrics (values may be string, float or integer)
        """
        if group == EMetrics.TEST_GROUP and self.subId:
            d = {"steps": iteration}
            d.update(values)
            self.test_history.append(d)

        obj = {
            "meta": {
                "training_id":self.trainingId,
                "time": int(time.time()*1000),
                "rindex": self.rIndex
            },
            "grouplabel":group,
            "etimes": {
                "iteration":self.__encode(iteration),
                "time_stamp":self.__encode(time.strftime("%Y-%m-%dT%H:%M:%S.%s"))
            },
            "values": { k:self.__encode(v) for k,v in values.items() }
        }

        if self.subId:
            obj["meta"]["subid"] = str(self.subId)

        if self.f:
            self.f.write(json.dumps(obj) + "\n")
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()
        if "RESULT_DIR" in os.environ:
            folder = os.environ["RESULT_DIR"]  # should use LOG_DIR?
        else:
            folder = "/tmp"
        if self.subId:
            open(os.path.join(folder,"val_dict_list.json"),"w").write(json.dumps(self.test_history))


