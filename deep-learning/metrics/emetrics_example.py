from emetrics import EMetrics

with EMetrics.open() as metrics:
    metrics.record(EMetrics.TEST_GROUP, 1, {"accuracy": 0.6})  # record TEST metric accuracy=0.6 after step 1
    metrics.record(EMetrics.TRAIN_GROUP, 1, {"accuracy": 0.67})  # record TRAIN metric accuracy=0.6 after step 1
    
    metrics.record(EMetrics.TEST_GROUP, 2, {"accuracy": 0.5})  # record TEST metric accuracy=0.5 after step 2
    metrics.record(EMetrics.TRAIN_GROUP, 2, {"accuracy": 0.54})  # record TRAIN metric accuracy=0.6 after step 1

    metrics.record(EMetrics.TEST_GROUP, 3, {"accuracy": 0.9})  # record TEST metric accuracy=0.9 after step 3
    metrics.record(EMetrics.TRAIN_GROUP, 1, {"accuracy": 0.91})  # record TRAIN metric accuracy=0.6 after step 1