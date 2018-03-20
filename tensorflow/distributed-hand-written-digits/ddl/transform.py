import tensorflow as tf
import os

modelpath = os.getenv("RESULT_DIR")+"/model"
chkptpath = os.getenv("RESULT_DIR")+"/mnist"
chkptnumber = "-20"

################################################################################
# Loads the base graph images->logits with weights from a checkpoint
################################################################################
def load_basegraph(restorepath):
    rname = restorepath+"_basegraph.meta"
    print(" Loading graph from "+rname)
    restorer = tf.train.import_meta_graph(rname)
    graph = tf.get_default_graph()
    # get access to the placeholder in graph
    pImgs = graph.get_tensor_by_name("x:0")
    pRes  = graph.get_tensor_by_name("pRes:0")                              
    kp = graph.get_tensor_by_name("keepprob:0");
    return restorer,pImgs,pRes,kp


################################################################################
def chkpt2model(graphloadpath,weightloadpath,modelwritepath):
    
    print("Running the program to transform meta graph and checkpoint in servable model")
    restorer,pImgs,pRes,kp = load_basegraph(graphloadpath)
    values, indices = tf.nn.top_k(pRes, 10)                                     
    # for modelbuilder 
    prd = tf.argmax(pRes, 1, name="predictor") 

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        restorer.restore(sess, weightloadpath)

        sess.graph._unsafe_unfinalize()
        export_path = modelwritepath
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        classification_inputs = tf.saved_model.utils.build_tensor_info(pImgs)
        classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prd)

        classification_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={tf.saved_model.signature_constants.CLASSIFY_INPUTS: classification_inputs},
            outputs={
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: 
                                               classification_outputs_classes
            },
            method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)

        tensor_info_x = tf.saved_model.utils.build_tensor_info(pImgs)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(pRes)
        tensor_info_c = tf.saved_model.utils.build_tensor_info(prd)
        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_x},
            outputs={'class_index': tensor_info_c},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images': prediction_signature,
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature,
            },
            legacy_init_op=legacy_init_op,
            clear_devices=True)
        
        save_path = str(builder.save())
        print("Model for serving is saved here: %s" % save_path)


chkpt2model(chkptpath,chkptpath+chkptnumber,modelpath)

