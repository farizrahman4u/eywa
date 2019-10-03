# -*- coding: utf-8 -*-
'''
General documentation architecture:

Home
Index

- Getting started
    Getting started with the eywa Classifier
    Getting started with the EntityExtractor
    

- Models
    About Keras models
        explain when one should use Sequential or functional API
        explain compilation step
        explain weight saving, weight loading
        explain serialization, deserialization
    Sequential
    Model (functional API)

- Layers
    About Keras layers
        explain common layer functions: get_weights, set_weights, get_config
        explain input_shape
        explain usage on non-Keras tensors
    Core Layers
    Convolutional Layers
    Pooling Layers
    Locally-connected Layers
    Recurrent Layers
    Embedding Layers
    Merge Layers
    Advanced Activations Layers
    Normalization Layers
    Noise Layers
    Layer Wrappers
    Writing your own Keras layers

- Preprocessing
    Sequence Preprocessing
    Text Preprocessing
    Image Preprocessing

Losses
Metrics
Optimizers
Activations
Callbacks
Datasets
Applications
Backend
Initializers
Regularizers
Constraints
Visualization
Scikit-learn API
Utils
Contributing

'''
from eywa.nlu import classifier
from eywa.nlu import entity_extractor
classifier_object=classifier.Classifier
entity_extractor_object=entity_extractor.EntityExtractor



EXCLUDE = {
    'switch',
    'blame',
    'Wrapper',
    'forward',
    'default_weights',
    'entities',
}

# For each class to document, it is possible to:
# 1) Document only the class: [classA, classB, ...]
# 2) Document all its methods: [classA, (classB, "*")]
# 3) Choose which methods to document (methods listed as strings):
# [classA, (classB, ["method1", "method2", ...]), ...]
# 4) Choose which methods to document (methods listed as qualified names):
# [classA, (classB, [module.classB.method1, module.classB.method2, ...]), ...]
PAGES = [
    {
        'page': 'models/Classifer.md',
        
        'classes': [
           classifier_object,
        ],
        'methods': [
            classifier_object.fit,
            classifier_object.predict,
            classifier_object.evaluate,
            classifier_object.serialize,
            classifier_object.deserialize,
            classifier_object.set_weights,
            classifier_object.get_weights,
        ],
       
    },
       {
        'page': 'models/EntityExtractor.md',
        
                'classes': [
                         entity_extractor_object,
           
        ],
        'methods': [
            entity_extractor_object.fit,
            entity_extractor_object.predict,
            entity_extractor_object.evaluate,
            entity_extractor_object.serialize,
            entity_extractor_object.deserialize,
            entity_extractor_object.set_weights,
            entity_extractor_object.get_weights,
        ],
         
    },

]

ROOT = 'http://keras.io/'


