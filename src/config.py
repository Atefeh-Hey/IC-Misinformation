NE_EMB_SIZE_WITH_FEATURES = 777     #size of the NE embeddings after concatenating word level features.
NE_EMB_SIZE = 768                   #size of the NE embeddings without concatenating word level features.
DOC_EMB_SIZE = 2352                 #size of the whole article embeddings with concatenating doc level features.
NE_LVL_FEATURES = True              # indicates if we want to use ne level features or not
SEQUENCE_LENGTH = 500               # Max length of input sequence for each input article--bseline or word-level features
SEQUENCE_LENGTH_ALL = 1             # Max length of input sequence for each input article--model with all features
K_FOLD = 10
EPOCHS = 100                        # number of epochs