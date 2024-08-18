This repository is for the internal consistency detection model using CNN and text-based features. Instructions on how to use different components and build models are provided in the thesis.
Due to their sizes, the following deployed encoding models are not included in the repository. Please download them and add them to the 'data' folder. These files could be downloaded from reliable sources and are publicly available.
1. GoogleNews-vectors-negative300.bin  (source: https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300)
2. glove.6B.100d.txt                   (source: https://www.kaggle.com/datasets/sawarn69/glove6b100dtxt)

Dataset:
Gold-standard dataset with consistency and veracity labels is provided in the 'data' folder, data.csv
Consistency labels are in the 'label' column. 0 shows an IC and 1 shows an NIC article.
Veracity labels are in the 'label_fk' column. 0 shows real and 1 shows fake article.
'Cleaned' is the content of news articles to be used for this experiment.

For more information, please get in touch with a.hey.ir@gmail.com
