# Sentiment Analysis using BiLSTM model 

## What is a biLSTM model

Bidirectional LSTMs in short BiLSTM are an addition to regular LSTMs which are used to enhance the performance of the model on sequence classification problems. BiLSTMs use two LSTMs to train on sequential input. The first LSTM is used on the input sequence as it is. The second LSTM is used on a reversed representation of the input sequence. It helps in supplementing additional context and makes our model fast.

![](/images/biLSTM_model.png)
## Datasets used

The dataset we have used is ISEAR (The International Survey on Emotion Antecedents and Reactions).
ISEAR dataset contains 7652 sentences. It has a total of seven sentiments - Joy, Fear, Anger, Sadness, Guilt, Shame, and Disgust.  

## Steps for making the model and then used for prediction

### Step 1: Importing the required libraries

import keras
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,Bidirectional
from nltk.tokenize import word_tokenize,sent_tokenize
from keras.layers import *
from sklearn.model_selection import cross_val_score 
import nltk
import pandas as pd
nltk.download('punkt')

### Step 2: Load the dataset

The next step is to load the dataset from our machine and preprocess it. In the dataset, there are some rows that contain -‘No response’. This sentence is completely useless for us. So, we will drop such rows.

 
df = pd.read_csv('isear.csv',header = None)

//The isear.csv contains rows with value 'No response'
//We need to remove such rows

df.drop(df[df[1] == '[ No response.]'].index, inplace = True)
print(df.head())

### Step 3: Tokeization

Apply a word tokenizer to convert each sentence into a list of words. Example: If there is a sentence- ‘I am happy’. Afterward tokenizing it will get converted into a list [‘I’,’am’, ‘happy’].

//The feel_arr will store all the sentences
//i.e feel_arr is the list of all sentences

feel_arr = df[1]
 
//Each  sentence in feel_arr is tokenized by the help of work tokenizer.
//If I have a sentence - 'I am happy'. 
//After word tokenizing it will convert into- ['I','am','happy']

feel_arr = [word_tokenize(sent) for sent in feel_arr]
print(feel_arr[0])

Output:

['[', 'On', 'days', 'when', 'I', 'feel', 'close', 'to', 'my', 'partner',
 'and', 'other', 'friends', '.', 'When', 'I', 'feel', 'at', 'peace', 'with',
 'myself', 'and', 'also', 'experience', 'a', 'close', 'contact', 'with', 
 'people', 'whom', 'I', 'regard', 'greatly', '.', ']']

### Step 4: Padding

The length of each sentence is different. To pass it through the model, the length of each sentence should be equal. By visualizing the dataset, we can see that the length of the sentence in the dataset is not greater than 100 words. So, now we will convert every sentence to 100 words. For this, we will take the help of padding.


//Defined a function padd in which each sentence length
//is fixed to 100.
//If length is less than 100 , then 
//the word- '<padd>' is append

def padd(arr):
    for i in range(100-len(arr)):
        arr.append('<pad>')
    return arr[:100]
   
//call the padd function for each sentence in feel_arr

for i in range(len(feel_arr)):
    feel_arr[i]=padd(feel_arr[i])
 print(feel_arr[0])

Output:

['[', 'On', 'days', 'when', 'I', 'feel', 'close', 'to', 'my', 'partner',
 'and', 'other', 'friends', '.', 'When', 'I', 'feel', 'at', 'peace', 
 'with', 'myself', 'and', 'also', 'experience', 'a', 'close', 'contact',
 'with', 'people', 'whom', 'I', 'regard', 'greatly', '.', ']', '<pad>', 
 '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
 '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
 '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
 '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
 '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>',
 '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 
 '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>',
 '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']

### Step 5: Word embedding using the glove 

Now, each word needs to be embedded in some numeric representation, as the model understands only numeric digits. So, for this, we have downloaded a predefined glove vector of 50 dimensions from the internet. This vector is used for the purpose of word embedding. Each word is represented in a vector of 50 dimensions. 

The glove vector contains almost all words in the English dictionary.

The first word of each row is the character that is to be embedded. And from the column to the last column, there is the numeric representation of that character in a 50d vector form. 

Create embeddings_index dictionary of words and their corresponding index

//Glove vector contains a 50 dimensional vector corresponding 
//to each word in dictionary.

vocab_f = 'glove.6B.50d.txt'
 
//embeddings_index is a dictionary which contains the mapping of word with its corresponding 50d vector.

embeddings_index = {}
with open(vocab_f, encoding='utf8') as f:
    for line in f:
        # splitting each line of the glove.6B.50d in a list of 
        # items- in which the first element is the word to be embedded,
        # and from second to the end of line contains the 50d vector.
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
         
//the embedding index of word 'happy'

print(embeddings_index['happy']) 

Output:

array([ 0.092086,  0.2571  , -0.58693 , -0.37029 ,  1.0828  , -0.55466 ,
       -0.78142 ,  0.58696 , -0.58714 ,  0.46318 , -0.11267 ,  0.2606  ,
       -0.26928 , -0.072466,  1.247   ,  0.30571 ,  0.56731 ,  0.30509 ,
       -0.050312, -0.64443 , -0.54513 ,  0.86429 ,  0.20914 ,  0.56334 ,
        1.1228  , -1.0516  , -0.78105 ,  0.29656 ,  0.7261  , -0.61392 ,
        2.4225  ,  1.0142  , -0.17753 ,  0.4147  , -0.12966 , -0.47064 ,
        0.3807  ,  0.16309 , -0.323   , -0.77899 , -0.42473 , -0.30826 ,
       -0.42242 ,  0.055069,  0.38267 ,  0.037415, -0.4302  , -0.39442 ,
        0.10511 ,  0.87286 ], dtype=float32)

Map each word of our dataset for their corresponding embedding vector
Now, each word of the dataset should be embedded in 50 dimensions vector with the help of the dictionary form above.

//Embedding each word of the feel_arr

embedded_feel_arr = []
for each_sentence in feel_arr:
    embedded_feel_arr.append([])
    for word in each_sentence:
        if word.lower() in embeddings_index:
            embedded_feel_arr[-1].append(embeddings_index[word.lower()])
        else:
            //if the word to be embedded is '<padd>' append 0 fifty times
            embedded_feel_arr[-1].append([0]*50)
             
print(embedded_feel_arr[0][0])

Output:

[-0.61201   0.98226   0.11539   0.014623  0.23873  -0.067035  0.30632
 -0.64742  -0.38517  -0.03691   0.094788  0.57631  -0.091557 -0.54825
  0.25255  -0.14759   0.13023   0.21658  -0.30623   0.30028  -0.23471
 -0.17927   0.9518    0.54258   0.31172  -0.51038  -0.65223  -0.48858
  0.13486  -0.40132   2.493    -0.38777  -0.26456  -0.49414  -0.3871
 -0.20983   0.82941  -0.46253   0.39549   0.014881  0.79485  -0.79958
 -0.16243   0.013862 -0.53536   0.52536   0.019818 -0.16353   0.30649
  0.81745 ]

Here, in the above example, the dictionary formed i.e embeddings_index contains the word and its corresponding 50d vector.

### Step 6: One Hot encoding for the target variables and split train and test dataset

Now, we are done with all the preprocessing parts, and now we need to perform the following operations:

1. Do one-hot encoding of each emotion.
2. Split the dataset into train and test sets.

//Converting x into numpy-array

X = np.array(embedded_feel_arr)
 
//Perform one-hot encoding on df[0] i.e emotion

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
Y = enc.fit_transform(np.array(df[0]).reshape(-1,1)).toarray()
 
//Split into train and test

from keras.layers import Embedding
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


### Step 7: Build the model

//Defining the BiLSTM Model
def model(X, Y, input_size1, input_size2, output_size):
    m = Sequential()
    m.add(Bidirectional(LSTM(100,input_shape=(input_size1,input_size2))))
    m.add(Dropout(0.5))
    m.add(Dense(output_size,activation='softmax'))
    m.compile('Adam','categorical_crossentropy',['accuracy'])
    m.fit(X,Y,epochs=32, batch_size=128)
    return m

### Step 7: Train the model

//calling the model function with the traning data inorder to build the model
bilstmModel = model(X_train, Y_train, 100, 50, 7)

//print the summary of the model
bilstmModel.summary()
![](/images/model_summary.png)

This is the diagram of the proposed model :

Here, the dimension of input is 100 X 50 where 100 is the number of words in each input sentence of the dataset and 50 represents the mapping of each word in a 50d vector.

The output of Bidirectional(LSTM) is 200 because above we have defined the dimensionality of output space to be 100. As it is a BiLSTM model, so dimensionality will be 100*2 =200, as a BiLSTM contains two LSTM layers- one forward and the other backward.

After this dropout layer is added to prevent overfitting. And at last dense layer is applied to convert the 200 output sequences to 7, as we have only 7 emotions, so the output should be of seven dimensions only.

### Step 8: Testing the model

bilstmModel.evaluate(X_test,Y_test)

![](/images/model_evaluation.png)

The accuracy of the model is 51%

### Step 9: Checking the results of the model

# Checking the results of our model with user given sentences stored in test.csv file

df_test = pd.read_csv('test.csv',header = None)
df_test.drop(df[df[1] == '[ No response.]'].index, inplace = True)
feel_arr_test=df_test[1]


feel_arr_test=[word_tokenize(sent) for sent in feel_arr_test]

for i in range(len(feel_arr1)):
    feel_arr_test[i]=padd(feel_arr_test[i])

embedded_feel_arr_test=[] 
for each_sentence in feel_arr_test:
    embedded_feel_arr_test.append([])
    for word in each_sentence:
        if word.lower() in embeddings_index:
            embedded_feel_arr_test[-1].append(embeddings_index[word.lower()])
        else:
            embedded_feel_arr_test[-1].append([0]*50)
            
X_test = np.array(embedded_feel_arr_test)
prediction = bilstmModel.predict(X_test)

// sentence: 'Grovelling people.' , Emotion: Digust.
// highest value of the prediction list is 0.22194855(at the last index 6) which is the index of disgust emotion

prediction[4]
![](/images/predicted_array_of_emotions.png)

Here we get our predicted emotion as disgust which is the correct result that we wanted stating that our model is working well for sentiment analysis.

