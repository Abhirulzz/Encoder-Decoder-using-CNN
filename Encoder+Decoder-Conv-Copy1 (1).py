
# coding: utf-8

# In[1]:


from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'fra-eng/fra.txt'


# In[2]:


# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


# In[3]:


input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])


# In[4]:


encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


# In[ ]:


encoder_input_data.shape


# In[ ]:


num_decoder_tokens


# In[5]:


#Encoder 
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
maxlen = 16
vocab_size = 70
nb_filter = 256
filter_kernels = [2, 2, 3, 3, 3, 3]
inputs = Input(shape=(maxlen, vocab_size), name='input_Encoder', dtype='float32')

conv = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[2],
                     border_mode='valid', activation='relu',
                     input_shape=(maxlen, vocab_size))
l1out = conv(inputs)
conv1 = Convolution1D(nb_filter=nb_filter, filter_length=2, dilation_rate=3,
                      border_mode='valid', activation='relu')
l2out = conv1(l1out)
conv3 = MaxPooling1D(pool_length=2)
l3out = conv3(l2out)


# In[6]:


#Decoder
de_maxlen = 59
de_vocab_size = 93
de_nb_filter = 256
deinputs = Input(shape=(None, de_vocab_size), name='input_decoder', dtype='float32')
deconv = Convolution1D(nb_filter=de_nb_filter, filter_length=filter_kernels[2],
                     border_mode='valid', activation='relu',
                     input_shape=(de_maxlen, de_vocab_size))
dl1out = deconv(deinputs)
deconv1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[0], dilation_rate=3,
                      border_mode='valid', activation='relu')
dl2out = deconv1(dl1out)
deconv3 = MaxPooling1D(pool_length=1)
dl3out = deconv3(dl2out)
from keras.layers import concatenate
merged = concatenate([l3out, dl3out], axis=1)
z = Dense(num_decoder_tokens, activation='relu')
dl4out = z(merged)
#z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))
pred = Dense(num_decoder_tokens, activation='softmax', name='output')
output = pred(dl4out)


# In[ ]:


deinputs


# In[7]:


model = Model([inputs, deinputs], output)


# In[8]:


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[9]:


#Fitting the model
history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=100,
          epochs=5,
          validation_split=0.2)


# In[10]:


encoder_model = Model(inputs, l3out)


# In[11]:


encoder_model.summary()


# In[12]:


conv3_input_inf = Input(shape=(5, 256), name='input_decoder2', dtype='float32')


# In[13]:


o1 = deconv(deinputs)
o2 = deconv1(o1)
o3 = deconv3(o2)
o4 = concatenate([conv3_input_inf,o3],axis=1)
o5 = z(o4)
o6 = pred(o5)


# In[16]:


decoder_model = Model([deinputs,conv3_input_inf], [o6])


# In[17]:


decoder_model.summary()


# In[18]:


# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


# In[25]:


encoder_model.predict(input_seq).shape


# In[26]:


states_value = encoder_model.predict(input_seq)


# In[23]:


# Generate empty target sequence of length 1.
target_seq = np.zeros((1, 1, num_decoder_tokens))
# Populate the first character of target sequence with the start character.
target_seq[0, 0, target_token_index['\t']] = 1.


# In[24]:


stop_condition = False
decoded_sentence = ''


# In[37]:


#Decoder
de_maxlen = 59
de_vocab_size = 93
de_nb_filter = 256
deinputs_x = Input(shape=(1, de_vocab_size), name='input_decoder_x', dtype='float32')
deconv_x = Convolution1D(nb_filter=de_nb_filter, filter_length=1,
                     border_mode='valid', activation='relu',
                     input_shape=(de_maxlen, de_vocab_size))
dl1out_x = deconv_x(deinputs_x)
#deconv1_x = Convolution1D(nb_filter=nb_filter, filter_length=1, dilation_rate=3, border_mode='valid', activation='relu')
#dl2out_x = deconv1_x(dl1out_x)
deconv3_x = MaxPooling1D(pool_length=1)
dl3out_x = deconv3_x(dl1out_x)


# In[ ]:


from keras.layers import concatenate


# In[39]:


merged_x = concatenate([conv3_input_inf, dl3out_x], axis=1)


# In[40]:


z_x = Dense(num_decoder_tokens, activation='relu')


# In[42]:


dl4out_x = z_x(merged_x)


# In[44]:


deconv5_x = MaxPooling1D(pool_length=4)
dl5out_x = deconv5_x(dl4out_x)


# In[45]:


dl5out_x


# In[48]:


#z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))
pred_x = Dense(num_decoder_tokens, activation='softmax', name='output')
output_x = pred_x(dl5out_x)


# In[49]:


output_x


# In[50]:


decoder_model = Model([deinputs_x,conv3_input_inf], [output_x])


# In[51]:


output_tokens = decoder_model.predict(
            [target_seq,states_value])


# In[52]:


output_tokens


# In[53]:


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens = decoder_model.predict(
            [target_seq,states_value])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        #states_value = [h, c]

    return decoded_sentence


# In[54]:


for seq_index in range(10):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)


# In[20]:


input_seq = encoder_input_data[0: 0 + 1]


# In[21]:


input_seq.shape


# In[ ]:


decode_sequence(input_seq)


# In[ ]:


# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)


# In[ ]:


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)


# In[ ]:


encoder_input_data[0:1].shape

