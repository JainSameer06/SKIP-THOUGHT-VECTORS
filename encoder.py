from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU
from keras.optimizers import Adam

#set num_encoder_tokens and num_decoder_tokens equal to the corpus vocabulary

encoder_inputs = Input(shape=(None,))
x = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
encoder = GRU(latent_dim, return_state=True, recurrent_initializer='orthogonal')
encoder_outputs, state_h = encoder(x)

decoder_inputs = Input(shape=(None,))
x = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
decoder_gru = GRU(latent_dim, return_sequences=True)(x, initial_state=state_h)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_gru)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


adam = Adam(lr=0.01, decay=1e-6,beta_1 =0.9 , beta_2 = 0.999, clipnorm=10)
model.compile(loss='mean_squared_error', optimizer=adam)

