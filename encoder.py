from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU
from keras.optimizers import Adam


encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = GRU(latent_dim, return_state=True, recurrent_initializer='orthogonal')
encoder_outputs, state_h = encoder(encoder_inputs)

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_gru = GRU(latent_dim, return_sequences=True)
decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


adam = Adam(lr=0.01, decay=1e-6,beta_1 =0.9 , beta_2 = 0.999, clipnorm=10)
model.compile(loss='mean_squared_error', optimizer=adam)

