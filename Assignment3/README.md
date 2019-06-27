数据处理3个阶段 原始-> UTM -> Normalization

> A slightly better method is to use a sequence autoencoder, which uses a RNN to read a long input sequence into a single vector. This vector will then be used to reconstruct the original sequence.

So the example reads everything into a single vector, then uses that vector to reconstruct the original sequence. If you want to iteratively generate something but you only have one input, you can repeat the vector. That means each time step will get the same input but a different hidden state.