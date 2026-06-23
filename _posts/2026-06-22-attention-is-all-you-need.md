---
title: 'Attention Is All You Need: A Practical Introduction to the Transformer'
date: 2026-06-22
permalink: /posts/2026/06/attention-is-all-you-need-practical-intro/
tags:
  - deep-learning
  - nlp
  - transformers
---

Attention Is All You Need: A Practical Introduction to the Transformer

An explainer for readers who know their way around CNNs and RNNs but haven't yet sat down with "Attention Is All You Need."

### Why Attention Was Introduced
Before the Transformer, the standard way to handle sequences — translation, language modeling, speech — was an encoder-decoder architecture built on recurrent networks (RNNs, LSTMs, GRUs). The encoder reads the input sequence one token at a time and compresses everything it has seen into a hidden state; the decoder then unpacks that hidden state, again one step at a time, to produce the output sequence.
The problem with this setup is the hidden state itself: by the time the encoder reaches the end of a long sentence, it has to represent everything relevant about that sentence in a single fixed-size vector. Early information tends to get diluted or overwritten by everything that comes after it — an information bottleneck. Attention was introduced specifically to fix this: rather than forcing the decoder to work from one compressed summary, it lets the decoder look back at every encoder hidden state and decide, at each output step, which parts of the input are most relevant right now. Attention mechanisms had already been bolted onto RNN encoder-decoders for a couple of years before 2017, and they worked well — but they were still wrapped around a recurrent backbone. The contribution of "Attention Is All You Need" is the claim in the title: you don't need the recurrence at all. Attention alone, with no recurrent or convolutional layers, is enough to build a state-of-the-art sequence model — and removing the recurrence turns out to unlock a lot more than just simplicity.

### The Limitations of RNNs
To see why dropping recurrence was worth doing, it's worth being specific about what recurrence costs you:

* **Sequential computation.** An RNN computes hidden state h_t from h_{t-1}, which means step t cannot start until step t-1 finishes. This is true at both training and inference time. No matter how much parallel hardware you have, you cannot parallelize the computation within one sequence — you can only parallelize across different sequences in a batch. For long sequences, this sequential chain becomes the dominant cost.
* **Long path lengths for long-range dependencies.** If token 1 and token 50 need to influence each other, the signal has to travel through 49 recurrent steps in both the forward and backward pass. The number of operations a signal needs to traverse between any two positions grows with the distance between them. The longer that path, the harder it is for the network to learn the dependency — gradients have more opportunities to shrink or distort along the way, and gating mechanisms in LSTMs/GRUs only partially relieve this.
* **The fixed-size bottleneck.** Even with attention layered on top of an RNN, the recurrent encoder itself is still generating its hidden states one after another, so the sequential bottleneck doesn't disappear — it just gets a partial workaround.

These three issues point at the same conclusion: what you really want is a mechanism where any two positions in a sequence can interact directly, in a constant number of operations, regardless of how far apart they are — and where the computation for different positions doesn't have to wait on each other. That's exactly what self-attention provides.

### Query, Key, and Value Vectors
The core operation inside a Transformer is built around three vectors derived from each token: a Query (Q), a Key (K), and a Value (V). The easiest way to build intuition is a lookup analogy. Imagine a system where every item in storage has a key describing what it is, and a value holding its actual content. When you want information, you issue a query describing what you're looking for. The query is compared against every key to produce a relevance score, and the output you get back is a weighted blend of the values, weighted by how relevant each key was to your query.
In a Transformer, every token in the sequence plays all three roles simultaneously: it produces a query (what is this token looking for in the rest of the sequence?), a key (how would this token be described, so other tokens can decide whether to attend to it?), and a value (what content does this token actually contribute once attended to?). Q, K, and V are all obtained by multiplying the token's embedding by three separate learned weight matrices — so the model learns, during training, what makes a good query, key, and value for the task at hand.

### Scaled Dot-Product Attention
With Q, K, and V in hand, the attention computation itself is a single formula:

Attention(Q, K, V) = softmax( QKᵗ / √d_k ) V

Step by step: QKᵗ computes a dot product between every query and every key — a similarity score for every pair of positions in the sequence. A large dot product means the query and key point in similar directions, i.e., high relevance. Softmax then turns each row of these scores into a proper probability distribution (the attention weights), and the result is used to take a weighted sum over the value vectors — so each token's output is a blend of every other token's value, weighted by how relevant that token's key was to the current query.
The one detail that looks arbitrary at first is the division by √d_k, where d_k is the dimensionality of the key vectors. It isn't arbitrary. If the components of q and k are independent random values with mean 0 and variance 1, their dot product has variance equal to d_k itself — so as the key dimension grows, the dot products grow large in magnitude purely as a side effect of higher dimensionality, not because the vectors are actually more "relevant" to each other. Large-magnitude inputs push softmax toward an extremely sharp, near one-hot distribution, and in that saturated regime the gradient of softmax is close to zero almost everywhere. That kills gradient flow through the attention layer. Dividing by √d_k rescales the dot products back down to roughly unit variance before the softmax, keeping the function in a regime where it actually has gradients to propagate. This is precisely why the mechanism is called scaled dot-product attention rather than just dot-product attention.

### Multi-Head Attention
A single attention computation forces the model to settle on one blend of relevance for every token — it has to average together whatever different relationships might exist (e.g. "which word is the subject of this verb" and "which word does this pronoun refer to") into one set of attention weights. With only one head, that averaging genuinely loses information that the model could otherwise keep separate.
Multi-head attention sidesteps this by running several scaled dot-product attention computations in parallel, each operating in its own lower-dimensional projection of Q, K, and V:

MultiHead(Q, K, V) = Concat(head₁, ..., head_h) Wᴼ, where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

Each head has its own learned projection matrices, so each one can specialize — one head might end up tracking nearby-word relationships, another might track long-range syntactic dependencies, another semantic similarity, and so on, all simultaneously, all computed in parallel. The outputs of all heads are concatenated and passed through one more learned projection (Wᴼ) to produce the final result. The original paper used 8 heads, each with key/value dimension d_k = d_v = d_model / h = 64 (with d_model = 512); because each head works in a reduced dimension, the total computational cost of multi-head attention ends up comparable to a single full-dimensional attention head — you get the representational benefit of multiple subspaces essentially for free.

### The Transformer Architecture
Multi-head attention is the centerpiece, but the full Transformer wraps it in a fairly specific structure:

* **Encoder-decoder stacks.** The encoder is a stack of N=6 identical layers; the decoder is also a stack of N=6 identical layers. Each encoder layer has two sub-layers: multi-head self-attention (every position attends to every other position in the input), followed by a position-wise feed-forward network applied independently to each position. Each decoder layer has three sub-layers: a masked multi-head self-attention layer, an encoder-decoder attention layer (queries come from the decoder, keys and values come from the encoder's output — this is what lets the decoder look back at the input), and a position-wise feed-forward network.
* **Masking in the decoder.** Because the decoder generates output one token at a time and shouldn't be able to peek at tokens it hasn't produced yet, its self-attention is masked so that position i can only attend to positions ≤ i.
* **Residual connections and layer normalization.** Every sub-layer's output is computed as LayerNorm(x + Sublayer(x)) — the same residual idea from ResNet, applied here to make the much deeper attention stack easier to optimize.
* **Position-wise feed-forward networks.** A simple two-layer fully connected network (with a ReLU in between) applied identically and independently to every position, giving each token's representation some additional nonlinear processing after the attention step mixes information across positions.
* **Positional encoding.** Since there's no recurrence and no convolution, the model has no inherent sense of token order — attention treats the sequence as an unordered set unless told otherwise. The fix is to add a positional encoding vector to each token's embedding before the first layer, using sine and cosine functions at different frequencies for each embedding dimension. This was chosen, in part, because it lets the model learn to attend by relative position fairly easily, since the encoding for any fixed offset is a simple linear function of the encoding at the current position.

### Why Transformers Became Dominant
Pulling all of this together, the case for the Transformer comes down to three properties that the paper measures directly against recurrent and convolutional alternatives:

* **Parallelization.** Self-attention has no sequential dependency between positions — every position's attention output can be computed simultaneously, given the inputs. That removes the core bottleneck that made RNNs slow to train on long sequences, and lets the Transformer make full use of parallel hardware like GPUs/TPUs during training.
* **Constant path length for long-range dependencies.** In self-attention, any two positions are connected by exactly one operation — a single dot product — regardless of how far apart they are in the sequence. Compare this to an RNN, where the path length between distant positions grows linearly with the distance. Shorter paths between positions make it easier for the network to learn long-range dependencies, since there's less room for signal to degrade along the way.
* **Better results at lower training cost.** The paper's own results back this up empirically: their Transformer models reached state-of-the-art translation quality on WMT 2014 English-to-German and English-to-French benchmarks, while requiring significantly less training time than the best previous recurrent and convolutional models.

Beyond what the paper itself reports, these same properties are why the architecture proved so easy to scale up afterward. A model whose core computation parallelizes cleanly across positions — and whose building blocks (self-attention, feed-forward layers, residual connections) don't depend on any particular modality — turns out to be a very convenient thing to make bigger, train on much larger datasets, and reuse across tasks. The encoder-decoder split itself also turned out to be more modular than it first appears: later work peeled the encoder and decoder stacks apart to build encoder-only and decoder-only models suited to different jobs. None of that would have been nearly as practical with an architecture whose fundamental computation couldn't be parallelized in the first place — which is the real significance of replacing recurrence with attention.