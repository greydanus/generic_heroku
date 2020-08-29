import torch
import torch.nn as nn
import numpy as np

import mobypy


def generate_word_embeddings(model, tokenizer, text):
	'''
	Given a BERT model and tokenizer, this function tokenizes the 'text' string
	and runs the tokenized text through the BERT model, which returns embeddings for each word
	as a tensor.

	Output is a list of tensors shaped [1 x num_bert_layers x ebedding_dim], aka [1 x 13 x 768],
	each tensor corresponds to a word in text.
	'''
	# Model will return different number of objects depending on how it's configured
	marked_text = "[CLS] " + text + " [SEP]"
	tokenized_text = tokenizer.tokenize(marked_text)
	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
	segments_ids = [1] * len(tokenized_text)
	tokens_tensor = torch.tensor([indexed_tokens])
	segments_tensors = torch.tensor([segments_ids])

	outputs = model(tokens_tensor, segments_tensors)

	return outputs[2]


def stack_embedding_list(hidden_states):
	'''
	Converts a list of word embeddings produced by BERT into a 
	stacked tensor that we can operate on
	'''
	token_embeddings = torch.stack(hidden_states, dim=0)
	token_embeddings = torch.squeeze(token_embeddings, dim=1)
	token_embeddings = token_embeddings.permute(1,0,2)
	return token_embeddings


def sum_layer_dimension(token_embeddings):
	'''
	Generates a global sentence embedding given a list of word 
	embeddings generated by BERT. 
	Output: Tensor of shape [num_words_in_list x embedding_dim]
	'''
	return torch.sum(token_embeddings[:, -4:, :], dim=1)


def dist(x, y):
	return nn.CosineSimilarity()(x,y).detach().item()


def get_ranked_synonyms(sentence_embedding, word_query):

	synonyms = mobypy.synonyms(word)
	scores = np.zeros((len(synonyms)))
	for i, synonym in enumerate(synonyms):
		candidate = sentence.replace(word, synonym)
		candidate_embedding = text_to_summed_embedding(candidate)
		scores[i] = dist(sentence_embedding, candidate_embedding)

	ranking = np.argsort(scores)[::-1]
	ranked_scores = scores[ranking]
	ranked_synonyms = [synonyms[i] for i in ranking]

	return ranked_synonyms, ranked_scores
