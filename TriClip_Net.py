# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
import clip
import random
# from clip import clip
from torch.nn import init as init
from functools import partial
from typing import Optional, Callable
from einops import rearrange, repeat
from mamba_ssm import Mamba
from timm.layers import DropPath, to_2tuple, trunc_normal_
from collections import OrderedDict
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
softmax = nn.Softmax(dim=1)

try:
	from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
	pass

_tokenizer = _Tokenizer()


def cal_similarity(image, text, sigma=10):
	# b, d = image.shape
	x_1 = torch.unsqueeze(image, 1)
	x_2 = torch.unsqueeze(text, 2)
	distance = torch.norm(x_1 - x_2, dim=0)
	similarity = torch.exp(-distance ** 2 / (2 * (sigma ** 2)))

	return similarity


class CNN_Encoder(nn.Module):
	def __init__(self, image_size, band1, band2, band3, embed_dim):
		super(CNN_Encoder, self).__init__()
		self.conv11 = nn.Sequential(
			nn.Conv2d(band1, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			# nn.Dropout(0.5),
		)
		self.conv12 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			# nn.Dropout(0.5),
		)
		self.conv13 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			nn.Dropout(0.5),
		)

		self.conv21 = nn.Sequential(
			nn.Conv2d(band2, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			# nn.Dropout(0.5),
		)
		self.conv22 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			# nn.Dropout(0.5),
		)
		self.conv23 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			nn.Dropout(0.5),
		)

		self.conv31 = nn.Sequential(
			nn.Conv2d(band3, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			# nn.Dropout(0.5),
		)
		self.conv32 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			# nn.Dropout(0.5),
		)
		self.conv33 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			nn.Dropout(0.5),
		)

		self.gap = nn.AdaptiveAvgPool2d(1)
		self.fc1 = nn.Conv2d(64, embed_dim, 1)
		self.fc2 = nn.Conv2d(64, embed_dim, 1)
		self.fc3 = nn.Conv2d(64, embed_dim, 1)

	def forward(self, x1, x2, x3):
		x1 = self.conv11(x1)
		x2 = self.conv21(x2)
		x3 = self.conv31(x3)

		x1 = self.conv12(x1)
		x2 = self.conv22(x2)
		x3 = self.conv32(x3)

		x1 = self.conv13(x1)
		x2 = self.conv23(x2)
		x3 = self.conv33(x3)

		x11, x22, x33 = self.gap(x1), self.gap(x2), self.gap(x3)
		x11 = self.fc1(x11).squeeze()
		x22 = self.fc2(x22).squeeze()
		x33 = self.fc3(x33).squeeze()

		return x1, x2, x3, x11, x22, x33


class LayerNorm(nn.LayerNorm):
	"""Subclass torch's LayerNorm to handle fp16."""

	def forward(self, x: torch.Tensor):
		orig_type = x.dtype
		ret = super().forward(x.type(torch.float32))
		return ret.type(orig_type)


class Text_Encoder(nn.Module):
	def __init__(self, embed_dim, context_length, vocab_size, transformer_width):
		super().__init__()
		self.context_length = context_length
		self.transformer_width = transformer_width

		self.mamba1 = Mamba(
			d_model=transformer_width,  # Model dimension d_model
			d_state=16,  # SSM state expansion factor
			d_conv=4,  # Local convolution width
			expand=2  # Block expansion factor
		)

		self.vocab_size = vocab_size
		self.token_embedding = nn.Embedding(vocab_size, transformer_width)
		self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width, requires_grad=True))
		self.ln_final = LayerNorm(transformer_width)

		self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim, requires_grad=True))

		self.initialize_parameters()
		self.dtype = torch.float32

	def initialize_parameters(self):
		nn.init.normal_(self.positional_embedding, std=0.01)

		if self.text_projection is not None:
			nn.init.normal_(self.text_projection, std=self.transformer_width ** -0.5)

	def forward(self, text):
		x = self.token_embedding(text).type(self.dtype)

		x = x + self.positional_embedding.type(self.dtype)
		x = self.mamba1(x)
		x = self.ln_final(x).type(self.dtype)

		x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

		return x


class Text_Decoder(nn.Module):
	def __init__(self, embed_dim=128, context_length=77, vocab_size=49408, transformer_width=64):
		super().__init__()
		self.context_length = context_length
		self.transformer_width = transformer_width
		self.decoder = nn.Linear(embed_dim, context_length * transformer_width)

		self.mamba1 = Mamba(
			d_model=transformer_width,  # Model dimension d_model
			d_state=16,  # SSM state expansion factor
			d_conv=4,  # Local convolution width
			expand=2  # Block expansion factor
		)

	def forward(self, encoded_text):
		x = self.decoder(encoded_text)
		x = x.view(x.size(0), self.context_length, self.transformer_width)
		x = self.mamba1(x)

		return x


class CNN_Classifier(nn.Module):
	def __init__(self, Classes):
		super(CNN_Classifier, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(64, 32, 1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(),
			nn.AdaptiveAvgPool2d(1),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(32, Classes, 1),
		)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)

		return x

class GCN_Fusion(nn.Module):
	def __init__(self, in_channel):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channel, in_channel, 1, padding=0)
		self.conv2 = nn.Conv2d(in_channel, in_channel, 1, padding=0)
		self.conv3 = nn.Conv2d(in_channel, in_channel, 1, padding=0)

	def knn_similarity(self, x, y, method, eta=1.0, alpha=0.08, beta=0.01, tau=10, prob_threshold=0.8):
		b, n, feature_dim = x.shape

		if method == "euclidean":
			x_expanded = x.unsqueeze(1)
			y_expanded = y.unsqueeze(2)
			D = torch.norm(x_expanded - y_expanded, p=2, dim=3)

		elif method == "cosine":
			x_norm = F.normalize(x, p=2, dim=2)
			y_norm = F.normalize(y, p=2, dim=2)
			similarity = torch.bmm(x_norm, y_norm.transpose(1, 2))
			D = 1 - similarity

		D_max = D.max(dim=2, keepdim=True).values  # (b, n, 1)
		D_normalized = D / D_max  # (b, n, n)

		mu = D.mean(dim=(1, 2))  # (b,)
		sigma = D.std(dim=(1, 2))  # (b,)

		prob_D = torch.softmax(-D / tau, dim=2)  # (b, n, n)
		entropy = -torch.sum(prob_D * torch.log(prob_D + 1e-8), dim=2)  # (b, n)
		entropy_expanded = entropy.unsqueeze(2).repeat(1, 1, n)  # (b, n, n)

		decay_factor = torch.exp(-eta * entropy_expanded)  # (b, n, n)
		distance_term = beta * (1 - D_normalized) * decay_factor  # (b, n, n)

		T = mu.unsqueeze(1).unsqueeze(2) + alpha * sigma.unsqueeze(1).unsqueeze(2) + distance_term  # (b, n, n)

		probs = torch.softmax(-D / tau, dim=-1)
		sorted_distances, sorted_indices = torch.sort(D, dim=-1)
		sorted_probs = torch.gather(probs, dim=-1, index=sorted_indices)

		cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

		mask = cumulative_probs > prob_threshold
		first_over_index = torch.argmax(mask.int(), dim=-1)
		has_true = mask.any(dim=-1)
		num_edges = torch.where(has_true, first_over_index, torch.tensor(n, device=x.device))

		ranks = torch.arange(n, device=x.device).expand(b, n, n)

		mask_selected = ranks < num_edges.unsqueeze(-1)

		adj_mask = torch.zeros((b, n, n), device=x.device, dtype=torch.bool)
		adj_mask.scatter_(-1, sorted_indices, mask_selected)

		final_mask = adj_mask & (D < T)

		adj = final_mask.float()

		return adj

	def build_knn_adjacency(self, hs_flat, ms_flat, radar_flat):
		A11 = self.knn_similarity(hs_flat, hs_flat, method="euclidean")
		A22 = self.knn_similarity(ms_flat, ms_flat, method="euclidean")
		A33 = self.knn_similarity(radar_flat, radar_flat, method="euclidean")

		A12 = self.knn_similarity(hs_flat, ms_flat, method="cosine")  # HS -> MS
		A21 = self.knn_similarity(ms_flat, hs_flat, method="cosine")  # MS -> HS

		A13 = self.knn_similarity(hs_flat, radar_flat, method="cosine")  # HS -> Radar
		A31 = self.knn_similarity(radar_flat, hs_flat, method="cosine")  # Radar -> HS

		A23 = self.knn_similarity(ms_flat, radar_flat, method="cosine")  # MS -> Radar
		A32 = self.knn_similarity(radar_flat, ms_flat, method="cosine")  # Radar -> MS

		return A11, A22, A33, A12, A21, A13, A31, A23, A32

	def forward(self, x1, x2, x3):
		b, c, h, w = x1.shape

		x1_flat = x1.reshape(b, c, -1).permute(0, 2, 1)
		x2_flat = x2.reshape(b, c, -1).permute(0, 2, 1)
		x3_flat = x3.reshape(b, c, -1).permute(0, 2, 1)

		A11, A22, A33, A12, A21, A13, A31, A23, A32 = self.build_knn_adjacency(x1_flat, x2_flat, x3_flat)

		x1 = self.conv1(x1).reshape(b, c, -1).permute(0, 2, 1)
		x2 = self.conv2(x2).reshape(b, c, -1).permute(0, 2, 1)
		x3 = self.conv3(x3).reshape(b, c, -1).permute(0, 2, 1)

		att11 = torch.bmm(x1, x1.permute(0, 2, 1))
		att12 = torch.bmm(x1, x2.permute(0, 2, 1))
		att13 = torch.bmm(x1, x3.permute(0, 2, 1))
		att22 = torch.bmm(x2, x2.permute(0, 2, 1))
		att21 = torch.bmm(x2, x1.permute(0, 2, 1))
		att23 = torch.bmm(x2, x3.permute(0, 2, 1))
		att33 = torch.bmm(x3, x3.permute(0, 2, 1))
		att31 = torch.bmm(x3, x1.permute(0, 2, 1))
		att32 = torch.bmm(x3, x2.permute(0, 2, 1))

		att11 = torch.softmax(A11 * att11, dim=2)
		att12 = torch.softmax(A12 * att12, dim=2)
		att13 = torch.softmax(A13 * att13, dim=2)
		att22 = torch.softmax(A22 * att22, dim=2)
		att21 = torch.softmax(A21 * att21, dim=2)
		att23 = torch.softmax(A23 * att23, dim=2)
		att33 = torch.softmax(A33 * att33, dim=2)
		att31 = torch.softmax(A31 * att31, dim=2)
		att32 = torch.softmax(A32 * att32, dim=2)

		att_1 = torch.cat([att11, att12, att13], dim=2)
		att_2 = torch.cat([att21, att22, att23], dim=2)
		att_3 = torch.cat([att31, att32, att33], dim=2)

		x_cat = torch.cat([x1_flat, x2_flat, x3_flat], dim=1)

		x11 = torch.bmm(att_1, x_cat).permute(0, 2, 1).reshape(b, c, h, w)
		x22 = torch.bmm(att_2, x_cat).permute(0, 2, 1).reshape(b, c, h, w)
		x33 = torch.bmm(att_3, x_cat).permute(0, 2, 1).reshape(b, c, h, w)

		x = x11 + x22 + x33

		return x


class TriClip(nn.Module):
	def __init__(self, image_size, band1, band2, band3, num_classes,
				 embed_dim, context_length, vocab_size, transformer_width):
		super().__init__()
		self.H = image_size
		self.W = image_size
		self.cnn_encoder = CNN_Encoder(image_size, band1, band2, band3, embed_dim)
		self.cnn_classifier = CNN_Classifier(num_classes)
		self.cnn_classifier2 = CNN_Classifier(num_classes)
		self.cnn_classifier3 = CNN_Classifier(num_classes)

		self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

		self.transformer_width = transformer_width

		self.encode_text1 = Text_Encoder(embed_dim, context_length, vocab_size, transformer_width)

		self.decode_text1 = Text_Decoder(embed_dim, context_length, vocab_size, transformer_width)
		self.decode_text2 = Text_Decoder(embed_dim, context_length, vocab_size, transformer_width)
		self.decode_text3 = Text_Decoder(embed_dim, context_length, vocab_size, transformer_width)

		self.mse = nn.MSELoss()

		self.gcn_fusion = GCN_Fusion(in_channel=64)
		self.pooling = nn.MaxPool2d(2)


	def complementary_mask(self, image_emb, text_emb):
		batch_size_i, emb_dim_i = image_emb.shape
		batch_size_t, emb_dim_t = text_emb.shape
		assert batch_size_i == batch_size_t and emb_dim_i == emb_dim_t

		test_mask = torch.ones((image_emb.shape[0], image_emb.shape[1]), dtype=torch.bool).cuda()
		image_mask = torch.zeros((text_emb.shape[0], text_emb.shape[1]), dtype=torch.bool).cuda()

		similarity = cal_similarity(image_emb, text_emb)
		diagonal = torch.diagonal(similarity)

		_, topk_indices = torch.topk(diagonal, k=math.floor(len(diagonal) * 0.8), largest=True)

		# Apply masks
		for idx in topk_indices:
			test_mask[:, idx] = 0
			image_mask[:, idx] = 1

		# Apply the masks to the embeddings
		image_masked = image_emb * image_mask.float()
		test_masked = text_emb * test_mask.float()

		# Return only the visible patch
		b, d = image_emb.shape
		image_visible = image_masked[image_mask].reshape(b, -1)
		b, d = text_emb.shape
		text_visible = test_masked[test_mask].reshape(b, -1)

		return image_visible, text_visible, image_mask, test_mask

	def forward(self, data1, data2, data3, text, y):
		data1 = data1.permute(0, 3, 1, 2)
		data2 = data2.permute(0, 3, 1, 2)
		data3 = data3.permute(0, 3, 1, 2)

		out1, out2, out3, image_features1, image_features2, image_features3 = self.cnn_encoder(data1, data2, data3)
		out11, out22, out33 = self.pooling(out1), self.pooling(out2), self.pooling(out3)

		if self.training:
			text_features1 = self.encode_text1(text)

			# 掩码重建损失
			image_visible1, text_visible1, image_mask1, text_mask1 = self.complementary_mask(image_features1,
																							 text_features1)
			image_visible2, text_visible2, image_mask2, text_mask2 = self.complementary_mask(image_features2,
																							 text_features1)
			image_visible3, text_visible3, image_mask3, text_mask3 = self.complementary_mask(image_features3,
																							 text_features1)

			text_fusion1 = torch.zeros_like(text_features1)
			text_fusion1[text_mask1] = rearrange(text_visible1, 'b d -> (b d)')
			text_fusion1[~text_mask1] = rearrange(image_visible1, 'b d -> (b d)')

			text_fusion2 = torch.zeros_like(text_features1)
			text_fusion2[text_mask2] = rearrange(text_visible2, 'b d -> (b d)')
			text_fusion2[~text_mask2] = rearrange(image_visible2, 'b d -> (b d)')

			text_fusion3 = torch.zeros_like(text_features1)
			text_fusion3[text_mask3] = rearrange(text_visible3, 'b d -> (b d)')
			text_fusion3[~text_mask3] = rearrange(image_visible3, 'b d -> (b d)')

			decoded_embeddings1 = self.decode_text1(text_fusion1)
			decoded_embeddings2 = self.decode_text2(text_fusion2)
			decoded_embeddings3 = self.decode_text3(text_fusion3)

			token_embeddings = self.encode_text1.token_embedding(text)

			re_loss1 = self.mse(token_embeddings, decoded_embeddings1)
			re_loss2 = self.mse(token_embeddings, decoded_embeddings2)
			re_loss3 = self.mse(token_embeddings, decoded_embeddings3)
			re_loss = (re_loss1 + re_loss2 + re_loss3) / 3

			# 对比损失
			image_features1 = image_features1 / image_features1.norm(dim=1, keepdim=True)
			image_features2 = image_features2 / image_features2.norm(dim=1, keepdim=True)
			image_features3 = image_features3 / image_features3.norm(dim=1, keepdim=True)
			text_features1 = text_features1 / text_features1.norm(dim=1, keepdim=True)

			logit_scale = self.logit_scale.exp()
			label = torch.arange(image_features1.shape[0]).cuda()

			logits_per_image1 = logit_scale * image_features1 @ text_features1.t()
			logits_per_text1 = logit_scale * text_features1 @ image_features1.t()
			loss_img1 = F.cross_entropy(logits_per_image1, label)
			loss_text1 = F.cross_entropy(logits_per_text1, label)
			loss_1 = (loss_img1 + loss_text1) / 2

			logits_per_image2 = logit_scale * image_features2 @ text_features1.t()
			logits_per_text2 = logit_scale * text_features1 @ image_features2.t()
			loss_img2 = F.cross_entropy(logits_per_image2, label)
			loss_text2 = F.cross_entropy(logits_per_text2, label)
			loss_2 = (loss_img2 + loss_text2) / 2

			logits_per_image3 = logit_scale * image_features3 @ text_features1.t()
			logits_per_text3 = logit_scale * text_features1 @ image_features3.t()
			loss_img3 = F.cross_entropy(logits_per_image3, label)
			loss_text3 = F.cross_entropy(logits_per_text3, label)
			loss_3 = (loss_img3 + loss_text3) / 2

			loss_c = (loss_1 + loss_2 + loss_3) / 3

		else:
			re_loss = torch.tensor(0)
			loss_c = torch.tensor(0)

		add = out1 + out2 + out3
		add1 = self.gcn_fusion(out1, out2, out3)
		add2 = self.gcn_fusion(out11, out22, out33)

		pred_1 = self.cnn_classifier(add)
		pred_2 = self.cnn_classifier2(add1)
		pred_3 = self.cnn_classifier3(add2)

		return pred_1, pred_2, pred_3, re_loss, loss_c
