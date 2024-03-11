import numpy as np
import scipy, torch
import sympy

def prox_1(v, rho):
	'''
	the proximal operator is P_X^k(v) = argmin_{x\in X} 1/2 \lVert x-v \rVert_2^2 + \hat{\rho}/2 \lVert x-x^k \rVert_1^2
	
	'''
	d, ll = np.size(v), np.linalg.norm(v, ord = 1)
	x, summ = np.zeros(np.shape(v)), np.zeros(d+1)
	idx = np.argsort(np.abs(v), axis = 0)
	for i in range(1, d+1):
		summ[d - i] = summ[d - i + 1] + np.abs(v[idx[d - i], 0])
	if ll == 0:  # l1_norm of z = 0(or say >= v(d))
		return x
	if ll * rho / (d*rho + 1) <= np.abs(v[idx[0], 0]): # l1_norm of z < v(0)
		for i in range(d):
			x[idx[i], 0] = v[idx[i], 0] - np.sign(v[idx[i], 0]) * ll * rho / (d*rho + 1)
	for i in range(1, d):
		dd = summ[i] * rho / ((d - i)*rho + 1)
		if dd > np.abs(v[idx[i-1], 0]) and dd <= np.abs(v[idx[i], 0]):
		# v(i-1) <= l1_norm of z < v(i), note that it can not have "=" on both sides
			for t in range(i, d):
				x[idx[t], 0] = v[idx[t], 0] - np.sign(v[idx[t], 0]) * summ[i] * rho / ((d - i)*rho + 1)
			# break
	return x


def prox_2(v, psi):
	if np.linalg.norm(v, ord = 1) <= psi:
		return v
	d, k = np.size(v), 1
	idx = np.flip(np.argsort(np.abs(v), axis = 0), axis = 0)
	s = np.zeros(d+1)
	for i in range(1, d):
		s[i] = s[i - 1] + i * (np.abs(v[idx[i-1], 0]) - np.abs(v[idx[i], 0]))
	s[d] = s[d-1] + d * np.abs(v[idx[d-1], 0])
	for tmp in range(1, d+1):
		if s[tmp-1] < psi and psi <= s[tmp]:
			k = tmp
			break
	# print(k, s)
	if k == d:
		lamb = (np.linalg.norm(v, ord = 1) - psi) / k
	else:
		lamb = (s[k] + k * np.abs(v[idx[k], 0]) - psi) / k
	x = np.zeros(np.shape(v))
	for i in range(d):
		if v[i, 0] > lamb:
			x[i, 0] = v[i, 0] - lamb
		if v[i, 0] < -lamb:
			x[i, 0] = v[i, 0] + lamb
	return x

def prox_box(v, psi):
	return np.where(np.abs(v) <= psi, v, psi * np.sign(v))

def prox_1_admm_l1(v, xk, rho, R, beta, epsi_hat):
	x, y, lamb = np.zeros(np.shape(v)), np.zeros(np.shape(v)), 0
	oldx, oldy, oldlamb = x, y, lamb
	while True:
		x = prox_2((v + beta * (y + xk) - lamb) / (beta + 1), R)
		y = prox_1((beta * (x - xk) + lamb) / beta, rho / beta)
		lamb = lamb + beta * (x - y - xk)
		if np.linalg.norm(x - y - xk) <= epsi_hat and beta * np.linalg.norm(y - oldy) <= epsi_hat:
			return x
		oldx, oldy, oldlamb = x, y, lamb

def prox_1_admm_box(v, xk, rho, R, beta, epsi_hat):
	x, y, lamb = np.zeros(np.shape(v)), np.zeros(np.shape(v)), 0
	oldx, oldy, oldlamb = x, y, lamb
	while True:
		x = prox_box((v + beta * (y + xk) - lamb) / (beta + 1), R)
		y = prox_1((beta * (x - xk) + lamb) / beta, rho / beta)
		lamb = lamb + beta * (x - y - xk)
		if np.linalg.norm(x - y - xk) <= epsi_hat and beta * np.linalg.norm(y - oldy) <= epsi_hat:
			return x
		oldx, oldy, oldlamb = x, y, lamb

def prox_MD(G, xk, alpha, p):
	m, x, d, xk_norm = np.zeros(np.shape(xk)), np.zeros(np.shape(xk)), np.size(xk), np.linalg.norm(xk[:,0], ord = p)
	q = p / (p-1)
	sum1 = 0
	for i in range(d):
		m[i, 0] = alpha * G[i, 0] / (np.exp(2) *np.log(d)) - xk_norm**(2-p) * np.abs(xk[i, 0])**(p-1) * np.sign(xk[i, 0]) # 
	norm_q = np.linalg.norm(m[:,0], q)
	for i in range(d):
		sum1 += np.abs(m[i,0])**(1/(p-1)) * (norm_q)**((p-2)/(p-1))
	
	for i in range(d):
		x[i,0] = -np.sign(m[i,0]) * np.abs(m[i,0])**(1/(p-1)) * (norm_q)**((p-2)/(p-1))
	return x

def MD_func(lamb, *data):
	m, alpha, p, R = data
	tmpm = (np.abs(np.abs(m) - alpha * lamb) + (np.abs(m) - alpha * lamb))/2
	sum1, sum2 = 0, 0
	for i in range(np.size(tmpm)):
		if tmpm[i, 0] != 0:
			sum1 += tmpm[i, 0]**(1/(p-1))
			sum2 += tmpm[i, 0]**(p/(p-1))
	return sum1 * (sum2 ** ((p-2)/p)) - R

