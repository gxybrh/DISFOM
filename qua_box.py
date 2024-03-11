import numpy as np
import sys, os
import scipy
sys.path.append(os.path.abspath("./prox_operator"))
from prox_1 import * 


def qua_loss(Sigma, xop, x, lambda_reg):
    return ((x-xop).T @ Sigma @ (x-xop)/2)[0, 0] + lambda_reg * np.sum(x**2 / (1 + x**2))

def qua_grad(Sigma, xop, x, lambda_reg):
    return Sigma @ (x-xop) + lambda_reg * (2 * x / (1 + x**2)**2)

def qua_batchgrad(alpha, b, x, lambda_reg):
    return alpha.T @ (alpha @ x - b) / batch_size + lambda_reg * (2 * x / (1 + x**2)**2)

def qua_measure(Sigma, xop, x, lambda_reg):
    g = Sigma @ (x-x_op) + lambda_reg * (2 * x / (1 + x**2)**2)
    v = g
    for i in range(np.shape(x)[0]):
        if np.abs(x[i, 0] == R) and np.sign(x[i,0]) != np.sign(g[i, 0]):
            v[i, 0] = 0
    return np.linalg.norm(v, np.inf)

def l2_measure(Sigma, xop, x, lambda_reg):
    return np.linalg.norm(Sigma @ (x-x_op) + lambda_reg * (2 * x / (1 + x**2)**2), 2)

def proximal(Sigma, xop, x, lambda_reg):
    c1, beta, k = 1/4, 1/2, 1
    while True:
        a = 1
        ori = np.zeros((d, 1))
        del1 = qua_loss(Sigma, xop, x - a * qua_grad(Sigma, xop, x, lambda_reg), lambda_reg)
        del2 = qua_loss(Sigma, xop, x, lambda_reg)
        gradel = - a * qua_grad(Sigma, xop, x, lambda_reg)
        tmpgra = qua_grad(Sigma, xop, x, lambda_reg).T

        while del1 - del2 > c1 * (tmpgra @ gradel)[0, 0]:
            del1 = qua_loss(Sigma, xop, x - a * qua_grad(Sigma, xop, x, lambda_reg), lambda_reg)
            del2 = qua_loss(Sigma, xop, x, lambda_reg)
            gradel = - a * qua_grad(Sigma, xop, x, lambda_reg)
            tmpgra = qua_grad(Sigma, xop, x, lambda_reg).T
            a *= beta
        x_temp = x - a * qua_grad(Sigma, xop, x, lambda_reg)
        if np.linalg.norm(x - x_temp, 1) <= 1e-10:
            break
        else:
            x = x_temp
    return x, qua_loss(Sigma, xop, x, lambda_reg)

def DIFOM_minibatch(tmpsigma, xop, x0, learning_rate, M, batch_size, t=0):
    x_ret = x0
    tmp_loss = qua_loss(Sigma, xop, x_ret, lambda_reg)
    delta = tmp_loss - min_val
    print(f"Epoch {0}, Loss: {tmp_loss - min_val}")
    epoch = 0
    loss_difom[0, t], fo_measure_difom[0, t], l2_difom[0, t] = delta, delta, np.linalg.norm(Sigma @ xop, np.inf)

    while True:
        x = x_ret
        alpha = np.hstack((np.clip(np.random.multivariate_normal(np.zeros(d1), np.eye(d1, d1), size = batch_size), -R, R) @ scipy.linalg.sqrtm(tmpsigma), np.clip(np.random.normal(size = (batch_size, d - d1)), -R, R)))
        w = np.clip(np.random.normal(0, 1, size = (batch_size, 1)), -R, R)
        b = np.dot(alpha, xop) + w
        grad = qua_batchgrad(alpha, b, x, lambda_reg)
        x = prox_1_admm_box(x - learning_rate * grad, x, 2, R, 1, epsi_hat)

        epoch = epoch + 1
        tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
        loss_difom[epoch, t] = (tmp_loss - min_val)
        measure, measure_2 = qua_measure(Sigma, xop, x, lambda_reg), l2_measure(Sigma, xop, x, lambda_reg)
        print(f"Epoch {epoch}, Loss: {(tmp_loss - min_val)}, Measure: {measure}")
        
        fo_measure_difom[epoch, t] = measure
        l2_difom[epoch, t] = measure_2
        if epoch == M:
            ret = loss_difom[np.random.randint(1, M+1), 0]
            print("loss:", ret, ", d:", d, ", delta:", delta, ", end norm:", np.linalg.norm(x_ret, 1))
            return loss_difom, fo_measure_difom, l2_difom
        x_ret = x

def proximal_SGD(tmpsigma, xop, x0, learning_rate, M, batch_size, t=0):
    x_ret = x0
    tmp_loss = qua_loss(Sigma, xop, x_ret, lambda_reg)
    delta = tmp_loss - min_val
    loss_prox[0, t], fo_measure_prox[0, t], l2_prox[0, t] = delta, delta, np.linalg.norm(Sigma @ xop, np.inf)



    print(f"Epoch {0}, Loss: {tmp_loss - min_val}")
    epoch = 0
    while True:
        x = x_ret
        alpha = np.hstack((np.clip(np.random.multivariate_normal(np.zeros(d1), np.eye(d1, d1), size = batch_size), -R, R) @ scipy.linalg.sqrtm(tmpsigma), np.clip(np.random.normal(size = (batch_size, d - d1)), -R, R)))
        w = np.clip(np.random.normal(0, 1, size = (batch_size, 1)), -R, R)
        b = np.dot(alpha, xop) + w
        grad = qua_batchgrad(alpha, b, x, lambda_reg)
        x_temp = prox_box(x - learning_rate * grad, R)

        x = x_temp
        epoch = epoch + 1
        tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
        loss_prox[epoch, t] = (tmp_loss - min_val)       
        measure = qua_measure(Sigma, xop, x, lambda_reg)
        fo_measure_prox[epoch, t] = measure
        print(f"Epoch {epoch}, Loss: {(tmp_loss - min_val)}, Measure: {measure}")
        l2_prox[epoch, t] = l2_measure(Sigma, xop, x, lambda_reg)

        if epoch == M:
            ret = loss_prox[np.random.randint(1, M+1), 0]
            print("loss:", ret, ", d:", d, ", delta:", delta, ", end norm:", np.linalg.norm(x_ret, 1))
            return loss_prox, fo_measure_prox, l2_prox
        x_ret = x


def DIFOM_svrg(tmpsigma, xop, x0, learning_rate, M, batch_size_1, t=0):
    x_ret = np.zeros(np.shape(x0))
    x = x0.copy()
    tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
    delta = tmp_loss - min_val
    epoch_length = int(np.power(batch_size_1, 1/3))
    batch_size = int(np.power(batch_size_1, 2/3))
    mm = np.floor(M/2)*(epoch_length - 1) + np.ceil(M/2)+1
    loss_difom_svrg[0, t], fo_measure_difom_svrg[0, t], l2_difom_svrg[0, t] = delta, delta, np.linalg.norm(Sigma @ xop, np.inf)

    
    print(f"Epoch {0}, Loss: {tmp_loss - min_val}")
    epoch, cnt = 0, 0
    while True:
        if cnt % epoch_length == 0:
            alpha = np.hstack((np.clip(np.random.multivariate_normal(np.zeros(d1), np.eye(d1, d1), size = batch_size_1), -R, R) @ scipy.linalg.sqrtm(tmpsigma), np.clip(np.random.normal(size = (batch_size_1, d - d1)), -R, R)))
            w = np.clip(np.random.normal(0, 1, size = (batch_size_1, 1)), -R, R)
            b = np.dot(alpha, xop) + w

            nk_grad = qua_batchgrad(alpha, b, x, lambda_reg).reshape((-1, 1))
            x_tilde = x
            epoch += 1
            cnt += 1
            x = prox_1_admm_box(x - learning_rate * nk_grad, x, 128, R, 1, epsi_hat)
            tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
            loss_difom_svrg[cnt, t] = (tmp_loss - min_val)
            measure = qua_measure(Sigma, xop, x, lambda_reg)
            fo_measure_difom_svrg[cnt, t] = measure
            l2_difom_svrg[cnt, t] = l2_measure(Sigma, xop, x, lambda_reg)

            print(f"Epoch {epoch}, Loss: {(tmp_loss - min_val)}, Measure: {measure}")
        else:
            cnt += 1
            if cnt % epoch_length == 0:
                epoch += 1
            alpha = np.hstack((np.clip(np.random.multivariate_normal(np.zeros(d1), np.eye(d1, d1), size = batch_size), -R, R) @ scipy.linalg.sqrtm(tmpsigma), np.clip(np.random.normal(size = (batch_size, d - d1)), -R, R)))
            w = np.clip(np.random.normal(0, 1, size = (batch_size, 1)), -R, R)
            b = np.dot(alpha, xop) + w

            grad = qua_batchgrad(alpha, b, x, lambda_reg).reshape((-1, 1))
            grad_nk = qua_batchgrad(alpha, b, x_tilde, lambda_reg).reshape((-1, 1))
            grad_corr = grad - grad_nk + nk_grad
            x = prox_1_admm_box(x - learning_rate * grad_corr, x, 128, R, 1, epsi_hat)
            tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
            measure = qua_measure(Sigma, xop, x, lambda_reg)
            loss_difom_svrg[cnt, t] = (tmp_loss - min_val)
            fo_measure_difom_svrg[cnt, t] = measure
            l2_difom_svrg[cnt, t] = l2_measure(Sigma, xop, x, lambda_reg)
            print(f"Epoch {epoch}, Loss: {(tmp_loss - min_val)}, Measure: {measure}")
        

        if epoch == M:
            ret = loss_difom_svrg[np.random.randint(1, M+1), 0]
            print("loss:", ret, ", d:", d, ", delta:", delta, ", end norm:", np.linalg.norm(x_ret, 1))
            return loss_difom_svrg, fo_measure_difom_svrg, l2_difom_svrg
        x_ret = x

def svrg(tmpsigma, xop, x0, learning_rate, M, batch_size_1, t=0):
    x_ret = np.zeros(np.shape(x0))
    x = x0.copy()
    tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
    delta = tmp_loss - min_val
    epoch_length = int(np.power(batch_size_1, 1/3))
    batch_size = int(np.power(batch_size_1, 2/3))
    mm = np.floor(M/2)*(epoch_length - 1) + np.ceil(M/2)+1
    loss_svrg[0, t], fo_measure_svrg[0, t], l2_svrg[0, t] = delta, delta, np.linalg.norm(Sigma @ xop, np.inf)

    
    print(f"Epoch {0}, Loss: {tmp_loss - min_val}")
    epoch, cnt = 0, 0
    while True:
        if cnt % epoch_length == 0:
            alpha = np.hstack((np.clip(np.random.multivariate_normal(np.zeros(d1), np.eye(d1, d1), size = batch_size_1), -R, R) @ scipy.linalg.sqrtm(tmpsigma), np.clip(np.random.normal(size = (batch_size_1, d - d1)), -R, R)))
            w = np.clip(np.random.normal(0, 1, size = (batch_size_1, 1)), -R, R)
            b = np.dot(alpha, xop) + w

            nk_grad = qua_batchgrad(alpha, b, x, lambda_reg).reshape((-1, 1))
            x_tilde = x
            epoch += 1
            cnt += 1
            x = prox_box(x - learning_rate * nk_grad, R)
            tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
            loss_svrg[cnt, t] = (tmp_loss - min_val)
            measure = qua_measure(Sigma, xop, x, lambda_reg)
            fo_measure_svrg[cnt, t] = measure
            if epoch <= 300:
                l2_svrg[cnt, t] = l2_measure(Sigma, xop, x, lambda_reg)

            print(f"Epoch {epoch}, Loss: {(tmp_loss - min_val)}, Measure: {measure}")
        else:
            cnt += 1
            if cnt % epoch_length == 0:
                epoch += 1
            alpha = np.hstack((np.clip(np.random.multivariate_normal(np.zeros(d1), np.eye(d1, d1), size = batch_size), -R, R) @ scipy.linalg.sqrtm(tmpsigma), np.clip(np.random.normal(size = (batch_size, d - d1)), -R, R)))
            w = np.clip(np.random.normal(0, 1, size = (batch_size, 1)), -R, R)
            b = np.dot(alpha, xop) + w

            grad = qua_batchgrad(alpha, b, x, lambda_reg).reshape((-1, 1))
            grad_nk = qua_batchgrad(alpha, b, x_tilde, lambda_reg).reshape((-1, 1))
            grad_corr = grad - grad_nk + nk_grad
            x = prox_box(x - learning_rate * grad_corr, R)
            tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
            measure = qua_measure(Sigma, xop, x, lambda_reg)
            loss_svrg[cnt, t] = (tmp_loss - min_val)
            fo_measure_svrg[cnt, t] = measure
            if epoch <= 300:
                l2_svrg[cnt, t] = l2_measure(Sigma, xop, x, lambda_reg)
            print(f"Epoch {epoch}, Loss: {(tmp_loss - min_val)}, Measure: {measure}")
        

        if epoch == M:
            ret = loss_svrg[np.random.randint(1, M+1), 0]
            print("loss:", ret, ", d:", d, ", delta:", delta, ", end norm:", np.linalg.norm(x_ret, 1))
            return loss_svrg, fo_measure_svrg, l2_svrg
        x_ret = x

def SMD_minibatch(tmpsigma, xop, x0, learning_rate, M, batch_size, t=0):
    x_ret = x0
    tmp_loss = qua_loss(Sigma, xop, x_ret, lambda_reg)
    # pre_loss = 0
    delta = tmp_loss - min_val
    loss_smd[0, t], fo_measure_smd[0, t], l2_smd[0, t] = delta, delta, np.linalg.norm(Sigma @ xop, np.inf)
    rho = lambda_reg / 2 - 1
    c = np.sqrt((tmp_loss)/(rho * L**2))
    alpha_k = c / np.sqrt(M)

    print(f"Epoch {0}, Loss: {tmp_loss - min_val}")
    epoch = 0
    while True:
        x = x_ret
        alpha = np.hstack((np.clip(np.random.multivariate_normal(np.zeros(d1), np.eye(d1, d1), size = batch_size), -R, R) @ scipy.linalg.sqrtm(tmpsigma), np.clip(np.random.normal(size = (batch_size, d - d1)), -R, R)))
        w = np.clip(np.random.normal(0, 1, size = (batch_size, 1)), -R, R)
        b = np.dot(alpha, xop) + w
        grad = qua_batchgrad(alpha, b, x, lambda_reg)
        x_temp = prox_MD(grad, x, alpha_k, 1+1/np.log(np.size(x)))

        x = x_temp
        epoch = epoch + 1
        # pre_loss = tmp_loss
        tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
        loss_smd[epoch, t] = (tmp_loss - min_val)      
        measure = qua_measure(Sigma, xop, x, lambda_reg)
        fo_measure_smd[epoch, t] = measure
        l2_smd[epoch, t] = l2_measure(Sigma, xop, x, lambda_reg)
        print(f"Epoch {epoch}, Loss: {(tmp_loss - min_val)}, Measure: {measure}")
        if epoch == M:
            ret = loss_smd[np.random.randint(1, M+1), 0]
            print("loss:", ret, ", d:", d, ", delta:", delta, ", end norm:", np.linalg.norm(x_ret, 1))
            return loss_smd, fo_measure_smd, l2_smd
        x_ret = x

def SMD_svrg(tmpsigma, xop, x0, learning_rate, M, batch_size_1, t=0):
    x_ret = np.zeros(np.shape(x0))
    x = x0.copy()
    tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
    delta = tmp_loss - min_val
    epoch_length = int(np.power(batch_size_1, 1/3))
    batch_size = int(np.power(batch_size_1, 2/3))
    mm = np.floor(M/2)*(epoch_length - 1) + np.ceil(M/2)+1
    loss_smd_svrg[0, t], fo_measure_smd_svrg[0, t], l2_smd_svrg[0, t] = delta, delta, np.linalg.norm(Sigma @ xop, np.inf)
    rho = lambda_reg / 2 - 1
    c = np.sqrt((tmp_loss)/(rho * L**2))
    alpha_k = c / np.sqrt(M)

    
    print(f"Epoch {0}, Loss: {tmp_loss - min_val}")
    epoch, cnt = 0, 0
    while True:
        if cnt % epoch_length == 0:
            alpha = np.hstack((np.clip(np.random.multivariate_normal(np.zeros(d1), np.eye(d1, d1), size = batch_size_1), -R, R) @ scipy.linalg.sqrtm(tmpsigma), np.clip(np.random.normal(size = (batch_size_1, d - d1)), -R, R)))
            w = np.clip(np.random.normal(0, 1, size = (batch_size_1, 1)), -R, R)
            b = np.dot(alpha, xop) + w

            nk_grad = qua_batchgrad(alpha, b, x, lambda_reg).reshape((-1, 1))
            x_tilde = x
            epoch += 1
            cnt += 1
            x = prox_MD(nk_grad, x, alpha_k, 1+1/np.log(np.size(x)))
            tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
            loss_smd_svrg[cnt, t] = (tmp_loss - min_val)
            measure = qua_measure(Sigma, xop, x, lambda_reg)
            fo_measure_smd_svrg[cnt, t] = measure
            l2_smd_svrg[cnt, t] = l2_measure(Sigma, xop, x, lambda_reg)

            print(f"Epoch {epoch}, Loss: {(tmp_loss - min_val)}, Measure: {measure}")
        else:
            cnt += 1
            if cnt % epoch_length == 0:
                epoch += 1
            alpha = np.hstack((np.clip(np.random.multivariate_normal(np.zeros(d1), np.eye(d1, d1), size = batch_size), -R, R) @ scipy.linalg.sqrtm(tmpsigma), np.clip(np.random.normal(size = (batch_size, d - d1)), -R, R)))
            w = np.clip(np.random.normal(0, 1, size = (batch_size, 1)), -R, R)
            b = np.dot(alpha, xop) + w

            grad = qua_batchgrad(alpha, b, x, lambda_reg).reshape((-1, 1))
            grad_nk = qua_batchgrad(alpha, b, x_tilde, lambda_reg).reshape((-1, 1))
            grad_corr = grad - grad_nk + nk_grad
            x = prox_MD(grad_corr, x, alpha_k, 1+1/np.log(np.size(x)))
            tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
            measure = qua_measure(Sigma, xop, x, lambda_reg)
            loss_smd_svrg[cnt, t] = (tmp_loss - min_val)
            fo_measure_smd_svrg[cnt, t] = measure
            l2_smd_svrg[cnt, t] = l2_measure(Sigma, xop, x, lambda_reg)
            print(f"Epoch {epoch}, Loss: {(tmp_loss - min_val)}, Measure: {measure}")
        

        if epoch == M:
            ret = loss_smd_svrg[np.random.randint(1, M+1), 0]
            print("loss:", ret, ", d:", d, ", delta:", delta, ", end norm:", np.linalg.norm(x_ret, 1))
            return loss_smd_svrg, fo_measure_smd_svrg, l2_smd_svrg
        x_ret = x



np.random.seed(10)
c = 7
R = 3
d, d1 = 10000, 100#
d = int(sys.argv[1])
print(d)
d1 = int(d / 16)
epsi_hat = 1e-4
num_exp = 3
lambda_reg = 2.5

Sigma = scipy.sparse.eye(d, format = 'csr')
Q = scipy.linalg.orth(np.random.uniform(0, 1, size = (d1, d1)))
D = np.diag(np.random.uniform(1, 2, size = d1))
tmpsigma = np.dot(np.dot(Q, D), Q.T)
i, j, v, cnt = np.zeros(d1*d1), np.zeros(d1*d1), np.zeros(d1*d1), 0
for ii in range(d1):
    for jj in range(d1):
        i[cnt], j[cnt], v[cnt] = ii, jj, tmpsigma[ii,jj]
        if ii == jj:
            v[cnt] -= 1
        cnt += 1
Sigma = (Sigma + scipy.sparse.csr_matrix((v, (i,j)), shape = (d, d))) * (1 - 2*R * np.exp(-R**2/2)/(np.sqrt(2*np.pi) * 0.99730020393674))


x_op = np.zeros((d, 1))
x_op[range(d1), :] = np.ones((d1, 1))

sigma = 2 * np.max(Sigma.diagonal())
vals, vecs = scipy.sparse.linalg.eigs(Sigma, which = 'LR')
L = np.max(np.real(vals)) + 2 * lambda_reg
learning_rate = 1 / L
epsi = 3


x0 = np.zeros((d, 1))
batch_size = 1000

min_x, min_val = proximal(Sigma, x_op, x_op, lambda_reg)
Delta = qua_loss(Sigma, x_op, x0, lambda_reg) - min_val
M = 300
mm = np.floor(M/2)*(int(np.power(batch_size, 1/3)) - 1) + np.ceil(M/2)+1
print(batch_size,M)

loss_difom, fo_measure_difom, l2_difom = np.zeros((M+1, num_exp)), np.zeros((M+1, num_exp)), np.zeros((M+1, num_exp))
loss_prox, fo_measure_prox, l2_prox = np.zeros((M+1, num_exp)), np.zeros((M+1, num_exp)), np.zeros((M+1, num_exp))
loss_smd, fo_measure_smd, l2_smd = np.zeros((M+1, num_exp)), np.zeros((M+1, num_exp)), np.zeros((M+1, num_exp))
loss_svrg, fo_measure_svrg, l2_svrg = np.zeros((int(mm), num_exp)), np.zeros((int(mm), num_exp)), np.zeros((int(mm), num_exp))
loss_difom_svrg, fo_measure_difom_svrg, l2_difom_svrg = np.zeros((int(mm), num_exp)), np.zeros((int(mm), num_exp)), np.zeros((int(mm), num_exp))
loss_smd_svrg, fo_measure_smd_svrg, l2_smd_svrg = np.zeros((int(mm), num_exp)), np.zeros((int(mm), num_exp)), np.zeros((int(mm), num_exp))
print(np.linalg.norm(min_x), min_val)
for t in range(num_exp):
    loss_difom, fo_measure_difom, l2_difom = DIFOM_minibatch(tmpsigma, x_op, x0, learning_rate, M, batch_size, t)
    loss_prox, fo_measure_prox, l2_prox = proximal_SGD(tmpsigma, x_op, x0, learning_rate, M, batch_size, t)
    loss_smd, fo_measure_smd, l2_smd = SMD_minibatch(tmpsigma, x_op, x0, learning_rate, M, batch_size, t)
    loss_svrg, fo_measure_svrg, l2_svrg = svrg(tmpsigma, x_op, x0, learning_rate/10, M, batch_size, t)
    loss_difom_svrg, fo_measure_difom_svrg, l2_difom_svrg = DIFOM_svrg(tmpsigma, x_op, x0, learning_rate, M, batch_size, t)
    loss_smd_svrg, fo_measure_smd_svrg, l2_smd_svrg = SMD_svrg(tmpsigma, x_op, x0, learning_rate, M, batch_size, t)
np.savetxt("./data/save_para/loss" + str(d) + "_difom.csv",loss_difom)
np.savetxt("./data/save_para/fo_measure" + str(d) + "_difom.csv",fo_measure_difom)

np.savetxt("./data/save_para/loss" + str(d) + "_prox.csv",loss_prox)
np.savetxt("./data/save_para/fo_measure" + str(d) + "_prox.csv",fo_measure_prox)

np.savetxt("./data/save_para/loss" + str(d) + "_smd.csv",loss_smd)
np.savetxt("./data/save_para/fo_measure" + str(d) + "_smd.csv",fo_measure_smd)


np.savetxt("./data/save_para/loss" + str(d) + "_svrg.csv",loss_svrg)
np.savetxt("./data/save_para/fo_measure" + str(d) + "_svrg.csv",fo_measure_svrg)

np.savetxt("./data/save_para/loss" + str(d) + "_difom_svrg.csv",loss_difom_svrg)
np.savetxt("./data/save_para/fo_measure" + str(d) + "_difom_svrg.csv",fo_measure_difom_svrg)

np.savetxt("./data/save_para/loss" + str(d) + "_smd_svrg.csv",loss_smd_svrg)
np.savetxt("./data/save_para/fo_measure" + str(d) + "_smd_svrg.csv",fo_measure_smd_svrg)




