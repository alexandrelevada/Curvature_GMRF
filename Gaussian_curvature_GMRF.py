'''
Python source code for the paper 'On the curvatures of Gaussian random field manifolds'

MCMC simulation to compute the principal, mean and Gaussian curvatures of the parametric space of a Gaussian random field during phase transitions

'''

import numpy as np
import scipy.misc as spm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle							
import time
import warnings
from mpl_toolkits.mplot3d import Axes3D
from imageio import imwrite
from numpy import log
from skimage.io import imsave
from scipy.optimize import root_scalar

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')


start = time.time()

MAX_IT = 1000      						
MEIO = MAX_IT//2
SIZE = 256								
METADE = MEIO
BURN_IN = 5

beta = 0				# inverse temperature
media = 0				# mean
sigma = 5				# variance

# first and second fundamental forms
tensor1 = np.zeros((3,3))
tensor2 = np.zeros((3,3))

# Initial random fiedl configuration
img = np.random.normal(media, sigma, (SIZE, SIZE))

# To store all configurations along the MCMC sinulation
sequence = np.zeros((MAX_IT, img.shape[0], img.shape[1]))

# Duplicate the edges
K = 1
img = np.lib.pad(img, ((K,K), (K,K)), 'symmetric')

nlin, ncol = (img.shape[0], img.shape[1])

# To measure the variation in the components of the first and seconf fundamental forms and curvatures
vetor_media = np.zeros(MAX_IT)
vetor_variancia = np.zeros(MAX_IT)
vetor_beta = np.zeros(MAX_IT)
vetor_betaMPL = np.zeros(MAX_IT)
vetor_ent_gauss = np.zeros(MAX_IT)
vetor_ent = np.zeros(MAX_IT)
vetor_phi = np.zeros(MAX_IT)
vetor_psi = np.zeros(MAX_IT)
vetor_mu1 = np.zeros(MAX_IT)
vetor_mu2 = np.zeros(MAX_IT)
vetor_sigma1 = np.zeros(MAX_IT)
vetor_sigma2 = np.zeros(MAX_IT)
vetor_sigbeta1 = np.zeros(MAX_IT)
vetor_sigbeta2 = np.zeros(MAX_IT)
vetor_ds2_1 = np.zeros(MAX_IT)
vetor_ds2_2 = np.zeros(MAX_IT)
vetor_curvaturaG = np.zeros(MAX_IT)
vetor_curvaturaM = np.zeros(MAX_IT)
vetor_K1 = np.zeros(MAX_IT)
vetor_K2 = np.zeros(MAX_IT)
vetor_K3 = np.zeros(MAX_IT)

col = 9  			# number of elements of a 3 x 3 patch
centro = col//2		# index to the central element
delta = 8     		# number of neighbors

sample = img.copy()

# Boolean variables to detect change in the sign of Gaussian curvature
OK1, OK2 = False, False

############## Main loop ####################
for iteracoes in range(0, MAX_IT):

	# Samples for the computation of the covariance matrix of the patches
	amostras = np.zeros(((nlin-2)*(ncol-2), col))
	ind = 0

	print('\nIteration ', iteracoes)
	
	for i in range(K, nlin-K):
		for j in range(K, ncol-K):

			neigh = img[i-1:i+2, j-1:j+2]
			neigh = np.reshape(neigh, neigh.shape[0]*neigh.shape[1])
			amostras[ind,:] = neigh
			ind += 1
			vizinhanca = np.concatenate((neigh[0:(neigh.size//2)], neigh[(neigh.size//2)+1:neigh.size]))
			central = neigh[neigh.size//2]
			
			# Probability of the current value of xi
			P1 = (1/np.sqrt(2*np.pi*sigma))*np.exp((-1/(2*sigma))*(central - media - beta*sum(vizinhanca - media))**2)
			# Choose a random value from a Gaussian distribution
			g = np.random.normal(media, sigma)
			# Discard outliers
			while (g < media - 3*np.sqrt(sigma)) or (g > media + 3*np.sqrt(sigma)):	
				g = np.random.normal(media, sigma)
			# Computes the probability of the new value g
			P2 = (1/np.sqrt(2*np.pi*sigma))*np.exp((-1/(2*sigma))*(g - media - beta*(vizinhanca - media).sum())**2)
			# Pick the smallest between 1 and P1/P2 = p
			limiar = 1
			razao = P2/P1
			if (razao < 1):
				limiar = razao
			# Accept the new value g with probability p
			epson = np.random.rand()
			if epson <= limiar:
				sample[i,j] = g

	img = sample.copy()
	nucleo = img[K:nlin-K, K:ncol-K]

	media_est = nucleo.mean()	
	variancia = nucleo.var()

	############# Compute the components of the fundamental forms ############
	mc = np.cov(amostras.T)		# covariance matrix

	# Computes (Sigma_{p}^{-})
	sigma_minus = mc.copy()
	sigma_minus[:,centro] = 0
	sigma_minus[centro,:] = 0

	# Estimation of the inverse temperature via maximum pseudo-likelihood (not being used here!)
	# We use the real inverse temperature parameter
	left_half = mc[centro, 0:centro]
	right_half = mc[centro, centro+1:col]
	rho = np.concatenate((left_half, right_half))
	
	# We use the real beta
	beta_cov = beta 	
	
	print('beta = %.6f' % beta_cov)
	print('mean = %.6f' % media_est)
	print('variance = %.6f' % variancia)

	# Computes the symmetrized KL divergence
	if iteracoes == BURN_IN:
		mediaA, varianciaA, betaA = media_est, variancia, beta_cov
		rho_sum_A = rho.sum()
		sigma_minus_sum_A = sigma_minus.sum()
	elif iteracoes == MEIO:
		mediaB, varianciaB, betaB = media_est, variancia, beta_cov
		rho_sum_B = rho.sum()
		sigma_minus_sum_B = sigma_minus.sum()
		dKLsym_AB = (1/(4*varianciaA*varianciaB))*( (varianciaA - varianciaB)**2 - 2*(betaB*varianciaA - betaA*varianciaB)*(rho_sum_A - rho_sum_B) + (betaB**2 * varianciaA - betaA**2 * varianciaB)*(sigma_minus_sum_A - sigma_minus_sum_B) + (mediaA - mediaB)**2 * (varianciaA*(1 - delta*betaB)**2 + varianciaB*(1 - delta*betaA)**2 ) )
		print()		
		print('dKLsym(A, B) = %f' %dKLsym_AB)
		print()
	elif iteracoes == MAX_IT - 1:
		mediaC, varianciaC, betaC = media, variancia, beta_cov
		rho_sum_C = rho.sum()
		sigma_minus_sum_C = sigma_minus.sum()
		dKLsym_BC = (1/(4*varianciaB*varianciaC))*( (varianciaB - varianciaC)**2 - 2*(betaC*varianciaB - betaB*varianciaC)*(rho_sum_B - rho_sum_C) + (betaC**2 * varianciaB - betaB**2 * varianciaC)*(sigma_minus_sum_B - sigma_minus_sum_C) + (mediaB - mediaC)**2 * (varianciaB*(1 - delta*betaC)**2 + varianciaC*(1 - delta*betaB)**2 ) )
		print()		
		print('dKLsym(B, c) = %f' %dKLsym_BC)
		print()

	
	####### Computation of component I_mu_mu	
	mu_1 = (1/variancia)*((1 - beta_cov*delta)**2)*(1 - (1/variancia)*(2*beta_cov*rho.sum() - (beta_cov**2)*sigma_minus.sum()))
	tensor1[0,0] = mu_1
	print('mu_1 = %.6f' %mu_1)	

	####### Computation of component II_mu_mu}	
	mu_2 = (1/variancia)*((1 - beta_cov*delta)**2)
	tensor2[0,0] = mu_2
	print('mu_2 = %.6f' %mu_2)	

	####### Computation of component I_sigma_sigma
	rho_sig = np.kron(rho, sigma_minus)
	sig_sig = np.kron(sigma_minus, sigma_minus)
	sigma_1 = (1/(2*variancia**2)) - (1/variancia**3)*(2*beta_cov*rho.sum() - (beta_cov**2)*sigma_minus.sum()) + (1/variancia**4)*(3*(beta_cov**2)*sum(np.kron(rho, rho)) - 3*(beta_cov**3)*rho_sig.sum() + 3*(beta_cov**4)*sig_sig.sum() )
	tensor1[1,1] = sigma_1
	print('sigma_1 = %.6f' %sigma_1)

	####### Computation of component II_sigma_sigma
	sigma_2 = (1/(2*variancia**2)) - (1/variancia**3)*(2*beta_cov*rho.sum() - (beta_cov**2)*sigma_minus.sum())
	tensor2[1,1] = sigma_2
	print('sigma_2 = %.6f' %sigma_2)

	####### Computational of component I_sigma_beta
	sigbeta_1 = (1/variancia**2)*(rho.sum() - beta_cov*sigma_minus.sum()) - (1/(2*variancia**3))*(6*beta_cov*sum(np.kron(rho, rho)) - 9*(beta_cov**2)*rho_sig.sum() + 3*(beta_cov**3)*sig_sig.sum() )
	tensor1[1,2] = sigbeta_1
	tensor1[2,1] = sigbeta_1
	print('sigbeta_1 = %.6f' %sigbeta_1)

	####### Computational of component II_sigma_beta
	sigbeta_2 = (1/variancia**2)*(rho.sum() - beta_cov*sigma_minus.sum())
	tensor2[1,2] = sigbeta_2
	tensor2[2,1] = sigbeta_2
	print('sigbeta_2 = %.6f' %sigbeta_2)

	####### Computation of component I_beta_beta (PHI)
	# First term
	T1 = (1/variancia)*sigma_minus.sum()
	# Second term
	T2 = (2/variancia**2)*sum(np.kron(rho, rho))
	# Third term
	T3 = -6*beta_cov*rho_sig.sum()/variancia**2
	# Fourth term
	T4 = 3*(beta_cov**2)*sig_sig.sum()/variancia**2

	phi = (T1+T2+T3+T4)
	tensor1[2,2] = phi
	print('PHI = %.6f' % phi)

	####### Computation of component II_beta_beta (PSI)
	psi = T1
	tensor2[2,2] = psi
	print('PSI = %.6f' % psi)

	###### Gaussian Curvature
	curvaturaG = -np.linalg.det(tensor2)/np.linalg.det(tensor1)
	print('Gaussian curvature = %.6f' %curvaturaG)

	# Shape operator
	shape_operator = np.dot(-tensor2, np.linalg.inv(tensor1))

	# Mean curvature
	curvaturaM = np.trace(shape_operator)
	print('Mean curvature = %.6f' %curvaturaM)

	# Principal curvatures
	v, w = np.linalg.eig(shape_operator)
	K1, K2, K3 = v[0], v[1], v[2]
	print('K1 = %.6f' %K1)
	print('K2 = %.6f' %K2)
	print('K3 = %.6f' %K3)

	####### Computation of the entropy
	entropia_gauss = 0.5*(np.log(2*np.pi) + np.log(variancia) + 1)
	entropia = entropia_gauss - ( (beta_cov/variancia)*rho.sum() - 0.5*(beta_cov**2)*psi )
	print('ENTROPY OF A GAUSSIAN R.V. = %.6f' %entropia_gauss)
	print('ENTROPY OF A GAUSSIAN RANDOM FIELD = %.6f' %entropia)

	# Stores the values of the measures at the current iteration
	vetor_media[iteracoes] = media_est
	vetor_variancia[iteracoes] = variancia
	vetor_betaMPL[iteracoes] = beta_cov
	vetor_phi[iteracoes] = phi
	vetor_psi[iteracoes] = psi
	vetor_ent_gauss[iteracoes] = entropia_gauss
	vetor_ent[iteracoes] = entropia
	vetor_mu1[iteracoes] = mu_1
	vetor_mu2[iteracoes] = mu_2
	vetor_sigma1[iteracoes] = sigma_1
	vetor_sigma2[iteracoes] = sigma_2
	vetor_sigbeta1[iteracoes] = sigbeta_1
	vetor_sigbeta2[iteracoes] = sigbeta_2
	vetor_curvaturaG[iteracoes] = curvaturaG
	vetor_curvaturaM[iteracoes] = curvaturaM
	vetor_K1[iteracoes] = K1
	vetor_K2[iteracoes] = K2
	vetor_K3[iteracoes] = K3

	# Gaussian curvature changing sign 
	if iteracoes > 20 and iteracoes < 300 and not OK1:
		if curvaturaG > 0:
			beta_negativo_positivo = beta_cov
			iteracao_negativo_positivo = iteracoes
			OK1 = True
			print('Gaussian curvature changes sign (- to +)')
			print(beta_cov)

	# Gaussian curvature changing sign
	if iteracoes > 500 and iteracoes < 950 and not OK2:
		if curvaturaG < 0:
			beta_positivo_negativo = beta_cov
			iteracao_positivo_negativo = iteracoes
			OK2 = True
			print('Mudança de sinal da curvatura Gausssiana (+ para -)')
			print(beta_cov)
			

	####### Computes the infinitesimal displacement (ds^2)
	dmu = vetor_media[iteracoes] - vetor_media[iteracoes-1]
	dsigma = vetor_variancia[iteracoes] - vetor_variancia[iteracoes-1]
	dbeta = vetor_betaMPL[iteracoes] - vetor_betaMPL[iteracoes-1]
	vetor_parametros = np.array([dmu, dsigma, dbeta])
	ds2_1 = np.dot(vetor_parametros, np.dot(tensor1, vetor_parametros))
	vetor_ds2_1[iteracoes] = np.sqrt(ds2_1)
	print('ds = %.15f' %np.sqrt(ds2_1))

    ##################################################################

	# Store the current system configuration
	sequence[iteracoes,:,:] = nucleo

	# Update the inverse temperature parameter
	print('beta = %.3f' % beta)
	
	if iteracoes <= MEIO:
		if beta < 0.3:			
			beta += 0.0006   	
	else:
		if beta > 0:
			beta -= 0.0006


###################################################

# Discard the first 5 samples (small burn in)
vetor_media = vetor_media[BURN_IN:]
vetor_variancia = vetor_variancia[BURN_IN:]
vetor_betaMPL = vetor_betaMPL[BURN_IN:]
vetor_phi = vetor_phi[BURN_IN:]
vetor_psi = vetor_psi[BURN_IN:]
vetor_ent_gauss = vetor_ent_gauss[BURN_IN:]
vetor_ent = vetor_ent[BURN_IN:]
vetor_mu1 = vetor_mu1[BURN_IN:]
vetor_mu2 = vetor_mu2[BURN_IN:]
vetor_sigma1 = vetor_sigma1[BURN_IN:]
vetor_sigma2 = vetor_sigma2[BURN_IN:]
vetor_sigbeta1 = vetor_sigbeta1[BURN_IN:]
vetor_sigbeta2 = vetor_sigbeta2[BURN_IN:]
vetor_ds2_1 = vetor_ds2_1[BURN_IN:]
vetor_curvaturaG = vetor_curvaturaG[BURN_IN:]
vetor_curvaturaM = vetor_curvaturaM[BURN_IN:]
vetor_K1 = vetor_K1[BURN_IN:]
vetor_K2 = vetor_K2[BURN_IN:]
vetor_K3 = vetor_K3[BURN_IN:]

print('\n*** END OF SIMULATION ***')

print('----- Elapsed time: %s seconds ----' % (time.time() - start))
print()

A = np.array([mediaA, varianciaA, betaA])
B = np.array([mediaB, varianciaB, betaB])
C = np.array([mediaC, varianciaC, betaC])

d_total = np.sum(vetor_ds2_1)
d_AB = np.sum(vetor_ds2_1[:MEIO])
d_BC = np.sum(vetor_ds2_1[MEIO:])

print('Model parameters in A (initial): ')
print('Mean A: %f' %mediaA)
print('Variance A: %f' %varianciaA)
print('Beta A: %f' %betaA)
print()
print('Model parameters in B (half): ')
print('Mean B: %f' %mediaB)
print('Variance B: %f' %varianciaB)
print('Beta B: %f' %betaB)
print()
print('Model parameters in C (final): ')
print('Mean: %f' %mediaC)
print('Variance C: %f' %varianciaC)
print('Beta C: %f' %betaC)
print()
print('Approximation to the total geodesic distance: %f' %d_total)
print('Approximation to the geodesic distance from A to B: %f' %d_AB)
print('Approximation to the geodesic distance from B to C: %f' %d_BC)
print('Symmetrized KL divergence from A to B: %f' %dKLsym_AB)
print('Symmetrized KL divergence from B to C: %f' %dKLsym_BC)
print('Euclidean distance from A to B: %f' %np.linalg.norm(A-B))
print('Euclidean distance from B to C: %f' %np.linalg.norm(B-C))


# Plot graphics and figures
plt.figure(1)
plt.plot(vetor_phi, 'b', label='PHI')
plt.xlabel('time')
plt.ylabel('First fundamental form - Beta')
plt.savefig('PHI.png')

plt.figure(2)
plt.plot(vetor_ent, 'k')
plt.xlabel('time')
plt.ylabel('Entropy')
plt.savefig('Entropy.png')

plt.figure(3)
plt.plot(vetor_phi[:METADE], vetor_ent[:METADE], 'b')
plt.plot(vetor_phi[METADE:], vetor_ent[METADE:], 'r')
plt.xlabel('Fisher information')
plt.ylabel('Entropy')
plt.savefig('Atrator_2D_PHI_Entropy.png')

plt.figure(4)
plt.plot(vetor_mu1[:METADE], vetor_ent[:METADE], 'b')
plt.plot(vetor_mu1[METADE:], vetor_ent[METADE:], 'r')
plt.xlabel('Fisher information (mean)')
plt.ylabel('Entropy')
plt.savefig('Mean_Entropy.png')

plt.figure(5)
plt.plot(vetor_sigma1[:METADE], vetor_ent[:METADE], 'b')
plt.plot(vetor_sigma1[METADE:], vetor_ent[METADE:], 'r')
plt.xlabel('Fisher information (sigma)')
plt.ylabel('Entropy')
plt.savefig('Sigma_Entropy.png')

plt.figure(6)
plt.plot(vetor_sigbeta1[:METADE], vetor_ent[:METADE], 'b')
plt.plot(vetor_sigbeta1[METADE:], vetor_ent[METADE:], 'r')
plt.xlabel('Fisher information (sigma/beta)')
plt.ylabel('Entropy')
plt.savefig('SigmaBeta_Entropy.png')

plt.figure(7)
plt.plot(vetor_ds2_1[5:], 'b')
plt.xlabel('time')
plt.ylabel('Infinitesimal displacements')
plt.savefig('DS.png')

plt.figure(8)
plt.plot(vetor_media, 'b')
plt.xlabel('time')
plt.ylabel('Estimated mean')
plt.savefig('Mean_Estimated.png')

plt.figure(9)
plt.plot(vetor_variancia, 'b')
plt.xlabel('time')
plt.ylabel('Estimated variance')
plt.savefig('Variance_Estimated.png')

plt.figure(10)
plt.plot(vetor_betaMPL, 'b')
plt.xlabel('time')
plt.ylabel('Inverse temperature')
plt.savefig('Inverse_Temperature.png')

plt.figure(11)
plt.plot(vetor_psi, 'r', label='PSI')
plt.xlabel('time')
plt.ylabel('Second fundamental form - Beta')
plt.savefig('PSI.png')

plt.figure(12)
plt.plot(vetor_curvaturaG, 'b', label='Gaussian curvature')
plt.plot(np.zeros(len(vetor_curvaturaG)), 'r--')
plt.xlabel('time')
plt.ylabel('Gaussian curvature')
plt.savefig('GC.png')

plt.figure(13)
plt.plot(vetor_curvaturaM, 'b', label='Mean curvature')
plt.xlabel('time')
plt.ylabel('Mean curvature')
plt.savefig('MC.png')

plt.figure(14)
plt.plot(vetor_K1, 'b', label='K1')
plt.xlabel('time')
plt.ylabel('Principal curvature 1 - K1')
plt.savefig('KC1.png')

plt.figure(15)
plt.plot(vetor_K2, 'b', label='K2')
plt.xlabel('time')
plt.ylabel('Principal curvature 2 - K2')
plt.savefig('KC2.png')

plt.figure(16)
plt.plot(vetor_K3, 'b', label='K3')
plt.xlabel('time')
plt.ylabel('Principal curvature 3 - K3')
plt.savefig('KC3.png')

plt.figure(17)
plt.plot(vetor_mu1, 'b', label='MU1')
plt.xlabel('time')
plt.ylabel('First fundamental form - Mean')
plt.savefig('First_Mean.png')

plt.figure(18)
plt.plot(vetor_mu2, 'r', label='MU2')
plt.xlabel('time')
plt.ylabel('Second fundamental form - Mean')
plt.savefig('Second_Mean.png')

plt.figure(19)
plt.plot(vetor_sigma1, 'b', label='SIGMA1')
plt.xlabel('time')
plt.ylabel('First fundamental form - Variance')
plt.savefig('First_Variance.png')

plt.figure(20)
plt.plot(vetor_sigma2, 'r', label='SIGMA2')
plt.xlabel('time')
plt.ylabel('Second fundamental form - Variance')
plt.savefig('Second_Variance.png')

plt.figure(21)
plt.plot(vetor_sigbeta1, 'b', label='SIGMA_BETA1')
plt.xlabel('time')
plt.ylabel('First fundamental form - Variance/Beta')
plt.savefig('First_Variance_Beta.png')

plt.figure(22)
plt.plot(vetor_sigbeta2, 'r', label='SIGMA_BETAr')
plt.xlabel('time')
plt.ylabel('Second fundamental form - Variance/Beta')
plt.savefig('Second_Variance_Beta.png')

plt.figure(23)
plt.plot(vetor_curvaturaG[:METADE], vetor_ent[:METADE], 'b')
plt.plot(vetor_curvaturaG[METADE:], vetor_ent[METADE:], 'r')
plt.xlabel('Gaussian curvature')
plt.ylabel('Entropy')
plt.savefig('GC_Entropy.png')

plt.figure(24)
plt.plot(vetor_curvaturaM[:METADE], vetor_ent[:METADE], 'b')
plt.plot(vetor_curvaturaM[METADE:], vetor_ent[METADE:], 'r')
plt.xlabel('Mean curvature')
plt.ylabel('Entropy')
plt.savefig('MC_Entropy.png')

# Initial configuration
imgA = sequence[10, :, :]
saidaA = np.uint8(255*(imgA - imgA.min())/(imgA.max() - imgA.min()))
plt.imsave('A.png', saidaA, cmap=cm.jet)

# Final configuration
imgB = sequence[METADE, :, :]
saidaB = np.uint8(255*(imgB - imgB.min())/(imgB.max() - imgB.min()))
plt.imsave('B.png', saidaB, cmap=cm.jet)

plt.clf()
plt.close('all')