''' Simulação de Monte Carlo via Metropolis para campos aleatórios Markovianos Gaussianos'''

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

#####################################################
############### Início do script ####################
#####################################################

start = time.time()

MAX_IT = 1000      						# número máximo de iterações
MEIO = MAX_IT//2
SIZE = 256								# dimensões do reticulado que define campo
METADE = MEIO
BURN_IN = 5

# Valor inicial do parametro beta
beta = 0				# estava 0
# Média
media = 0				#estava 0 - é a média usada para gerar as ocorrências (incrementada no fim da iteração)
# Variância (usava sigma = 5)
sigma = 5					# estava 5

# Armazenar as matrizes de informação de Fisher a cada iteração
tensor1 = np.zeros((3,3))
tensor2 = np.zeros((3,3))

# Cria campo aleatório inicial (amostras iid gaussianas)
img = np.random.normal(media, sigma, (SIZE, SIZE))

# Cria pilha de imagens para armazenar configurações do campo no tempo
sequence = np.zeros((MAX_IT, img.shape[0], img.shape[1]))

# Espelha bordas (boundary value problem)
K = 1
img = np.lib.pad(img, ((K,K), (K,K)), 'symmetric')

nlin, ncol = (img.shape[0], img.shape[1])

# Estima o parâmetro beta na imagem inicial
numerador = 0
denominador = 0

# Para armazenar estatísticas fundamentais em cada iteração
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

col = 9
centro = col//2		# centro da matriz de covarâncias
delta = 8     		# número de vizinhos

for i in range(K, nlin-K):
	for j in range(K, ncol-K):

		neigh = img[i-K:i+K+1, j-K:j+K+1]
		neigh = np.reshape(neigh, neigh.shape[0]*neigh.shape[1])
		vizinhanca = np.concatenate((neigh[0:(neigh.size//2)], neigh[(neigh.size//2)+1:neigh.size]))
		central = neigh[neigh.size//2]
		numerador += (central - media)*sum(vizinhanca - media)
		denominador += sum(vizinhanca - media)**2

beta_inicial = numerador/denominador
print('beta inicial estimado = ', beta_inicial)

sample = img.copy()

OK1, OK2 = False, False

############## Loop Principal ####################
for iteracoes in range(0, MAX_IT):

	# Para obter as amostras (patches) para computar a matriz de covariâncias
	amostras = np.zeros(((nlin-2)*(ncol-2), col))
	ind = 0

	print('\nIteração ', iteracoes)
	# Alterar no caso de vizinhança de 3, 4 e 5 ordens
	for i in range(K, nlin-K):
		for j in range(K, ncol-K):

			neigh = img[i-1:i+2, j-1:j+2]
			neigh = np.reshape(neigh, neigh.shape[0]*neigh.shape[1])
			amostras[ind,:] = neigh
			ind += 1
			vizinhanca = np.concatenate((neigh[0:(neigh.size//2)], neigh[(neigh.size//2)+1:neigh.size]))
			central = neigh[neigh.size//2]
			
			# Calcula a probabilidade para valor atual de xi
			P1 = (1/np.sqrt(2*np.pi*sigma))*np.exp((-1/(2*sigma))*(central - media - beta*sum(vizinhanca - media))**2)
			# Escolher um novo rótulo g aleatoriamente
			# Número aleatório sorteado a partir de uma distribuição normal
			g = np.random.normal(media, sigma)
			while (g < media - 3*np.sqrt(sigma)) or (g > media + 3*np.sqrt(sigma)):	# estava 5
				g = np.random.normal(media, sigma)
			# Calcula probabilidade do novo valor g
			P2 = (1/np.sqrt(2*np.pi*sigma))*np.exp((-1/(2*sigma))*(g - media - beta*(vizinhanca - media).sum())**2)
			#Escolher o menor entre 1 e a razão entre P2 e P1
			limiar = 1
			razao = P2/P1
			if (razao < 1):
				limiar = razao
			# Aceita novo valor com probabilidade p
			epson = np.random.rand()
			if epson <= limiar:
				sample[i,j] = g

	img = sample.copy()
	nucleo = img[K:nlin-K, K:ncol-K]

	media_est = nucleo.mean()	# é a média estimada
	variancia = nucleo.var()

	############# Calcular matriz de informação de Fisher ############
	mc = np.cov(amostras.T)			# matriz de covariâncias

	# Computa a matriz sem a linha e a coluna central (Sigma_{p}^{-})
	sigma_minus = mc.copy()
	sigma_minus[:,centro] = 0
	sigma_minus[centro,:] = 0

	# Estima o inverso da temperatura via MPL (beta)
	left_half = mc[centro, 0:centro]
	right_half = mc[centro, centro+1:col]
	rho = np.concatenate((left_half, right_half))
	
	#### Beta estimado x Beta real
	beta_cov = beta 	# usando o parâmetro real para testes!
	
	print('beta = %.6f' % beta_cov)
	print('média = %.6f' % media_est)
	print('variância = %.6f' % variancia)

	# Pega o ponto A na iteração 10, o ponto B na iteração 550 e o ponto C na iteração 999
	if iteracoes == BURN_IN:
		mediaA, varianciaA, betaA = media_est, variancia, beta_cov
		rho_sum_A = rho.sum()
		sigma_minus_sum_A = sigma_minus.sum()
	elif iteracoes == MEIO:
		mediaB, varianciaB, betaB = media_est, variancia, beta_cov
		rho_sum_B = rho.sum()
		sigma_minus_sum_B = sigma_minus.sum()
		dKL_AB = 0.5*log(varianciaB/varianciaA) - (1/(2*varianciaA))*( varianciaA - 2*betaA*rho_sum_A + (betaA**2)*sigma_minus_sum_A  ) + (1/(2*varianciaB))*( (varianciaA - 2*betaB*rho_sum_A + (betaB**2)*sigma_minus_sum_A) + ((mediaA - mediaB)**2)*(1 - delta*betaB)**2 )
		dKLsym_AB = (1/(4*varianciaA*varianciaB))*( (varianciaA - varianciaB)**2 - 2*(betaB*varianciaA - betaA*varianciaB)*(rho_sum_A - rho_sum_B) + (betaB**2 * varianciaA - betaA**2 * varianciaB)*(sigma_minus_sum_A - sigma_minus_sum_B) + (mediaA - mediaB)**2 * (varianciaA*(1 - delta*betaB)**2 + varianciaB*(1 - delta*betaA)**2 ) )
		print()		
		print('dKL(A, B) = %f' %dKL_AB)
		print()
		print('dKLsym(A, B) = %f' %dKLsym_AB)
		print()
		#input()
	elif iteracoes == MAX_IT - 1:
		mediaC, varianciaC, betaC = media, variancia, beta_cov
		rho_sum_C = rho.sum()
		sigma_minus_sum_C = sigma_minus.sum()
		dKL_BC = 0.5*log(varianciaC/varianciaB) - (1/(2*varianciaB))*( varianciaB - 2*betaB*rho_sum_B + (betaB**2)*sigma_minus_sum_B  ) + (1/(2*varianciaC))*( (varianciaB - 2*betaC*rho_sum_B + (betaC**2)*sigma_minus_sum_B) + ((mediaB - mediaC)**2)*(1 - delta*betaC)**2 )
		dKLsym_BC = (1/(4*varianciaB*varianciaC))*( (varianciaB - varianciaC)**2 - 2*(betaC*varianciaB - betaB*varianciaC)*(rho_sum_B - rho_sum_C) + (betaC**2 * varianciaB - betaB**2 * varianciaC)*(sigma_minus_sum_B - sigma_minus_sum_C) + (mediaB - mediaC)**2 * (varianciaB*(1 - delta*betaC)**2 + varianciaC*(1 - delta*betaB)**2 ) )
		print()		
		print('dKL(B, C) = %f' %dKL_BC)
		print()
		print('dKLsym(B, c) = %f' %dKLsym_BC)
		print()
		#input()

	######## ESTIMA MATRIZES DE INFORMAÇÃO DE FISHER

	####### Cálculo do componente I_mu_mu^{1}	
	mu_1 = (1/variancia)*((1 - beta_cov*delta)**2)*(1 - (1/variancia)*(2*beta_cov*rho.sum() - (beta_cov**2)*sigma_minus.sum()))
	tensor1[0,0] = mu_1
	print('mu_1 = %.6f' %mu_1)	

	####### Cálculo do componente I_mu_mu^{2}	
	mu_2 = (1/variancia)*((1 - beta_cov*delta)**2)
	tensor2[0,0] = mu_2
	print('mu_2 = %.6f' %mu_2)	

	####### Cálculo do componente I_sigma_sigma^{1}
	rho_sig = np.kron(rho, sigma_minus)
	sig_sig = np.kron(sigma_minus, sigma_minus)
	sigma_1 = (1/(2*variancia**2)) - (1/variancia**3)*(2*beta_cov*rho.sum() - (beta_cov**2)*sigma_minus.sum()) + (1/variancia**4)*(3*(beta_cov**2)*sum(np.kron(rho, rho)) - 3*(beta_cov**3)*rho_sig.sum() + 3*(beta_cov**4)*sig_sig.sum() )
	tensor1[1,1] = sigma_1
	print('sigma_1 = %.6f' %sigma_1)

	####### Cálculo do componente I_sigma_sigma^{2}
	sigma_2 = (1/(2*variancia**2)) - (1/variancia**3)*(2*beta_cov*rho.sum() - (beta_cov**2)*sigma_minus.sum())
	tensor2[1,1] = sigma_2
	print('sigma_2 = %.6f' %sigma_2)

	####### Cálculo do componente I_sigma_beta^{1}
	sigbeta_1 = (1/variancia**2)*(rho.sum() - beta_cov*sigma_minus.sum()) - (1/(2*variancia**3))*(6*beta_cov*sum(np.kron(rho, rho)) - 9*(beta_cov**2)*rho_sig.sum() + 3*(beta_cov**3)*sig_sig.sum() )
	tensor1[1,2] = sigbeta_1
	tensor1[2,1] = sigbeta_1
	print('sigbeta_1 = %.6f' %sigbeta_1)

	####### Cálculo do componente I_sigma_beta^{2}
	sigbeta_2 = (1/variancia**2)*(rho.sum() - beta_cov*sigma_minus.sum())
	tensor2[1,2] = sigbeta_2
	tensor2[2,1] = sigbeta_2
	print('sigbeta_2 = %.6f' %sigbeta_2)

	####### Cálculo do componente I_beta_beta^{1} ou simplesmente Phi
	# Primeiro termo
	T1 = (1/variancia)*sigma_minus.sum()
	# Segundo termo
	T2 = (2/variancia**2)*sum(np.kron(rho, rho))
	# Terceiro termo
	T3 = -6*beta_cov*rho_sig.sum()/variancia**2
	# Quarto termo
	T4 = 3*(beta_cov**2)*sig_sig.sum()/variancia**2

	phi = (T1+T2+T3+T4)
	tensor1[2,2] = phi
	print('PHI = %.6f' % phi)

	####### Cálculo do componente I_beta_beta^{2} ou simplesmente Psi
	psi = T1
	tensor2[2,2] = psi
	print('PSI = %.6f' % psi)

	###### Curvatura Gaussiana #####
	curvaturaG = -np.linalg.det(tensor2)/np.linalg.det(tensor1)
	print('Curvatura Gaussiana = %.6f' %curvaturaG)

	# Shape operator
	shape_operator = np.dot(-tensor2, np.linalg.inv(tensor1))

	# Mean curvature
	curvaturaM = np.trace(shape_operator)
	print('Curvatura Média = %.6f' %curvaturaM)

	# Principal curvatures
	v, w = np.linalg.eig(shape_operator)
	K1, K2, K3 = v[0], v[1], v[2]
	print('K1 = %.6f' %K1)
	print('K2 = %.6f' %K2)
	print('K3 = %.6f' %K3)

	####### Cálculo da entropia de Shannon
	entropia_gauss = 0.5*(np.log(2*np.pi) + np.log(variancia) + 1)
	entropia = entropia_gauss - ( (beta_cov/variancia)*rho.sum() - 0.5*(beta_cov**2)*psi )
	print('ENTROPIA GAUSSIANA = %.6f' %entropia_gauss)
	print('ENTROPIA GMRF = %.6f' %entropia)

	# Armazena estatísticas calculadas em cada iteração
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

	# Pontos em que curvatura troca de sinal	
	if iteracoes > 20 and iteracoes < 300 and not OK1:
		if curvaturaG > 0:
			beta_negativo_positivo = beta_cov
			iteracao_negativo_positivo = iteracoes
			OK1 = True
			print('Mudança de sinal da curvatura Gausssiana (- para +)')					# beta = 0.178 	betaMPL = 0.1788		iteracao = 297
			print(beta_cov)
			#input()
	# Pontos em que curvatura troca de sinal
	if iteracoes > 500 and iteracoes < 950 and not OK2:
		if curvaturaG < 0:
			beta_positivo_negativo = beta_cov
			iteracao_positivo_negativo = iteracoes
			OK2 = True
			print('Mudança de sinal da curvatura Gausssiana (+ para -)')					# beta = 0.148		betaMPL = 0.1482		iteracao = 755
			print(beta_cov)
			#input()

	####### Calcula deslocamento infinitesimal ds2
	dmu = vetor_media[iteracoes] - vetor_media[iteracoes-1]
	dsigma = vetor_variancia[iteracoes] - vetor_variancia[iteracoes-1]
	dbeta = vetor_betaMPL[iteracoes] - vetor_betaMPL[iteracoes-1]
	vetor_parametros = np.array([dmu, dsigma, dbeta])
	ds2_1 = np.dot(vetor_parametros, np.dot(tensor1, vetor_parametros))
	vetor_ds2_1[iteracoes] = np.sqrt(ds2_1)
	print('ds (tensor1) = %.15f' %np.sqrt(ds2_1))

    ##################################################################

	# Armazena sequencia de imagens
	sequence[iteracoes,:,:] = nucleo

	################# Modulando o comportamento do sistema (original) ###########
	print('beta real = %.3f' % beta)
	
	# Com beta real
	if iteracoes <= MEIO:
		if beta < 0.3:			# original é 0.8
			beta += 0.0006   	# original é 0.002
	else:
		if beta > 0:
			beta -= 0.0006

###################################################

# Descarta primeiras amostras
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

print('\n*** FIM DA SIMULAÇÃO ***')

print('----- Tempo total de execução: %s segundos ----' % (time.time() - start))
print()

A = np.array([mediaA, varianciaA, betaA])
B = np.array([mediaB, varianciaB, betaB])
C = np.array([mediaC, varianciaC, betaC])

d_total = np.sum(vetor_ds2_1)
d_AB = np.sum(vetor_ds2_1[:MEIO])
d_BC = np.sum(vetor_ds2_1[MEIO:])

print('Parâmetros estimados do modelo em A: ')
print('Média A: %f' %mediaA)
print('Variância A: %f' %varianciaA)
print('Beta A: %f' %betaA)
print()
print('Parâmetros estimados do modelo em B: ')
print('Média B: %f' %mediaB)
print('Variância B: %f' %varianciaB)
print('Beta B: %f' %betaB)
print()
print('Parâmetros estimados do modelo em C: ')
print('Média C: %f' %mediaC)
print('Variância C: %f' %varianciaC)
print('Beta C: %f' %betaC)
print()
print('Distância geodésica total: %f' %d_total)
print('Distância geodésica de A a B: %f' %d_AB)
print('Distância geodésica de B a C: %f' %d_BC)
print('Divergência KL de A a B: %f' %dKL_AB)
print('Divergência KL de B a C: %f' %dKL_BC)
print('Divergência KL simetrizada de A a B: %f' %dKLsym_AB)
print('Divergência KL simetrizada de B a C: %f' %dKLsym_BC)
print('Distância Euclidiana de A a B: %f' %np.linalg.norm(A-B))
print('Distância Euclidiana de B a C: %f' %np.linalg.norm(B-C))

# Plota PHI e PSI juntos
plt.figure(1)
plt.plot(vetor_phi, 'b', label='PHI')
plt.xlabel('time')
plt.ylabel('First fundamental form - Beta')
plt.savefig('PHI.png')

# Plota a entropia
plt.figure(2)
plt.plot(vetor_ent, 'k')
#plt.axis([0, 1000, 2.1, 2.62])
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

# Deslocalmentos infinitesimais
plt.figure(7)
plt.plot(vetor_ds2_1[5:], 'b')
plt.xlabel('time')
plt.ylabel('Infinitesimal displacements')
plt.savefig('DS.png')

# Plota parâmetro média
plt.figure(8)
plt.plot(vetor_media, 'b')
plt.xlabel('time')
plt.ylabel('Estimated mean')
plt.savefig('Mean_Estimated.png')

# Plota parâmetro variância
plt.figure(9)
plt.plot(vetor_variancia, 'b')
plt.xlabel('time')
plt.ylabel('Estimated variance')
#plt.legend()
plt.savefig('Variance_Estimated.png')

# Plota parâmetro Beta
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

# Plota as configurações A (10) e B (550)
imgA = sequence[10, :, :]
saidaA = np.uint8(255*(imgA - imgA.min())/(imgA.max() - imgA.min()))
plt.imsave('A.png', saidaA, cmap=cm.jet)
#imsave('A.png', saidaA)

imgB = sequence[METADE, :, :]
saidaB = np.uint8(255*(imgB - imgB.min())/(imgB.max() - imgB.min()))
plt.imsave('B.png', saidaB, cmap=cm.jet)
#imsave('B.png', saidaB)

plt.clf()
plt.close('all')