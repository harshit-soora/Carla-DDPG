''' Utility module containing 
functions for plotting etc
'''
import matplotlib.pyplot as plt


''' Utility function to plot loss
input : Actor-Critic Loss
'''
def plot(loss, x_label, y_label):
	plt.plot([i+1 for i in range(0, len(loss), 2)], loss[::2])
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.show()
