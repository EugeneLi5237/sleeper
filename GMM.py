from sklearn.mixture import GaussianMixture
'''
1.train a two Gaussian mixture with data preloaded to the instance
2.save the model to the instance
note: data should be filtered prior to training
'''
def train(self):
    gmm = GaussianMixture(n_components = 2)
    ir_tem =np.asarray(self.__data_buffer)
    self.__model = gmm.fit(ir_tem.reshape(-1,1))
    return
'''
1. plot the GaussianMixture with pdf
'''
def plt_hist(self):
      fit  = self.__model
      mu1 = fit.means_[0, 0]
      mu2 = fit.means_[1, 0]
      var1, var2 = fit.covariances_
      wgt1, wgt2 = fit.weights_
      gmm_train_ir = self.__data_buffer
      x = np.linspace(min(ir_tem),max(ir_tem),1000)
      norm_1 = wgt1 * norm.pdf(np.reshape(x,[1000,1]),loc = mu1, scale = np.sqrt(var1))
      norm_2 = wgt2 * norm.pdf(np.reshape(x,[1000,1]),loc = mu2, scale = np.sqrt(var2))
      plt.figure()
      plt.hist(ir_tem,bins=50,density =True)
      plt.plot(x, norm_1,'r-')
      plt.plot(x, norm_2,'b-')
      return

def plt_labels(self):
    fit = self.__model
    gmm_train_t = self.__time_buffer
    ir_tem =np.asarray(self.__data_buffer)
    pred = fit.predict(ir_tem.reshape(-1,1))
    max_ir = max(ir_tem)
    min_ir = min(ir_tem)
    rescale =[(x*4)/(max_ir-min_ir) for x in  ir_tem ]
    plt.figure()
    plt.plot(gmm_train_t,rescale)
    plt.plot(gmm_train_t,pred)
    plt.title("GMM Labels")
    plt.show(block = False)
    return
