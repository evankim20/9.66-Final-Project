import numpy as np

class RSA:
    """
    Inputs: 
    - referents: np.array (1 x r)
    - murmurs:   np.array (m x 1)
    - lexicon:   np.array (m x r) 
    - prior:     np.array (1 x r)
    - cost:      np.array (m x 1)
    - alpha:     float

    All returned prob distribtuions will be in the shape (m x r) with the referents as the columns and murmurs as the rows
    """
    def __init__(self, referents, murmurs, lexicon, prior, cost, alpha=1):
        self.referents = referents
        self.murmurs = murmurs
        self.lexicon = lexicon
        self.prior = prior
        self.cost = cost
        self.alpha = alpha

    def literal_listener(self):
        p_lit = np.zeros((self.murmurs.shape[0], self.referents.shape[1]))
        for i in range(self.murmurs.shape[0]):
            margin = np.dot(self.prior, self.lexicon[i])
            p_lit[i] = np.divide(np.multiply(self.lexicon[i], self.prior), margin)
        return p_lit

    def pragmatic_speaker(self):
        p_lit = self.literal_listener()
        p_s = np.zeros((self.murmurs.shape[0], self.referents.shape[1]))
        for j in range(self.referents.shape[1]):
            with np.errstate(divide='ignore'):
                log_p_lit =  np.log(p_lit[:,[j]])
            exp_col = np.exp(np.multiply(log_p_lit +  self.cost, self.alpha))
            margin = np.sum(exp_col)
            p_s[:,[j]] = np.divide(exp_col, margin)
        return p_s
            
    def pragmatic_listener(self):
        p_s = self.pragmatic_speaker()
        p_l = np.zeros((self.murmurs.shape[0], self.referents.shape[1]))
        for i in range(self.murmurs.shape[0]):
            margin = np.dot(p_s[i].reshape(1, self.referents.shape[1]), self.prior.T)
            p_l[i] = np.divide(np.multiply(p_s[i], self.prior), margin)
        return p_l
            
referents = np.array([['r1', 'r2']])
murmurs = np.transpose(np.array([['hat', 'glasses']]))
lexicon = np.array([[0, 1], [1, 1]])
prior = np.array([[0.5, 0.5]])
cost = np.transpose(np.array([[-6, 0]]))
alpha = 1
rsa_obj = RSA(referents, murmurs, lexicon, prior, cost, alpha)
print("LITERAL LISTENER:", rsa_obj.literal_listener())
print("PRAGMATIC SPEAKER:", rsa_obj.pragmatic_speaker())
print("PRAGMATIC LISTENTER:", rsa_obj.pragmatic_listener())