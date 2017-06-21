import theano
import numpy as np
from theano import tensor as T
from collections import OrderedDict


# TODO: ???
# theano.config.floatX = 'float32'


def make_vector(shape):
    return np.zeros(shape, dtype=theano.config.floatX)


def make_matrix(shape):
    return np.random.normal(scale=0.1, size=shape).astype(theano.config.floatX)


class ChildSumLSTM:
    def __init__(self, hdim, edim, learning_rate=0.01, momentum=0.9):
        self.hdim = hdim
        self.edim = edim
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.recursive_unit = self.make_leaf_unit()
        self.leaf_unit = self.make_leaf_unit()

    def make_rec_unit(self):
        """ Создаем юнит для нерутовой ноды.
        
        :return: theano unit
        """
        W_i = theano.shared(make_matrix((self.hdim, self.edim)))
        U_i = theano.shared(make_matrix((self.hdim, self.hdim)))
        b_i = theano.shared(make_vector(self.hdim))
        W_f = theano.shared(make_matrix((self.hdim, self.edim)))
        U_f = theano.shared(make_matrix((self.hdim, self.hdim)))
        b_f = theano.shared(make_vector(self.hdim))
        W_o = theano.shared(make_matrix((self.hdim, self.edim)))
        U_o = theano.shared(make_matrix((self.hdim, self.hdim)))
        b_o = theano.shared(make_vector(self.hdim))
        W_u = theano.shared(make_matrix((self.hdim, self.edim)))
        U_u = theano.shared(make_matrix((self.hdim, self.hdim)))
        b_u = theano.shared(make_vector(self.hdim))
        self.params.extend([
            W_i, U_i, b_i,
            W_f, U_f, b_f,
            W_o, U_o, b_o,
            W_u, U_u, b_u
        ])

        def unit(parent_x, child_h, child_c, child_exists):
            """ Функция для пересчета. Все, как в статье.

            :param parent_x: векторизация родителя
            :param child_h: h детей, shape=[self.degree, self.hdim]
            :param child_c: c детей, shape=[self.degree, self.hdim]
            :param child_exists: 1 если ребенок есть, 0 иначе, shape=[self.degree]
            :return: h и с для родителя
            """
            h_tilde = T.sum(child_h, axis=0)
            i = T.nnet.sigmoid(T.dot(W_i, parent_x) + T.dot(U_i, h_tilde) + b_i)
            o = T.nnet.sigmoid(T.dot(W_o, parent_x) + T.dot(U_o, h_tilde) + b_o)
            u = T.tanh(T.dot(W_u, parent_x) + T.dot(U_u, h_tilde) + b_u)
            f = T.nnet.sigmoid(T.dot(W_f, parent_x).dimshuffle('x', 0)
                               + T.dot(child_h, U_f)
                               + b_f.domshuffle('x', 0)) * child_exists.dimshuffle(0, 'x')
            c = i * u + T.sum(f * child_c, axis=0)
            h = o * T.tanh(c)
            return h, c

        return unit

    def make_leaf_unit(self):
        dummy = 0 * theano.shared(make_vector([self.degree, self.hdim]))

        def unit(leaf_x):
            """ Функция для пересчета. Просто передаем нули в функцию, которую создаем выше.
            
            :param leaf_x: векторизация листа
            :return: h, c для родителя
            """
            return self.recursive_unit(leaf_x, dummy, dummy, dummy.sum(axis=1))

        return unit

    def compute_tree(self, emb_x, tree):
        """ Создаем вычислительный граф.

        :param emb_x: получем по номеру векторизацию вершины 
        :param tree: список детей по номеру
        :return: список внутренних векторов
        """
        num_nodes = tree.shape[0]

    def gradient_descent(self, loss):
        grad = T.grad(loss, self.params)
        momentum_velocity = [0.] * len(grad)
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grad)))
        updates = OrderedDict()
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        scaling_den = T.maximum(5.0, grad_norm)
        for n, (param, grad) in enumerate(zip(self.params, grad)):
            grad = T.switch(not_finite, 0.1 * param, grad * (5.0 / scaling_den))
            velocity = momentum_velocity[n]
            update_step = self.momentum * velocity - self.learning_rate * grad
            momentum_velocity[n] = update_step
            updates[param] = param + update_step
        return updates
