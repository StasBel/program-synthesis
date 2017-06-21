import numpy as np
import theano
from theano import tensor as T
from collections import OrderedDict
from lstm.tree_rnn import gen_nn_inputs

theano.config.floatX = 'float32'


class SemanticLSTM:
    """Data is represented in a tree structure.
        Every leaf and internal node has a data (provided by the input)
        and a memory or hidden state.  The hidden state is computed based
        on its own data and the hidden states of its children.  The
        hidden state of leaves is given by a custom init function.
        The entire tree's embedding is represented by the final
        state computed at the root.
        """

    def __init__(self, num_emb, emb_dim, hidden_dim, K=2,
                 degree=2, learning_rate=0.01, momentum=0.9,
                 trainable_embeddings=True, irregular_tree=False):
        assert emb_dim > 1 and hidden_dim > 1
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.K = K
        self.degree = degree
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.irregular_tree = irregular_tree
        self.params = []
        self.embeddings = theano.shared(self.init_matrix([self.num_emb, self.emb_dim]))
        if trainable_embeddings:
            self.params.append(self.embeddings)

        self.y = T.scalar(name='y')

        self.x1 = T.ivector(name='x1')  # word indices
        self.tree1 = T.imatrix(name='tree1')  # shape [None, self.degree]
        self.x2 = T.ivector(name='x2')  # word indices
        self.tree2 = T.imatrix(name='tree2')  # shape [None, self.degree]

        self.num_words1 = self.x1.shape[0]
        emb_x1 = self.embeddings[self.x1]
        emb_x1 = emb_x1 * T.neq(self.x1, -1).dimshuffle(0, 'x')  # zero-out non-existent embeddings
        self.num_words2 = self.x2.shape[0]
        emb_x2 = self.embeddings[self.x2]
        emb_x2 = emb_x2 * T.neq(self.x2, -1).dimshuffle(0, 'x')  # zero-out non-existent embeddings

        self.tree_states1 = self.compute_tree(emb_x1, self.tree1, self.num_words1)
        self.tree_states2 = self.compute_tree(emb_x2, self.tree2, self.num_words2)

        self.final_state1 = self.tree_states1[-1]
        self.final_state2 = self.tree_states2[-1]

        self.output_fn = self.create_output_fn()
        self.pred_y = self.output_fn(self.final_state1, self.final_state2)
        self.loss = self.loss_fn(self.y, self.pred_y)

        updates = self.gradient_descent(self.loss)

        self._train = theano.function([self.x1, self.tree1, self.x2, self.tree2, self.y],
                                      [self.loss, self.pred_y],
                                      updates=updates)

        self._predict = theano.function([self.x1, self.tree1, self.x2, self.tree2],
                                        self.pred_y)

    def init_matrix(self, shape):
        return np.random.normal(scale=0.1, size=shape).astype(theano.config.floatX)

    def init_vector(self, shape):
        return np.zeros(shape, dtype=theano.config.floatX)

    def train_step_inner(self, x1, tree1, x2, tree2, y):
        return self._train(x1, tree1[:, :-1], x2, tree2[:, :-1], y)

    def train_step(self, root_node1, root_node2, y):
        x1, tree1 = gen_nn_inputs(root_node1, max_degree=self.degree, only_leaves_have_vals=False)
        x2, tree2 = gen_nn_inputs(root_node2, max_degree=self.degree, only_leaves_have_vals=False)
        return self.train_step_inner(x1, tree1, x2, tree2, y)

    def predict(self, root_node1, root_node2):
        x1, tree1 = gen_nn_inputs(root_node1, max_degree=self.degree, only_leaves_have_vals=False)
        x2, tree2 = gen_nn_inputs(root_node2, max_degree=self.degree, only_leaves_have_vals=False)
        return self._predict(x1, tree1[:, :-1], x2, tree2[:, :-1])

    def create_output_fn(self):
        self.W_A = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.W_D = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_h = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_p = theano.shared(self.init_matrix([self.K, self.hidden_dim]))
        self.b_p = theano.shared(self.init_vector([self.K]))
        self.params.extend([self.W_A, self.W_D, self.b_h, self.W_p, self.b_p])

        def fn(hL, hR):
            hA = hL * hR
            hD = abs(hL - hR)
            hS = T.nnet.sigmoid(T.dot(self.W_A, hA) + T.dot(self.W_D, hD) + self.b_h)
            p = T.nnet.softmax(T.dot(self.W_p, hS) + self.b_p)
            return T.sum(np.arange(1, self.K + 1, dtype=theano.config.floatX) * p)

        return fn

    def loss_fn(self, y, pred_y):
        return T.sum(T.sqr(y - pred_y))

    def gradient_descent(self, loss):
        """Momentum GD with gradient clipping."""
        grad = T.grad(loss, self.params)
        self.momentum_velocity_ = [0.] * len(grad)
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grad)))
        updates = OrderedDict()
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        scaling_den = T.maximum(5.0, grad_norm)
        for n, (param, grad) in enumerate(zip(self.params, grad)):
            grad = T.switch(not_finite, 0.1 * param,
                            grad * (5.0 / scaling_den))
            velocity = self.momentum_velocity_[n]
            update_step = self.momentum * velocity - self.learning_rate * grad
            self.momentum_velocity_[n] = update_step
            updates[param] = param + update_step
        return updates

    def create_recursive_unit(self):
        self.W_i = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_i = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_i = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_f = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_f = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_f = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_o = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_o = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_o = theano.shared(self.init_vector([self.hidden_dim]))
        self.W_u = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.U_u = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.b_u = theano.shared(self.init_vector([self.hidden_dim]))
        self.params.extend([
            self.W_i, self.U_i, self.b_i,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
            self.W_u, self.U_u, self.b_u])

        def unit(parent_x, child_h, child_c, child_exists):
            h_tilde = T.sum(child_h, axis=0)
            i = T.nnet.sigmoid(T.dot(self.W_i, parent_x) + T.dot(self.U_i, h_tilde) + self.b_i)
            o = T.nnet.sigmoid(T.dot(self.W_o, parent_x) + T.dot(self.U_o, h_tilde) + self.b_o)
            u = T.tanh(T.dot(self.W_u, parent_x) + T.dot(self.U_u, h_tilde) + self.b_u)

            f = (T.nnet.sigmoid(
                T.dot(self.W_f, parent_x).dimshuffle('x', 0) +
                T.dot(child_h, self.U_f.T) +
                self.b_f.dimshuffle('x', 0)) *
                 child_exists.dimshuffle(0, 'x'))

            c = i * u + T.sum(f * child_c, axis=0)
            h = o * T.tanh(c)
            return h, c

        return unit

    def create_leaf_unit(self):
        dummy = 0 * theano.shared(self.init_vector([self.degree, self.hidden_dim]))

        def unit(leaf_x):
            return self.recursive_unit(
                leaf_x,
                dummy,
                dummy,
                dummy.sum(axis=1))

        return unit

    def compute_tree(self, emb_x, tree, num_words):
        self.recursive_unit = self.create_recursive_unit()
        self.leaf_unit = self.create_leaf_unit()
        num_nodes = tree.shape[0]  # num internal nodes
        num_leaves = num_words - num_nodes

        # compute leaf hidden states
        (leaf_h, leaf_c), _ = theano.map(
            fn=self.leaf_unit,
            sequences=[emb_x[:num_leaves]])
        if self.irregular_tree:
            init_node_h = T.concatenate([leaf_h, leaf_h], axis=0)
            init_node_c = T.concatenate([leaf_c, leaf_c], axis=0)
        else:
            init_node_h = leaf_h
            init_node_c = leaf_c

        # use recurrence to compute internal node hidden states
        def _recurrence(cur_emb, node_info, t, node_h, node_c, last_h):
            child_exists = node_info > -1
            offset = num_leaves * int(self.irregular_tree) - child_exists * t
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x')
            child_c = node_c[node_info + offset] * child_exists.dimshuffle(0, 'x')
            parent_h, parent_c = self.recursive_unit(cur_emb, child_h, child_c, child_exists)
            node_h = T.concatenate([node_h,
                                    parent_h.reshape([1, self.hidden_dim])])
            node_c = T.concatenate([node_c,
                                    parent_c.reshape([1, self.hidden_dim])])
            return node_h[1:], node_c[1:], parent_h

        dummy = theano.shared(self.init_vector([self.hidden_dim]))
        (_, _, parent_h), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, init_node_c, dummy],
            sequences=[emb_x[num_leaves:], tree, T.arange(num_nodes)],
            n_steps=num_nodes)

        return T.concatenate([leaf_h, parent_h], axis=0)

    def get_state(self):
        return self.embeddings, self.W_A, self.W_D, self.b_h, self.W_p, \
               self.b_p, self.W_i, self.U_i, self.b_i, self.W_f, self.U_f, \
               self.b_f, self.W_o, self.U_o, self.b_o, self.W_u, self.U_u, self.b_u

    def set_state(self, state):
        self.embeddings, self.W_A, self.W_D, self.b_h, self.W_p, \
        self.b_p, self.W_i, self.U_i, self.b_i, self.W_f, self.U_f, \
        self.b_f, self.W_o, self.U_o, self.b_o, self.W_u, self.U_u, self.b_u = state
