__doc__ = """Implementation of Tree LSTMs described in http://arxiv.org/abs/1503.00075"""

import tree_rnn

import theano
from theano import tensor as T


class ChildSumTreeLSTM(tree_rnn.TreeRNN):
    def create_recursive_unit(self):
        # тут все по статье
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
        # добавляем все в параметры обучения
        self.params.extend([
            self.W_i, self.U_i, self.b_i,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
            self.W_u, self.U_u, self.b_u])
        # cобственно функция для вычислений
        def unit(parent_x, child_h, child_c, child_exists):
            """ Функция для пересчета.
            
            :param parent_x: векторизация родителя
            :param child_h: h детей, shape=[self.degree, self.hidden_dim]
            :param child_c: c детей, shape=[self.degree, self.hidden_dim]
            :param child_exists: 1 если ребенок есть, 0 иначе, shape=[self.degree]
            :return: h и с для родителя
            """
            # тут все по статье
            h_tilde = T.sum(child_h, axis=0)
            i = T.nnet.sigmoid(T.dot(self.W_i, parent_x) + T.dot(self.U_i, h_tilde) + self.b_i)
            o = T.nnet.sigmoid(T.dot(self.W_o, parent_x) + T.dot(self.U_o, h_tilde) + self.b_o)
            u = T.tanh(T.dot(self.W_u, parent_x) + T.dot(self.U_u, h_tilde) + self.b_u)
            # TODO: лул
            f = (T.nnet.sigmoid(
                # умножаем и переводим в строку
                T.dot(self.W_f, parent_x).dimshuffle('x', 0) +
                T.dot(child_h, self.U_f.T) +
                self.b_f.dimshuffle('x', 0)) *
                # оставляем для существующих детей, остальное зануляем
                 child_exists.dimshuffle(0, 'x'))
            # тут все по статье (при сумме зануленные значения не учитываются, поэтому от числа детей не зависим)
            c = i * u + T.sum(f * child_c, axis=0)
            h = o * T.tanh(c)
            return h, c
        return unit

    def create_leaf_unit(self):
        dummy = 0 * theano.shared(self.init_vector([self.degree, self.hidden_dim]))
        def unit(leaf_x):
            # эта функция определена выше (unit), сюда мы просто передаем нули для h и c
            return self.recursive_unit(
                leaf_x,
                dummy,
                dummy,
                dummy.sum(axis=1))
        return unit

    def compute_tree(self, emb_x, tree):
        """ Создаем вычислительный граф.
        
        :param emb_x: получем по номеру векторизацию вершины 
        :param tree: список детей по номеру
        :return: список внутренних векторов
        """
        # создаем функцию для вычислений внутренней ноды
        self.recursive_unit = self.create_recursive_unit()
        # создаем функцию для листа
        self.leaf_unit = self.create_leaf_unit()
        # количество внутренних нод
        num_nodes = tree.shape[0]  # num internal nodes
        # количество листьев
        num_leaves = self.num_words - num_nodes
        # считаем скрытые слои для листьев
        # по задумке, листья идут первыми с списке
        (leaf_h, leaf_c), _ = theano.map(
            fn=self.leaf_unit,
            sequences=[emb_x[:num_leaves]])
        # если дерево нерегулярно, то мы стакаем вектора
        if self.irregular_tree:
            init_node_h = T.concatenate([leaf_h, leaf_h], axis=0)
            init_node_c = T.concatenate([leaf_c, leaf_c], axis=0)
        else:
            # leaf_h/c shape=[num_leaves, hidden_dim]
            init_node_h = leaf_h
            init_node_c = leaf_c

        # use recurrence to compute internal node hidden states
        def _recurrence(cur_emb, node_info, t, node_h, node_c, last_h):
            """ Пересчет для внутренних вершин. Функция для tensor.scan
            
            :param cur_emb: векторизация
            :param node_info: список детей
            :param t: порядковый номер внутренней вершины с нуля
            :param node_h: 
            :param node_c: 
            :param last_h: последняя посчитанная h, нужна для итога scan
            :return: 
            """
            # вектор 1 - сын есть, 0 - сына нет (для нерегулярных)
            child_exists = node_info > -1
            # идея тут такая: если дерево нерегулярно, то удалять по 1ой вершине не получиться
            # поэтому мы удваиваем значения h и c вверху и используем их здесь
            offset = num_leaves * int(self.irregular_tree) - child_exists * t
            # зануляем несуществующие
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x')
            child_c = node_c[node_info + offset] * child_exists.dimshuffle(0, 'x')
            # считаем
            parent_h, parent_c = self.recursive_unit(cur_emb, child_h, child_c, child_exists)
            # добавляем посчитанные значения (по 1му)
            node_h = T.concatenate([node_h, parent_h.reshape([1, self.hidden_dim])])
            node_c = T.concatenate([node_c, parent_c.reshape([1, self.hidden_dim])])
            # от одной вершины можно избавиться
            return node_h[1:], node_c[1:], parent_h

        # считаем вектора внутренних вершин
        dummy = theano.shared(self.init_vector([self.hidden_dim]))
        (_, _, parent_h), _ = theano.scan(
            # функция - выше
            fn=_recurrence,
            # стартовые последние три аргумента
            outputs_info=[init_node_h, init_node_c, dummy],
            # списки для передаваемых аргументов - первые три в функции
            sequences=[emb_x[num_leaves:], tree, T.arange(num_nodes)],
            # число шагов = число внутренних вершин
            n_steps=num_nodes)

        # возвращаем внутренние слои листьев и родителей в таком же порядке
        return T.concatenate([leaf_h, parent_h], axis=0)
