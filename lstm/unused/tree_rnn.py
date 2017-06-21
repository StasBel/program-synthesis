__doc__ = """Tree RNNs aka Recursive Neural Networks."""

import numpy as np
import theano
from theano import tensor as T
from theano.compat.python2x import OrderedDict

theano.config.floatX = 'float32'


class Node(object):
    def __init__(self, val=None):
        self.children = []
        self.val = val
        self.idx = None
        self.height = 1
        self.size = 1
        self.num_leaves = 1
        self.parent = None
        self.label = None

    def _update(self):
        self.height = 1 + max([child.height for child in self.children if child] or [0])
        self.size = 1 + sum(child.size for child in self.children if child)
        self.num_leaves = (all(child is None for child in self.children) +
                           sum(child.num_leaves for child in self.children if child))
        if self.parent is not None:
            self.parent._update()

    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        self._update()

    def add_children(self, other_children):
        self.children.extend(other_children)
        for child in other_children:
            child.parent = self
        self._update()


class BinaryNode(Node):
    def __init__(self, val=None):
        super(BinaryNode, self).__init__(val=val)

    def add_left(self, node):
        if not self.children:
            self.children = [None, None]
        self.children[0] = node
        node.parent = self
        self._update()

    def add_right(self, node):
        if not self.children:
            self.children = [None, None]
        self.children[1] = node
        node.parent = self
        self._update()

    def get_left(self):
        if not self.children:
            return None
        return self.children[0]

    def get_right(self):
        if not self.children:
            return None
        return self.children[1]


def gen_nn_inputs(root_node, max_degree=None, only_leaves_have_vals=True, with_labels=False):
    """ Функция для генерации переменных x и tree из простого дерева.
    
    Как генерируем:
    "x, tree = gen_nn_inputs(root_node, max_degree=self.degree, only_leaves_have_vals=False)"
    "Given a root node, returns the appropriate inputs to NN.
    The NN takes in
        x: the values at the leaves (e.g. word indices)
        tree: a (n x degree) matrix that provides the computation order.
            Namely, a row tree[i] = [a, b, c] in tree signifies that a
            and b are children of c, and that the computation
            f(a, b) -> c should happen on step i."
    :param root_node: рутовая нода
    :param max_degree: максимальное количество детей, превращаем дерево в регулярне с -1 индексами для отсутствия детей.
    :param only_leaves_have_vals: значения только в листьях
    :param with_labels: есть ли метки
    :return: двойка из значений для всех вершин и структуры дерева (список детей для каждой внутренней вершины 
    + сама вершина), напоминаю, что каждой вершине сопоставлен уникальный номер в .idx
    """
    # очищаем индексы в дереве
    _clear_indices(root_node)
    # собираем значения и метки слево направо для листьев
    x, leaf_labels = _get_leaf_vals(root_node)
    # собираем структуру дерева, значения и метки снизу вверху, слева направо для внутренних вершин
    tree, internal_x, internal_labels = _get_tree_traversal(root_node, len(x), max_degree)
    # итого: каждая нода получила номер, мы собрали порядок, получили все значения и метки, проверим
    assert all(v is not None for v in x)
    # если значения не только в листьях
    if not only_leaves_have_vals:
        # то проверим что нет None
        assert all(v is not None for v in internal_x)
        # объединим списки значений
        x.extend(internal_x)
    # если передали max_degree
    if max_degree is not None:
        # то проверим что, дерево max_degree-регулярное
        assert all(len(t) == max_degree + 1 for t in tree)
    # если есть метки
    if with_labels:
        # объединим метки
        labels = leaf_labels + internal_labels
        # True - метка есть, False - иначе
        labels_exist = [l is not None for l in labels]
        # метка если она есть, 0 иначе
        labels = [l or 0 for l in labels]
        # четверка из массивов x и дерева, плюс меток и существования меток, типы внизу
        return (np.array(x, dtype='int32'),
                np.array(tree, dtype='int32'),
                np.array(labels, dtype=theano.config.floatX),
                np.array(labels_exist, dtype=theano.config.floatX))
    # двойка из значений для всех вершин и структуры дерева (список детей для каждой внутренней вершины + сама вершина)
    return (np.array(x, dtype='int32'),
            np.array(tree, dtype='int32'))


def _clear_indices(root_node):
    """ Выставляет None на всех .idx из всех вершин дерева.
    
    :param root_node: рутовая вершина для дерева
    :return: ничего
    """
    root_node.idx = None
    [_clear_indices(child) for child in root_node.children if child]


def _get_leaf_vals(root_node):
    """ Собираем значения и метки со всех листьев в порядке обхода слево направо.
    
    Также нумеруем листья с нуля.
    "Get leaf values in deep-to-shallow, left-to-right order."
    :param root_node: корневая вершина
    :return: значения и метки всех листьев слево направо
    """
    all_leaves = []
    layer = [root_node]
    while layer:
        next_layer = []
        for node in layer:
            if all(child is None for child in node.children):
                all_leaves.append(node)
            else:
                next_layer.extend([child for child in node.children[::-1] if child])
        layer = next_layer

    vals = []
    labels = []
    # итерируемся по листьям слево направо
    for idx, leaf in enumerate(reversed(all_leaves)):
        # нумеруем (с нуля)
        leaf.idx = idx
        # собираем значения
        vals.append(leaf.val)
        # и метку
        labels.append(leaf.label)
    return vals, labels


def _get_tree_traversal(root_node, start_idx=0, max_degree=None):
    """ Получаем вычислительный порядок для дерева.
    
    "Get computation order of leaves -> root."
    :param root_node: рутовая вершина
    :param start_idx: стартовый индекс
    :param max_degree: максимальное число детей
    :return: тройку: 1) список из списка индексов детей + индекс родителя в конце (-1 для 
    отсутствующих, дополняем до max_degree если есть), 2) список значений, 3) список
    меток, обход везде с нижнего слоя до верхнего, слева направо
    """
    if not root_node.children:
        return [], [], []
    layers = []
    layer = [root_node]
    # получаем список слоев вершин (лист листов)
    while layer:
        layers.append(layer[:])
        next_layer = []
        [next_layer.extend([child for child in node.children if child])
         for node in layer]
        layer = next_layer
    tree = []
    internal_vals = []
    labels = []
    idx = start_idx
    # c последнего слоя
    for layer in reversed(layers):
        # слева направо
        for node in layer:
            # проверяем, что если индекс уже есть, значит это лист
            if node.idx is not None:
                # must be leaf
                assert all(child is None for child in node.children)
                continue
            # получаем список индексов детей (или -1 если ребенка нет)
            # ремарка: None дети есть везде, по крайней мере у бинарного дерева
            child_idxs = [(child.idx if child else -1) for child in node.children]
            # если передали max_degree, то давайте дополним -1 чтобы было регулярное дерево
            if max_degree is not None:
                child_idxs.extend([-1] * (max_degree - len(child_idxs)))
            # просто проверили что, получили индексы (номер вершины или -1)
            assert not any(idx is None for idx in child_idxs)
            # присвоили новый индекс
            node.idx = idx
            # добавили зависимость + плюс в конце индекс родителя
            tree.append(child_idxs + [node.idx])
            # добавили значение или -1 если его нет
            internal_vals.append(node.val if node.val is not None else -1)
            # добавили метку
            labels.append(node.label)
            # увеличили счетчик
            idx += 1
    return tree, internal_vals, labels


class TreeRNN(object):
    """Data is represented in a tree structure.
    Every leaf and internal node has a data (provided by the input)
    and a memory or hidden state.  The hidden state is computed based
    on its own data and the hidden states of its children.  The
    hidden state of leaves is given by a custom init function.
    The entire tree's embedding is represented by the final
    state computed at the root.
    """

    def __init__(self, num_emb, emb_dim, hidden_dim, output_dim,
                 degree=2, learning_rate=0.01, momentum=0.9,
                 trainable_embeddings=True,
                 labels_on_nonroot_nodes=False,
                 irregular_tree=False):
        """ Инициализация экземпляра.
        
        :param num_emb: число классов (различных вершин) для которых надо сделать векторизацию
        :param emb_dim: размер вектора векторизации
        :param hidden_dim: размер вектора скрытого слоя
        :param output_dim: размер выходного вектора
        :param degree: количество (максимальное?) детей у вершины
        :param learning_rate: скорость обучения 
        :param momentum: какой-то параметр из градиентного спуска
        :param trainable_embeddings: надо ли обучать векторизацию 
        :param labels_on_nonroot_nodes: надо ли предсказывать класс для некорневых вершин
        :param irregular_tree: имеет ли каждая нода неодинаковое количество вершин
        """
        # проверяем минимальную корректность
        assert emb_dim > 1 and hidden_dim > 1
        # просто все сохраняем
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.degree = degree
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.irregular_tree = irregular_tree
        # сюда мы сохраняем все параметры обучения
        self.params = []
        # это просто вектор векторизации
        self.embeddings = theano.shared(self.init_matrix([self.num_emb, self.emb_dim]))
        # если мы будем его обучать, то давайте добавим его в параметры обучения
        if trainable_embeddings:
            self.params.append(self.embeddings)
        # создаем переменную 'x' - индекс для каждой ноды
        self.x = T.ivector(name='x')  # word indices
        # список детей для вершины
        self.tree = T.imatrix(name='tree')  # shape [None, self.degree]
        # если нужны классы на некорневых вершин
        if labels_on_nonroot_nodes:
            # тогда выходной вектор для каждой вершины
            self.y = T.fmatrix(name='y')  # output shape [None, self.output_dim]
            # TODO: ???
            self.y_exists = T.fvector(name='y_exists')  # shape [None]
        else:
            # иначе - просто выходной вектор (для корня)
            self.y = T.fvector(name='y')  # output shape [self.output_dim]
        # num_words - это число всех вершин в дереве
        self.num_words = self.x.shape[0]  # total number of nodes (leaves + internal) in tree
        # получаем список векторизацию для каждой вершины
        emb_x = self.embeddings[self.x]
        # обнуляем несуществующие векторизации на всякий случай (хотя -1 в списке x у нас быть не должно)
        emb_x = emb_x * T.neq(self.x, -1).dimshuffle(0, 'x')  # zero-out non-existent embeddings
        # здесь мы конструируем вычисличельный граф
        # emb_x - получем по номеру векторизацию вершины
        # self.tree - получем по номеру вектор детей
        self.tree_states = self.compute_tree(emb_x, self.tree)

        # финальный вектор (вектор корня)
        self.final_state = self.tree_states[-1]

        # если нужны метки на нерутовых нодах
        if labels_on_nonroot_nodes:
            self.output_fn = self.create_output_fn_multi()
            self.pred_y = self.output_fn(self.tree_states)
            self.loss = self.loss_fn_multi(self.y, self.pred_y, self.y_exists)
        else:
            # конструируем саму выходную функцию (внутри - софтмакс)
            self.output_fn = self.create_output_fn()

            # собственно - считает функцию
            self.pred_y = self.output_fn(self.final_state)

            # и считаем потери
            self.loss = self.loss_fn(self.y, self.pred_y)

        # градиентный спуск
        updates = self.gradient_descent(self.loss)

        # входные данные для тренировки - векторизация, структура дерева и правильные ответы
        train_inputs = [self.x, self.tree, self.y]

        # если для подсчета нужны нерутовые ноды - добавляем зависимость ответов
        if labels_on_nonroot_nodes:
            train_inputs.append(self.y_exists)

        # функция для тренировки
        self._train = theano.function(train_inputs,
                                      [self.loss, self.pred_y],
                                      updates=updates)

        # функция для вычисления
        self._evaluate = theano.function([self.x, self.tree],
                                         self.final_state)

        # функция для предсказания
        self._predict = theano.function([self.x, self.tree],
                                        self.pred_y)

    def _check_input(self, x, tree):
        """ Проверяем, что все в порядке.
        
        :param x: 
        :param tree: 
        :return: 
        """
        assert np.array_equal(tree[:, -1], np.arange(len(x) - len(tree), len(x)))
        if not self.irregular_tree:
            assert np.all((tree[:, 0] + 1 >= np.arange(len(tree))) |
                          (tree[:, 0] == -1))
            assert np.all((tree[:, 1] + 1 >= np.arange(len(tree))) |
                          (tree[:, 1] == -1))

    def train_step_inner(self, x, tree, y):
        # проверяем, что данные корректные
        self._check_input(x, tree)
        # запускаем тренировку, удаляя номер родителя
        return self._train(x, tree[:, :-1], y)

    def train_step(self, root_node, y):
        x, tree = gen_nn_inputs(root_node, max_degree=self.degree, only_leaves_have_vals=False)
        return self.train_step_inner(x, tree, y)

    def evaluate(self, root_node):
        x, tree = gen_nn_inputs(root_node, max_degree=self.degree, only_leaves_have_vals=False)
        self._check_input(x, tree)
        return self._evaluate(x, tree[:, :-1])

    def predict(self, root_node):
        x, tree = gen_nn_inputs(root_node, max_degree=self.degree, only_leaves_have_vals=False)
        self._check_input(x, tree)
        return self._predict(x, tree[:, :-1])

    def init_matrix(self, shape):
        return np.random.normal(scale=0.1, size=shape).astype(theano.config.floatX)

    def init_vector(self, shape):
        return np.zeros(shape, dtype=theano.config.floatX)

    def create_output_fn(self):
        """ Выходная функция для рутовой ноды.
        
        :return: 
        """
        # параметры обучения
        self.W_out = theano.shared(self.init_matrix([self.output_dim, self.hidden_dim]))
        self.b_out = theano.shared(self.init_vector([self.output_dim]))

        # добавляем в обучение
        self.params.extend([self.W_out, self.b_out])

        # сама функция
        def fn(final_state):
            return T.nnet.softmax(
                T.dot(self.W_out, final_state) + self.b_out)

        return fn

    def create_output_fn_multi(self):
        self.W_out = theano.shared(self.init_matrix([self.output_dim, self.hidden_dim]))
        self.b_out = theano.shared(self.init_vector([self.output_dim]))
        self.params.extend([self.W_out, self.b_out])

        def fn(tree_states):
            return T.nnet.softmax(
                T.dot(tree_states, self.W_out.T) +
                self.b_out.dimshuffle('x', 0))

        return fn

    def loss_fn(self, y, pred_y):
        return T.sum(T.sqr(y - pred_y))

    def loss_fn_multi(self, y, pred_y, y_exists):
        return T.sum(T.sum(T.sqr(y - pred_y), axis=1) * y_exists, axis=0)

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
