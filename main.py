import os
import pickle
from itertools import chain
from pycparser import c_ast
from cparser.cparser import parse_c_file
from embedder.embedder import make_embedding
from lstm.translator import translate_ast, LabelContex
from lstm.tree_lstm import ChildSumTreeLSTM
from lstm.semantic_lstm import SemanticLSTM


def get_c_files(path):
    l = []
    for path, dirs, files in os.walk(path):
        for file in files:
            if file[0] != ".":
                l.append("{}/{}".format(path, file))
    return l


DUMMY_C_FILES = get_c_files("c_files/dummy")
CODEFLAWS_C_FILES = get_c_files("c_files/codeflaws")


class EdgesCollector(c_ast.NodeVisitor):
    def __init__(self):
        self._edges = []

    def visit(self, node):
        my_name = node.__class__.__name__
        for _, child in node.children():
            child_name = child.__class__.__name__
            self._edges.append((my_name, child_name))
        super().visit(node)

    def get_edges(self):
        return self._edges


def get_edges(ast):
    visitor = EdgesCollector()
    visitor.visit(ast)
    return visitor.get_edges()


def do_embeding():
    edges = list(chain(*(get_edges(parse_c_file(file)) for file in CODEFLAWS_C_FILES[:1])))
    embedder = make_embedding(edges, embedding_size=5)
    return embedder


CTX_FILE = "./pickled/ctx.serialized"
DATA_FILE = "./pickled/data.serialized"
STATE_FILE = "./pickled/state.serialized"
MAX_DEGREE = 50


def make_ctx():
    ctx = LabelContex()
    for file in chain(CODEFLAWS_C_FILES, DUMMY_C_FILES):
        try:
            ast = parse_c_file(file)
            translate_ast(ast, ctx)
        except:
            pass
    return ctx


def do_pickling_ctx(restore=True):
    ctx = None
    if restore and os.path.isfile(CTX_FILE):
        with open(CTX_FILE, "rb") as f:
            ctx = pickle.load(f)
    else:
        ctx = make_ctx()
        with open(CTX_FILE, "wb") as f:
            pickle.dump(ctx, f, pickle.HIGHEST_PROTOCOL)
    return ctx


def make_data():
    data = []
    ctx = do_pickling_ctx()
    k = 0
    for file1, file2 in zip(CODEFLAWS_C_FILES, CODEFLAWS_C_FILES[1:]):
        print(k)
        k += 1
        try:
            tree1, degree1 = translate_ast(parse_c_file(file1), ctx)
            tree2, degree2 = translate_ast(parse_c_file(file2), ctx)
            y = int(file1[18:].split("-")[:2] == file2[18:].split("-")[:2])
            if degree1 <= MAX_DEGREE and degree2 <= MAX_DEGREE:
                data.append([tree1, tree2, y])
        except:
            pass
    return data


def do_pickling_data(restore=True):
    data = None
    if restore and os.path.isfile(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            data = pickle.load(f)
    else:
        data = make_data()
        with open(DATA_FILE, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return data


def make_model():
    ctx = do_pickling_ctx()
    return SemanticLSTM(num_emb=len(ctx.label_dict),
                        emb_dim=5,
                        hidden_dim=5,
                        degree=MAX_DEGREE,
                        irregular_tree=True)


def do_pickling_model():
    state = None
    if os.path.isfile(STATE_FILE):
        with open(STATE_FILE, "rb") as f:
            state = pickle.load(f)
    ctx = do_pickling_ctx()
    model = SemanticLSTM(num_emb=len(ctx.label_dict),
                         emb_dim=5,
                         hidden_dim=5,
                         degree=MAX_DEGREE,
                         irregular_tree=True)
    if state:
        model.set_state(state)
    return model


def save_state(model):
    with open(STATE_FILE, "wb") as f:
        pickle.dump(model.get_state(), f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # example_c = CODEFLAWS_C_FILES[0]
    # ast = parse_c_file(example_c)
    # embedder = do_embeding()
    # tree = make_tree(ast, embedder)
    # ctx = LabelContex()
    # tree = translate_ast(ast, ctx)
    # print(ctx.max_degree)
    # model = ChildSumTreeLSTM(num_emb=len(ctx.label_dict),
    #                          emb_dim=2,
    #                          hidden_dim=2,
    #                          output_dim=1,
    #                          degree=ctx.max_degree,
    #                          irregular_tree=True)
    # print(model.evaluate(tree))

    data = do_pickling_data()

    model = do_pickling_model()

    print("BEGIN TEST/TRAIN")

    N = 1000
    TRAIN, TEST = data[:int(0.8 * N)], data[int(0.8 * N) + 1: N]
    for t1, t2, y in TRAIN:
        print("ONE OF A TRAIN")
        try:
            model.train_step(t1, t2, y + 1)
        except:
            pass

    right = 0
    for t1, t2, y in TEST:
        print("ONE OF A TEST")
        print(model.predict(t1, t2))
        if abs(model.predict(t1, t2) - (y + 1)) <= 0.5:
            right += 1
    print("RATE: ", right / len(TEST))

    save_state(model)
