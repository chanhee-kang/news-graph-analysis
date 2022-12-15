from preprocessing import term_matrix

def get_network(df : pd.DataFrame):
    graph = nx.from_numpy_matrix(df.values)
    