import esda
import libpysal

def convert_matrix_to_W(matrix):
    nb_dict ={}
    w_dict = {}
    for i in range(0, matrix.shape[0]):
        nb_dict.update({i:matrix.getrow(i).indices})
        w_dict.update({i:matrix.getrow(i).data})

    W = libpysal.weights.W(neighbors=nb_dict, weights=w_dict)
    return W

def calculate_autocorrelation(y, w):
    w = convert_matrix_to_W(w) if type(w) != libpysal.weights.weights.W else w

    moran_i = esda.Moran(y, w, transformation='O',permutations=999)

    lisa = esda.Moran_Local(y, w, transformation='O',permutations=999)
    
    return {
        'moran_I': moran_i.I,
        'moran_z_sim': moran_i.z_sim,
        'moran_p_sim': moran_i.p_sim,
        'moran_p_z_sim': moran_i.p_z_sim,
        'lisa_p_sim': (lisa.p_sim < 0.05).mean(),
        'lisa_p_z_sim': (lisa.p_z_sim < 0.05).mean()
    }