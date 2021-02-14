import strlearn as sl

random_states = [1111, 2222, 3333, 4444, 5555]
drifts = ['sudden', 'incremental']
noises = [0.0, 0.01, 0.05]
imbalance_ratios = [
    [0.95, 0.05],
    [0.90, 0.10],
    [0.85, 0.15],
    [0.80, 0.20],
    [0.70, 0.30],
]
concept_kwargs = {
    "n_chunks": 200,
    "chunk_size": 500,
    "n_classes": 2,
    "n_drifts": 1,
    "n_features": 10,
    "n_informative": 8,
    "n_redundant": 2,
    "n_repeated": 0,
}


stream = sl.streams.StreamGenerator(**concept_kwargs, y_flip=0.01, weights=[0.85, 0.15], random_state=1111)

n_drifts = concept_kwargs["n_drifts"]
stream_size = (concept_kwargs["n_chunks"]*concept_kwargs["chunk_size"]) / 1000
# stream_name = "stream_strlearn_%s_dirft_%03dk_f%02d_%.2fb_%.2fn_rs%d.arff" % ( "s", stream_size, concept_kwargs["n_features"], 0.15, 0.01, 1111)
stream_name = "stream_sl_%dd_%s_%03dk_f%02d_b%02d_n%02d_rs%03d" % (n_drifts, "s", stream_size, concept_kwargs["n_features"], 0.15*100, 0.01*100, 1111)

stream.save_to_arff("streams/param_setup/%s" % stream_name)
