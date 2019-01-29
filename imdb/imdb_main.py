from imdb.Dataers import Dataers
from imdb.Filers import Filers
from imdb.Processers import Processers

# Dataers.fetch_sentences()
# Dataers.train_word2vec_model()

# Filers.make_tagged_sentences()

# for enum_type in Filers.EnumTypes:
#     Filers.make_data_for_network(enum_type)

# Processers.train_cnn_original(Filers.EnumTypes.ALL_INDEXES)
# Processers.train_cnn_modified(Filers.EnumTypes.ALL_INDEXES, Filers.EnumTypes.ADJ_INDEXES)
Processers.train_vdcnn_pos(Filers.EnumTypes.ALL_INDEXES)
# Processers.train_lstm_kaggle(Filers.EnumTypes.ALL_WORDS)
# Processers.train_cnn_lstm(Filers.EnumTypes.ALL_INDEXES)
# Processers.train_cnn_lstm_paper(Filers.EnumTypes.ALL_INDEXES)
