from digikala.Dataers import Dataers
from digikala.Filers import Filers
from digikala.Processers import Processers

# Dataers.fetch_sentences()
# Dataers.train_word2vec_model()

# Filers.make_tagged_sentences()

# for enum_type in Filers.EnumTypes:
#     Filers.make_data_for_network(enum_type)
# Filers.cluster_data(Filers.EnumTypes.ALL_WORDS)
# Filers.load_clustered_data(Filers.EnumTypes.ALL_INDEXES)
# Filers.cluster_data()
# Processers.train_cnn_original(Filers.EnumTypes.ALL_INDEXES)
# Processers.train_cnn_modified(Filers.EnumTypes.ALL_INDEXES, Filers.EnumTypes.ADJ_INDEXES)
# Processers.train_vdcnn_pos(Filers.EnumTypes.ALL_INDEXES)
Processers.train_cnn_lstm(Filers.EnumTypes.ALL_INDEXES)
# Processers.train_cnn_lstm_paper(Filers.EnumTypes.ALL_INDEXES)

# for enum_type in Filers.EnumTypes:
#     if enum_type in (Filers.EnumTypes.NOTNOUN_INDEXES,
#                      Filers.EnumTypes.ADJ_INDEXES,
#                      Filers.EnumTypes.ADV_INDEXES,
#                      Filers.EnumTypes.VERB_INDEXES,
#                      Filers.EnumTypes.ADJ_ADV_VERB_INDEXES):
#         Processers.train_cnn_lstm(enum_type)
#
#
# for enum_type in Filers.EnumTypes:
#     if enum_type in (Filers.EnumTypes.ALL_INDEXES,
#                      Filers.EnumTypes.NOTNOUN_INDEXES,
#                      Filers.EnumTypes.ADJ_INDEXES,
#                      Filers.EnumTypes.ADV_INDEXES,
#                      Filers.EnumTypes.VERB_INDEXES,
#                      Filers.EnumTypes.ADJ_ADV_VERB_INDEXES):
#         Processers.train_vdcnn_pos(enum_type)
