###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import torch
import habana_frameworks.torch as htorch
from vllm.hpu.cache_ops import insert_or_update_cache

def with_mark_steps(fn):
    def wrapped(*args, **kwargs):
        htorch.core.mark_step()
        result = fn(*args, **kwargs)
        del args
        del kwargs
        htorch.core.mark_step()
        return result
    return wrapped


def profile_reicpes(recipe_names):
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    import tqdm
    recipe_names_short = [name.replace('.graph_dumps/HabanaFusedOpLazy_', '') for name in recipe_names]
    recipes = [Path(Path.cwd().joinpath(name + '-PostGraph-symbol.pbtxt')).open('r').read() for name in recipe_names]

    def generic_similarity_backend(recipes, similarity_func, backend_name=''):
        num_recipes = len(recipes)
        sim_tri = np.zeros((num_recipes, num_recipes))
        total = (num_recipes * (num_recipes + 1)) // 2 - num_recipes
        backend_txt = f' with {backend_name}' if backend_name != '' else ''
        with tqdm.tqdm(total=total, desc=f" computing similarity matrix{backend_txt}") as pbar:
            for i in range(num_recipes):
                for j in range(i):
                    sim_tri[i,j] = similarity_func(recipes[i], recipes[j])
                    pbar.update(1)
        sim = sim_tri.T + sim_tri
        sim_idx = np.arange(sim_tri.shape[0])
        sim[sim_idx,sim_idx] = 1
        return sim 

    def cosine_similarity_rad_backend(recipes):
        from strsimpy.cosine import Cosine
        s = Cosine(2)
        return generic_similarity_backend(recipes, s.similarity, "Cosine (rad)"), "cosine similarity, 1 = max similarity"

    def cosine_similarity_deg_backend(recipes):
        from strsimpy.cosine import Cosine
        s = Cosine(2)
        rad = generic_similarity_backend(recipes, s.similarity, "cosine similarity")
        deg = np.degrees(np.arccos(rad))
        return deg, "cosine similarity (deviation in deg, 0 = max similarity)"

    def overlap_coefficient_backend(recipes):
        from strsimpy.overlap_coefficient import OverlapCoefficient
        s = OverlapCoefficient(2)
        return generic_similarity_backend(recipes, s.similarity, OverlapCoefficient.__name__),  OverlapCoefficient.__name__

    def normalized_levenshtein_backend(recipes):
        from strsimpy.normalized_levenshtein import NormalizedLevenshtein
        s = NormalizedLevenshtein()
        return generic_similarity_backend(recipes, s.similarity, NormalizedLevenshtein.__name__), NormalizedLevenshtein.__name__

    def jaro_winkler_backend(recipes):
        from strsimpy.jaro_winkler import JaroWinkler
        s = JaroWinkler()
        return generic_similarity_backend(recipes, s.similarity, JaroWinkler.__name__), JaroWinkler.__name__
    
    def tfidf_weird_backend(recipes):
        def tfidf_single_elem(x,y):
            from sklearn.feature_extraction.text import TfidfVectorizer
            vect = TfidfVectorizer() 
            tfidf = vect.fit_transform([x,y])                                                                                                                                                                                                                       
            sim_sparse = tfidf * tfidf.T 
            sim = sim_sparse.toarray()
            return sim[0,1]
        return generic_similarity_backend(recipes, tfidf_single_elem, 'TfidfVectorizer (weird)'), 'TfidfVectorizer (weird)'
    
    def tfidf_backend(recipes):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vect = TfidfVectorizer() 
        tfidf = vect.fit_transform(recipes)                                                                                                                                                                                                                       
        sim_sparse = tfidf * tfidf.T 
        sim = sim_sparse.toarray()
        return sim, 'TfidfVectorizer'
    
    sim, backend_name = tfidf_backend(recipes)
    plt.rcParams["figure.figsize"] = [16,16]
    plt.rcParams["figure.dpi"] = 300
    cm = ConfusionMatrixDisplay(sim, display_labels=recipe_names_short)
    cm.plot(xticks_rotation='vertical', text_kw={"fontsize":5})
    cm.ax_.set_xlabel("Target recipe number")
    cm.ax_.set_ylabel("Source recipe number")
    plt.title(f'Recipe similarity ({backend_name})')
    return plt
#    plt.savefig('similarity.png')


class Matmul(torch.nn.Module):
    def __init__(self):
        super(Matmul, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


class Softmax(torch.nn.Module):
      def __init__(self):
        super().__init__()

      def forward(self, x, dim = None, inv_head = None):
        return torch.softmax(x, dim)


class VLLMKVCache(torch.nn.Module):
    def __init__(self):
        super(VLLMKVCache, self).__init__()

    def forward(self, input, cache, block_indices, block_offset):
        insert_or_update_cache(input, cache, block_indices, block_offset)
        return cache

    def fetch_from_cache(self, cache, blocks):
        return cache.index_select(0, blocks)
