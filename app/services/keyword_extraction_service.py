import joblib
import hazm
import numpy as np
import nltk

from app.config.settings import POS_TAGGER_MODEL, TF_IDF_MODEL

class KeywordExtractionService:
    def __init__(self):
        self.model = None
        self.feature_names = None

    def load_model(self):
        from app.services.index import get_ner_service  # Lazy import to avoid circular dependency

        self.model = joblib.load(TF_IDF_MODEL)
        self.feature_names = self.model.get_feature_names_out()
        self.ner_service = get_ner_service()

    def extract_keywords(self, text, n=10):
        if self.model is None or self.feature_names is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        tf_idf_vector = self.model.transform([text])
        all_candidates = self.extract_all_candidates(text)
        keywords = self.__extract_topn_from_vector(
            all_candidates, tf_idf_vector.tocoo(), self.feature_names, n
        )

        if not self.ner_service.loaded:
            raise ValueError("NER model not loaded.")
            
        entities = self.ner_service.get_full_entity_names(text)

        for s in ["organization", "person", "location"]:
            for item in entities[s]:
                keywords.append({"keyword": item, "similarity": None})

        for idx, item in enumerate(keywords):
            for idx2, item2 in enumerate(keywords):
                if idx != idx2 and item2["keyword"] in item["keyword"]:
                    keywords.remove(item2)
        return keywords

    def extract_all_candidates(self, text):
        tagger = hazm.POSTagger(model=POS_TAGGER_MODEL)
        grammers = ["""NP:{<NOUN,EZ>?<NOUN.*>}""", """NP:{<NOUN.*><ADJ.*>?}"""]
        token_tag_list = tagger.tag_sents([hazm.WordTokenizer().tokenize(text)])
        all_candidates = set()
        for grammer in grammers:
            all_candidates.update(
                self.__extract_candidates_per_grammar(token_tag_list, grammer)
            )

        return np.array(list(all_candidates))

    def __extract_candidates_per_grammar(self, tagged, grammer):
        keyphrase_candidate = set()
        np_parser = nltk.RegexpParser(grammer)
        trees = np_parser.parse_sents(tagged)
        for tree in trees:
            for subtree in tree.subtrees(
                filter=lambda t: t.label() == "NP"
            ):  # For each nounphrase
                keyphrase_candidate.add(
                    " ".join(word for word, tag in subtree.leaves())
                )
        keyphrase_candidate = {kp for kp in keyphrase_candidate if len(kp.split()) <= 5}
        keyphrase_candidate = list(keyphrase_candidate)
        return keyphrase_candidate

    def __extract_topn_from_vector(self, candidates, coo_matrix, feature_names, topn):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
        score_vals = []
        feature_vals = []

        for idx, score in sorted_items:
            if feature_names[idx] in candidates:
                score_vals.append(round(score, 3))
                feature_vals.append(feature_names[idx])

        results = []
        for idx in range(len(feature_vals)):
            results.append(
                {"keyword": feature_vals[idx], "similarity": score_vals[idx]}
            )

        for item in results:
            for item2 in results:
                if (
                    item["keyword"] != item2["keyword"]
                    and item2["keyword"] in item["keyword"]
                ):
                    results.remove(item2)

        return results[:topn]
