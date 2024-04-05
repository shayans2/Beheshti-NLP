from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import hazm


class KeywordExtractorModel:
    def __init__(
        self,
        use_idf=True,
        ngram_range=(1, 2),
        max_df=0.8,
        min_df=1,
        stop_words=hazm.stopwords_list(),
    ):
        self.model = TfidfVectorizer(
            use_idf=use_idf,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            stop_words=stop_words,
        )

    def get_feature_names(self, data):
        self.model.fit_transform(data)
        feature_names = self.model.get_feature_names_out()
        return feature_names

    def save_model(self, file_name, compress=False):
        model = joblib.dump(self.model, file_name, compress=compress)
        return model
