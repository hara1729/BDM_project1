from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.pipeline import Pipeline

import numpy as np
import matplotlib.pyplot as plt

import scipy

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

from cleaner import clean

import warnings
import sys

from typing import List, Union, Optional
from typeguard import check_type, typechecked

warnings.filterwarnings("ignore")

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token, pos = 'n') for token in tokens if token.lower() not in stop_words]
    return ' '.join(lemmatized)

def stem_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    stemmed = [stemmer.stem(token) for token in tokens if token.lower() not in stop_words]
    return ' '.join(stemmed)

def category_to_binary(category):
    if category == 'sports':
        return 1
    elif category == "climate":
        return 0
    else:
        raise ValueError(f"Invalid Category {category}")

def categorical_labels(category):
    category_map = {0:"basketball", 1:"baseball", 2:"tennis",
                    3:"football", 4:"soccer", 5:"forest fire", 6:"flood",
                    7:"earthquake", 8:"drought", 9:"heatwave"}

    inverse_map = {v: k for (k, v) in category_map.items()}

    try:
        return inverse_map[category]
    except KeyError:
        sys.exit(f"Invalid category {category} found")


@typechecked
class Problem:
    def __init__(self, path_to_csv, path_to_glove_txt, random_state = 42):
        self.random_state = random_state
        self._get_dataframes(path_to_csv)
        self.path_to_glove_txt = path_to_glove_txt

    def _get_dataframes(self, path):
        df = pd.read_csv(path)
        df["clean_text"] = df["full_text"].apply(lambda x: clean(x))
        df['lemmatized_text'] = df['clean_text'].apply(lemmatize_text)
        df['stemmed_text'] = df['clean_text'].apply(stem_text)
        df['binary_category'] = df['root_label'].apply(category_to_binary)
        df['leaf_category'] = df['leaf_label'].apply(categorical_labels)
        
        self.train_df_clean, self.test_df_clean = train_test_split(df[["clean_text", "binary_category"]], 
                                                                   test_size = 0.2, 
                                                                   random_state = self.random_state)
        self.train_df_lemmatized, self.test_df_lemmatized = train_test_split(df[["lemmatized_text", "binary_category"]], 
                                                                             test_size = 0.2,
                                                                             random_state = self.random_state)
        
        self.train_df_stemmed, self.test_df_stemmed = train_test_split(df[["stemmed_text", "binary_category"]], 
                                                                       test_size = 0.2,
                                                                       random_state = self.random_state)

        self.df = df
    
    def _do_tfidf(self, fit_on: Optional[str] = "clean_text", min_df: Optional[int] = 3):
        check_type(fit_on, Optional[str])
        check_type(min_df, Optional[int])

        assert fit_on in ["clean_text", "lemmatized_text", "stemmed_text"], f"TF-IDF must be learnt on either of [`clean_text`, `lemmatized_text`]"
        assert min_df > 0, f"min_df should be strictly positive. Found {min_df}"
        
        if fit_on == "clean_text":
            df_train, df_test = self.train_df_clean, self.test_df_clean
        elif fit_on == "lemmatized_text":
            df_train, df_test = self.train_df_lemmatized, self.test_df_lemmatized
        else:
            df_train, df_test = self.train_df_stemmed, self.test_df_stemmed

        vectorizer = CountVectorizer(analyzer = "word", token_pattern = r'\b[^\d\W]+\b', stop_words = "english", min_df = min_df)
        train_X_count = vectorizer.fit_transform(df_train[fit_on])
        test_X_count = vectorizer.transform(df_test[fit_on])

        tfidf_transformer = TfidfTransformer()
        train_X_tfidf = tfidf_transformer.fit_transform(train_X_count)
        test_X_tfidf = tfidf_transformer.transform(test_X_count)

        return (train_X_tfidf, test_X_tfidf), (train_X_tfidf.shape, test_X_tfidf.shape)

    def _do_LSI(self, n_components: int, tfidf_features: Optional[np.ndarray] = None, fit_on: Optional[str] = "lemmatized_text"):
        if tfidf_features is None:
            (train_X_tfidf, _), _ = self._do_tfidf(fit_on = fit_on, min_df = 3)
        else:
            train_X_tfidf = tfidf_features
            print(train_X_tfidf.shape)
        
        lsi = TruncatedSVD(n_components = n_components, random_state = self.random_state)
        U_sigma = lsi.fit_transform(train_X_tfidf)
        V_transpose = lsi.components_
        reconstruction = np.dot(U_sigma, V_transpose)

        explained_variance_ratio = lsi.explained_variance_ratio_
        total_explained_variance = np.sum(explained_variance_ratio)

        lsi_error = np.linalg.norm(reconstruction - train_X_tfidf, 'fro')

        return lsi, U_sigma, total_explained_variance, lsi_error
    
    def _do_NMF(self, n_components: int, tfidf_features: Optional[np.ndarray] = None, fit_on: Optional[str] = "lemmatized_text"):
        if tfidf_features is None:
            (train_X_tfidf, _), _ = self._do_tfidf(fit_on = fit_on, min_df = 3)
        else:
            train_X_tfidf = tfidf_features
        
        nmf = NMF(n_components = n_components, random_state = self.random_state)
        W = nmf.fit_transform(train_X_tfidf)
        H = nmf.components_
        reconstruction = np.dot(W, H)
        nmf_error = np.linalg.norm(train_X_tfidf - reconstruction, 'fro')

        return nmf, reconstruction, None, nmf_error

    def _fit_SVM(self, kernel: Optional[str] = 'linear', 
                       C: Optional[float] = 1.0, 
                       probability: Optional[bool] = True,
                       tfidf_features: Optional[np.ndarray] = None,
                       fit_on: Optional[str] = "lemmatized_text",
                       lsi_num_components: Optional[int] = 128,
                       ):
        
        if tfidf_features is None:
            (train_X_tfidf, _), _ = self._do_tfidf(fit_on = fit_on, min_df = 3)
        else:
            train_X_tfidf = tfidf_features
            print(train_X_tfidf.shape)
        
        if fit_on == "clean_text":
            df_train, df_test = self.train_df_clean, self.test_df_clean
        elif fit_on == "lemmatized_text":
            df_train, df_test = self.train_df_lemmatized, self.test_df_lemmatized
        else:
            df_train, df_test = self.train_df_stemmed, self.test_df_stemmed
        
        lsi_object, X_train_lsi, _, _ = self._do_LSI(n_components = lsi_num_components, 
                                                     tfidf_features = train_X_tfidf, 
                                                     fit_on = fit_on)

        svm = SVC(kernel = kernel, C = C, probability = probability, random_state = self.random_state)
        svm.fit(X_train_lsi, df_train["binary_category"])

        return lsi_object, svm
    
    def _predict_SVM(self, svm, lsi_object, 
                           X_test: Union[str, np.ndarray, scipy.sparse._csr.csr_matrix], 
                           y_test: Union[str, np.ndarray, pd.core.series.Series], 
                           roc: bool = False):
        
        if isinstance(X_test, str):
            assert X_test in ["clean_text", "lemmatized_text", "stemmed_text"]
            (_, X_test), _ = self._do_tfidf(fit_on = X_test, min_df = 3)
        
        if isinstance(y_test, str):
            assert y_test in ["clean_text", "lemmatized_text"]
            if y_test == "clean_text":
                y_test = self.test_df_clean["binary_category"]
            else:
                y_test = self.test_df_lemmatized["binary_category"]
        
        X_test = lsi_object.transform(X_test)
        y_pred = svm.predict(X_test)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        if roc:
            assert svm.probability, "set probability attribute to SVM to `True` to plot ROC curve"
            y_scores = svm.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = None, None, None

        return report, conf_matrix, [tpr, fpr, roc_auc]
    
    def _fit_logistic_classifier(self, penalty: Optional[str] = 'l2',
                                       C: Optional[float] = 1.0,
                                       tfidf_features: Optional[np.ndarray] = None,
                                       fit_on: Optional[str] = "lemmatized_text",
                                       lsi_num_components: Optional[int] = 128):
        
        if tfidf_features is None:
            (train_X_tfidf, _), _ = self._do_tfidf(fit_on = fit_on, min_df = 3)
        else:
            train_X_tfidf = tfidf_features
        
        if fit_on == "clean_text":
            df_train, df_test = self.train_df_clean, self.test_df_clean
        elif fit_on == "lemmatized_text":
            df_train, df_test = self.train_df_lemmatized, self.test_df_lemmatized
        else:
            df_train, df_test = self.train_df_stemmed, self.test_df_stemmed
        
        lsi_object, X_train_lsi, _, _ = self._do_LSI(n_components = lsi_num_components, 
                                                     tfidf_features = train_X_tfidf, 
                                                     fit_on = fit_on)
        
        regressor = LogisticRegression(C = C, solver = 'saga', max_iter = 1000, random_state = self.random_state)
        regressor.fit(X_train_lsi, df_train["binary_category"])

        return lsi_object, regressor

    def _predict_logistic_classifier(self, regressor, lsi_object,
                                           X_test: Union[str, np.ndarray, scipy.sparse._csr.csr_matrix], 
                                           y_test: Union[str, np.ndarray, pd.core.series.Series], 
                                           roc: bool = False):

        if isinstance(X_test, str):
            assert X_test in ["clean_text", "lemmatized_text", "stemmed_text"]
            (_, X_test), _ = self._do_tfidf(fit_on = X_test, min_df = 3)
        
        if isinstance(y_test, str):
            assert y_test in ["clean_text", "lemmatized_text"]
            if y_test == "clean_text":
                y_test = self.test_df_clean["binary_category"]
            else:
                y_test = self.test_df_lemmatized["binary_category"]
        
        X_test = lsi_object.transform(X_test)
        y_pred = regressor.predict(X_test)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        if roc:
            y_scores = regressor.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = None, None, None

        return report, conf_matrix, [tpr, fpr, roc_auc]
    
    
    def _fit_naiveBayes(self, tfidf_features: Optional[np.ndarray] = None,
                              fit_on: Optional[str] = "lemmatized_text",
                              lsi_num_components: Optional[int] = 128):
        if tfidf_features is None:
            (train_X_tfidf, _), _ = self._do_tfidf(fit_on = fit_on, min_df = 3)
        else:
            train_X_tfidf = tfidf_features
        
        if fit_on == "clean_text":
            df_train, df_test = self.train_df_clean, self.test_df_clean
        elif fit_on == "lemmatized_text":
            df_train, df_test = self.train_df_lemmatized, self.test_df_lemmatized
        else:
            df_train, df_test = self.train_df_stemmed, self.test_df_stemmed
        
        lsi_object, X_train_lsi, _, _ = self._do_LSI(n_components = lsi_num_components, 
                                                     tfidf_features = train_X_tfidf, 
                                                     fit_on = fit_on)
        
        clf = GaussianNB()
        clf.fit(X_train_lsi, df_train["binary_category"])

        return lsi_object, clf

    
    def _predict_naiveBayes(self, clf, lsi_object,
                                  X_test: Union[str, np.ndarray, scipy.sparse._csr.csr_matrix], 
                                  y_test: Union[str, np.ndarray, pd.core.series.Series], 
                                  roc: bool = False):
        
        if isinstance(X_test, str):
            assert X_test in ["clean_text", "lemmatized_text", "stemmed_text"]
            (_, X_test), _ = self._do_tfidf(fit_on = X_test, min_df = 3)
        
        if isinstance(y_test, str):
            assert y_test in ["clean_text", "lemmatized_text"]
            if y_test == "clean_text":
                y_test = self.test_df_clean["binary_category"]
            else:
                y_test = self.test_df_lemmatized["binary_category"]
        
        X_test = lsi_object.transform(X_test)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        if roc:
            y_scores = clf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = None, None, None

        return report, conf_matrix, [tpr, fpr, roc_auc]
    
    def _load_glove_embeddings(self):
        with open(self.path_to_glove_txt, encoding = "utf8") as f:
            word_embeddings = dict()
            for line in f:
                values = line.split(' ')
                word_embeddings[values[0]] = np.asarray(values[1:], dtype = np.float32)
        return word_embeddings
    
    def _get_text_embeddings(self, text, word_embeddings):
        '''
        Feature Engineering to get text embeddings from GLoVE embeddings
        '''
        if isinstance(text, str):
            words = text.split()
        else:
            words = text
        embeddings = []
        for word in words:
            if word in word_embeddings.keys():
                embeddings.append(word_embeddings[word])
        
        embeddings = np.asarray(embeddings, dtype = np.float32)
        if embeddings.shape[0] == 0:
            embeddings = np.random.random([len(words), 300])
        embeddings = embeddings.reshape([embeddings.shape[0], 10, 30])
        embeddings = np.mean(embeddings, axis = -1)
        embeddings = np.max(embeddings, axis = -1)

        # keep the top K = 300 embeddings while maintaining their relative order

        abs_embeddings = np.abs(embeddings)
        sorted_indices = np.argsort(abs_embeddings)
        top_indices = sorted_indices[-300:]
        top_indices = np.sort(top_indices)
        embeddings = embeddings[top_indices]
        embeddings = embeddings[..., np.newaxis]
        print(f"feature shape: {embeddings.shape}")
        return embeddings.T
    
    def _get_max_len(self, df):
        max_len = 0
        for arr  in df:
            if arr.shape[1] > max_len:
                max_len = arr.shape[1]
        
        return max_len
    
    def _pad_to_MAX_LEN(self, arr, max_len):
        padding = [
                    [0, 0],
                    [0, max(0, max_len - arr.shape[1])]
                  ]
        
        arr = np.pad(arr, pad_width = padding, mode = "symmetric")
        norm = np.linalg.norm(arr, axis = 1, keepdims = True)
                
        return arr/norm
    
    def Q3(self):
        fit_on = "clean_text"
        min_dfs = range(1, 11)
        sizes = []
        for min_df in min_dfs:
            _, (train_X_tfidf_shape, _) = self._do_tfidf(fit_on = fit_on , min_df = min_df)
            sizes.append(np.prod(train_X_tfidf_shape))
        
        fig, ax = plt.subplots(1, 1)
        ax.plot(min_dfs, sizes, marker = 'o', label = "tfidf_matrix_size")
        ax.legend()
        ax.set_xlabel(r"min_df $\rightarrow$ ")
        ax.set_ylabel(r"tfidf matrix size $\rightarrow$ ")
        ax,grid()
        plt.show()

    def Q4(self):
        fit_on = "clean_text"
        n_components = [1, 10, 50, 100, 200, 500, 1000, 2000]
        explained_variance_ratios = []
        (train_X_tfidf, _), _ = self._do_tfidf(fit_on = fit_on, min_df = 3)
        for n in n_components:
            _, _, explained_variance, _ = self._do_LSI(n_components = n, tfidf_features = train_X_tfidf, fit_on = fit_on)
            explained_variance_ratios.append(explained_variance)
        fig, ax = plt.subplots(1, 1)
        ax.plot(n_components, explained_variance_ratios, marker = 'o', label = "explained variances\nfor different\nn_components")
        ax.legend()
        ax.set_xlabel(r"num components $\rightarrow$ ")
        ax.set_ylabel(r"explained variance $\rightarrow$ ")
        plt.show()

        N = 50
        _, _, _, lsi_error = self._do_LSI(n_components = N, tfidf_features = train_X_tfidf, fit_on = fit_on)
        _, _, _, nmf_error = self._do_NMF(n_components = N, tfidf_features = train_X_tfidf, fit_on = fit_on)

        print(f"Using num_components = {N}:")
        print(f"Reconstruction error in LSI: {lsi_error}")
        print(f"Reconstruction error in NMF: {nmf_error}")

    def Q5(self):
        fit_on = "lemmatized_text"
        (train_X_tfidf, test_X_tfidf), _ = self._do_tfidf(fit_on = fit_on, min_df = 3)

        if fit_on == "clean_text":
            y = self.train_df_clean['binary_category']
        else:
            y = self.train_df_lemmatized['binary_category']
        
        lsi, svm = self._fit_SVM(kernel = 'linear', C = 0.0001, 
                                 probability = True, tfidf_features = train_X_tfidf, 
                                 fit_on = fit_on, lsi_num_components = 128)
        
        report, conf_matrix, [tpr, fpr, roc_auc] = self._predict_SVM(svm = svm, lsi_object = lsi,
                                                                     X_test = test_X_tfidf, y_test = fit_on, roc = True)

        print('-'*70)
        print(r"Hard margin SVM report ($\gamma = 0.0001 $):")
        print('-'*70)
        print(report)
        print('-'*27, "Confusion Matrix", '-'*25)
        print(conf_matrix)
        print('-'*70)

        fig, ax = plt.subplots(1, 1)
        ax.plot(fpr, tpr, label = 'ROC curve (Soft Margin)')
        ax.plot([0, 1], [0, 1], linestyle = '--')
        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC curve')
        ax.legend()
        plt.show()

        
        lsi, svm = self._fit_SVM(kernel = 'linear', C = 1000, 
                                 probability = True, tfidf_features = train_X_tfidf, 
                                 fit_on = fit_on, lsi_num_components = 128)
        
        report, conf_matrix, [tpr, fpr, roc_auc] = self._predict_SVM(svm = svm, lsi_object = lsi,
                                                                     X_test = test_X_tfidf, y_test = fit_on, roc = True)

        print('-'*70)
        print(r"Hard margin SVM report ($\gamma = 1000 $):")
        print('-'*70)
        print(report)
        print('-'*27, "Confusion Matrix", '-'*25)
        print(conf_matrix)
        print('-'*70)

        fig, ax = plt.subplots(1, 1)
        ax.plot(fpr, tpr, label = r'ROC curve (Hard Margin)($\gamma = 1000 $)')
        ax.plot([0, 1], [0, 1], linestyle = '--')
        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC curve')
        ax.legend()
        plt.show()


        lsi, svm = self._fit_SVM(kernel = 'linear', C = 100000, 
                                 probability = True, tfidf_features = train_X_tfidf, 
                                 fit_on = fit_on, lsi_num_components = 128)
        
        report, conf_matrix, [tpr, fpr, roc_auc] = self._predict_SVM(svm = svm, lsi_object = lsi,
                                                                     X_test = test_X_tfidf, y_test = fit_on, roc = True)

        print('-'*70)
        print(r"Hard margin SVM report ($\gamma = 100000 $):")
        print('-'*70)
        print(report)
        print('-'*27, "Confusion Matrix", '-'*25)
        print(conf_matrix)
        print('-'*70)

        fig, ax = plt.subplots(1, 1)
        ax.plot(fpr, tpr, label = r'ROC curve (Hard Margin)($\gamma = 1000 $)')
        ax.plot([0, 1], [0, 1], linestyle = '--')
        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC curve')
        ax.legend()
        plt.show()

        svc = SVC()
        Cs = [10**k for k in range(-3, 7)]
        C_grid = {'C': Cs}
        gs = GridSearchCV(svc, C_grid, cv = 5, scoring = 'accuracy', n_jobs = -1)
        gs.fit(train_X_tfidf, y)
        C_best = gs.best_params_['C']
        print(r"Best $\gamma$ obtained through grid search: ", C_best)

    def Q6(self):
        fit_on = "lemmatized_text"
        (train_X_tfidf, test_X_tfidf), _ = self._do_tfidf(fit_on = fit_on, min_df = 3)

        if fit_on == "clean_text":
            df_train, df_test = self.train_df_clean, self.test_df_clean
        elif fit_on == "lemmatized_text":
            df_train, df_test = self.train_df_lemmatized, self.test_df_lemmatized
        else:
            df_train, df_test = self.train_df_stemmed, self.test_df_stemmed

        lsi, regressor = self._fit_logistic_classifier(penalty = None, C = 1e9, 
                                                       tfidf_features = train_X_tfidf, 
                                                       fit_on = fit_on, lsi_num_components = 128)
        
        report, conf_matrix, [tpr, fpr, roc_auc] = self._predict_logistic_classifier(regressor = regressor, lsi_object = lsi,
                                                                                     X_test = test_X_tfidf, y_test = fit_on, roc = True)
        
        print('-'*70)
        print("[6.a]Un-regularized Logistic Classifier report:")
        print('-'*70)
        print(report)
        print('-'*27, "Confusion Matrix", '-'*25)
        print(conf_matrix)
        print('-'*70)

        fig, ax = plt.subplots(1, 1)
        ax.plot(fpr, tpr, label = 'ROC curve')
        ax.plot([0, 1], [0, 1], linestyle = '--')
        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC curve (Un-regularized Logistic Classifier)')
        ax.legend()
        plt.show()

        reg = LogisticRegression(penalty = 'l1', solver = 'saga', tol = 0.001, max_iter = 1000, random_state = self.random_state)
        Cs = [10**k for k in range(-5, 6)]
        C_grid = {'C': Cs}
        gs = GridSearchCV(reg, C_grid, cv = 5, scoring = 'accuracy', n_jobs = -1)
        gs.fit(train_X_tfidf, y)
        C_best_l1 = gs.best_params_['C']
        print("[6.b.1] Best L1 regularization strength obtained through grid search: ", C_best_l1)

        reg = LogisticRegression(penalty = 'l2', solver = 'saga', tol = 0.001, max_iter = 1000, random_state = self.random_state)
        Cs = [10**k for k in range(-5, 6)]
        C_grid = {'C': Cs}
        gs = GridSearchCV(reg, C_grid, cv = 5, scoring = 'accuracy', n_jobs = -1)
        gs.fit(train_X_tfidf, y)
        C_best_l2 = gs.best_params_['C']
        print("[6.b.1] Best L2 regularization strength obtained through grid search: ", C_best_l2)

        lsi_dict = dict()
        regressor_dict = dict()

        for penalty, C in zip([None, 'l1', 'l2'], [1e9, C_best_l1, C_best_l2]):
            lsi, regressor = self._fit_logistic_classifier(penalty = penalty, C = C, 
                                                           tfidf_features = train_X_tfidf, 
                                                           fit_on = fit_on, lsi_num_components = 128)
            if penalty is None:
                lsi_dict['un-reg'] = lsi
                regressor_dict['un-reg'] = regressor
            
            elif penalty == 'l1':
                lsi_dict['l1'] = lsi
                regressor_dict['l1'] = regressor

            elif penalty == 'l2':
                lsi_dict['l2'] = lsi
                regressor_dict['l2'] = regressor

        print("[6.b.2]")
        for regularization in ['un-reg', 'l1', 'l2']:
            report, _, _ = self._predict_logistic_classifier(regressor = regressor_dict[regularization], 
                                                             lsi_object = lsi_dict[regularization],
                                                             X_test = test_X_tfidf, y_test = fit_on, roc = False)
            if regularization == 'un-reg':
                message = "Un-regularized Logistic Classifier report"
            elif regularization == 'l1':
                message = "L1-regularized Logistic Classifier report (with best params found through grid search)"
            elif regularization == 'l2':
                message = "L2-regularized Logistic Classifier report (with best params found through grid search)"
            
            print(message)
            print('\n')
            print(report)

    def Q7(self):
        fit_on = "lemmatized_text"
        (train_X_tfidf, test_X_tfidf), _ = self._do_tfidf(fit_on = fit_on, min_df = 3)

        if fit_on == "clean_text":
            df_train, df_test = self.train_df_clean, self.test_df_clean
        elif fit_on == "lemmatized_text":
            df_train, df_test = self.train_df_lemmatized, self.test_df_lemmatized
        else:
            df_train, df_test = self.train_df_stemmed, self.test_df_stemmed

        lsi, clf = self._fit_naiveBayes(tfidf_features = train_X_tfidf, 
                                        fit_on = fit_on, lsi_num_components = 128)
        
        report, conf_matrix, [tpr, fpr, roc_auc] = self._predict_naiveBayes(clf = clf, lsi_object = lsi,
                                                                            X_test = test_X_tfidf, y_test = fit_on, roc = True)

        print('-'*70)
        print("[7] Gaussian Naive Bayes Classifier report:")
        print('-'*70)
        print(report)
        print('-'*27, "Confusion Matrix", '-'*25)
        print(conf_matrix)
        print('-'*70)

        fig, ax = plt.subplots(1, 1)
        ax.plot(fpr, tpr, label = 'ROC curve')
        ax.plot([0, 1], [0, 1], linestyle = '--')
        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC curve (Gaussian Naive Bayes Classifier)')
        ax.legend()
        plt.show()

    def Q8(self):
        fit_on = "lemmatized_text"

        if fit_on == "clean_text":
            df_train, df_test = self.train_df_clean, self.test_df_clean
        elif fit_on == "lemmatized_text":
            df_train, df_test = self.train_df_lemmatized, self.test_df_lemmatized
        else:
            df_train, df_test = self.train_df_stemmed, self.test_df_stemmed

        pipeline = Pipeline([
                                ('vect', CountVectorizer(analyzer = "word", token_pattern = r'\b[^\d\W]+\b', stop_words = "english")),
                                ('tfidf', TfidfTransformer()),
                                ('dim_red', 'passthrough'),    
                                ('clf', 'passthrough')
                            ])

        param_grid = [
                        {
                            'vect__min_df': [3, 5],
                            'dim_red': [TruncatedSVD(), NMF(max_iter = 1000)],
                            'dim_red__n_components': [5, 30, 80],
                            'clf': [SVC(C = 1.0, max_iter = 1000)],
                        },
                        {
                            'vect__min_df': [3, 5],
                            'dim_red': [TruncatedSVD(), NMF(max_iter = 1000)],
                            'dim_red__n_components': [5, 30, 80],
                            'clf': [LogisticRegression(C = 1000, solver = 'saga', max_iter = 1000)],
                            'clf__penalty': ['l1', 'l2'],
                        },
                        {
                            'vect__min_df': [3, 5],
                            'dim_red': [TruncatedSVD(), NMF(max_iter = 1000)],
                            'dim_red__n_components': [5, 30, 80],
                            'clf': [GaussianNB()]
                        }
                    ]

        gs = GridSearchCV(pipeline, param_grid, cv = 5, scoring = 'accuracy', n_jobs = -1)

        gs.fit(df_train[fit_on], df_train["binary_category"])

        print("Best Parameters Found (Using Lemmatized text):")
        for param_name in gs.best_params_.keys():
            print(f"{param_name}: {gs.best_params_[param_name]}")

        results = gs.cv_results_
        top_5_indices = sorted(range(len(results['mean_test_score'])), 
                               key=lambda i: results['mean_test_score'][i], reverse=True)[:5]
        top_5_params = [results['params'][i] for i in top_5_indices]

        for idx, params in enumerate(top_5_params):
            print(f"Top {idx + 1} Parameters(Using Lemmatized text):\n{params}")

        for idx, params in enumerate(top_5_params):
            pipeline.set_params(**params)
            pipeline.fit(df_train[fit_on], df_train["binary_category"]) 
            predictions = pipeline.predict(df_test[fit_on])
            print(f"\nPerformance for Top {idx + 1} Parameters(Using Lemmatized text):")
            print(classification_report(df_test["binary_category"], predictions))

        print("="*100)
        
        # ----------------------------------------------------------------------------------------------------------------------------------

        fit_on = "stemmed_text"

        if fit_on == "clean_text":
            df_train, df_test = self.train_df_clean, self.test_df_clean
        elif fit_on == "lemmatized_text":
            df_train, df_test = self.train_df_lemmatized, self.test_df_lemmatized
        else:
            df_train, df_test = self.train_df_stemmed, self.test_df_stemmed

        pipeline = Pipeline([
                                ('vect', CountVectorizer(analyzer = "word", token_pattern = r'\b[^\d\W]+\b', stop_words = "english")),
                                ('tfidf', TfidfTransformer()),
                                ('dim_red', 'passthrough'),    
                                ('clf', 'passthrough')
                            ])

        param_grid = [
                        {
                            'vect__min_df': [3, 5],
                            'dim_red': [TruncatedSVD(), NMF(max_iter = 1000)],
                            'dim_red__n_components': [5, 30, 80],
                            'clf': [SVC(C = 1.0, max_iter = 1000)],
                        },
                        {
                            'vect__min_df': [3, 5],
                            'dim_red': [TruncatedSVD(), NMF(max_iter = 1000)],
                            'dim_red__n_components': [5, 30, 80],
                            'clf': [LogisticRegression(C = 1000, solver = 'saga', max_iter = 1000)],
                            'clf__penalty': ['l1', 'l2'],
                        },
                        {
                            'vect__min_df': [3, 5],
                            'dim_red': [TruncatedSVD(), NMF(max_iter = 1000)],
                            'dim_red__n_components': [5, 30, 80],
                            'clf': [GaussianNB()]
                        }
                    ]

        gs = GridSearchCV(pipeline, param_grid, cv = 5, scoring = 'accuracy', n_jobs = -1)

        gs.fit(df_train[fit_on], df_train["binary_category"])

        print("Best Parameters Found (Using Stemmed text):")
        for param_name in gs.best_params_.keys():
            print(f"{param_name}: {gs.best_params_[param_name]}")

        results = gs.cv_results_
        top_5_indices = sorted(range(len(results['mean_test_score'])), 
                               key=lambda i: results['mean_test_score'][i], reverse=True)[:5]
        top_5_params = [results['params'][i] for i in top_5_indices]

        for idx, params in enumerate(top_5_params):
            print(f"Top {idx + 1} Parameters(Using Stemmed text):\n{params}")

        for idx, params in enumerate(top_5_params):
            pipeline.set_params(**params)
            pipeline.fit(df_train[fit_on], df_train["binary_category"]) 
            predictions = pipeline.predict(df_test[fit_on])
            print(f"\nPerformance for Top {idx + 1} Parameters (Using Stemmed text:")
            print(classification_report(df_test["binary_category"], predictions))

    def Q9(self):
        fit_on = "clean_text"

        word_embeddings = self._load_glove_embeddings()

        if fit_on == "clean_text":
            df_train, df_test = train_test_split(self.df[["clean_text", "binary_category"]], 
                                                         test_size = 0.2,
                                                         random_state = self.random_state)
        elif fit_on == "lemmatized_text":
            df_train, df_test = train_test_split(self.df[["lemmatized_text", "binary_category"]], 
                                                         test_size = 0.2,
                                                         random_state = self.random_state)
        elif fit_on == "stemmed_text":
            df_train, df_test = train_test_split(self.df[["stemmed_text", "binary_category"]], 
                                                         test_size = 0.2,
                                                         random_state = self.random_state)
        elif fit_on == "keywords":
            df_train, df_test = train_test_split(self.df[["keywords", "binary_category"]], 
                                                         test_size = 0.2,
                                                         random_state = self.random_state)
        else:
            raise ValueError("Invalid `fit_on`")

        df_train["embeddings"] = df_train[fit_on].apply(self._get_text_embeddings, args = [word_embeddings])
        df_test["embeddings"] = df_test[fit_on].apply(self._get_text_embeddings, args = [word_embeddings])

        max_len = self._get_max_len(df_train["embeddings"])
        df_train["embeddings"] = df_train["embeddings"].apply(lambda x: self._pad_to_MAX_LEN(x, max_len))

        max_len = self._get_max_len(df_test["embeddings"])
        df_test["embeddings"] = df_test["embeddings"].apply(lambda x: self._pad_to_MAX_LEN(x, max_len))

        embeddings = []
        for emb in df_train["embeddings"]:
            embeddings.append(emb)

        embeddings = np.squeeze(np.asarray(embeddings))

        lsi, svm = self._fit_SVM(probability = True, 
                                 tfidf_features = embeddings, 
                                 fit_on = "lemmatized_text", 
                                 lsi_num_components = 20)
        
        embeddings = []
        for emb in df_test["embeddings"]:
            embeddings.append(emb)

        embeddings = np.squeeze(np.asarray(embeddings))
        
        report, conf_matrix, [tpr, fpr, roc_auc] = self._predict_SVM(svm = svm, lsi_object = lsi,
                                                                     X_test = embeddings, y_test = "lemmatized_text", roc = True)

        print('-'*70)
        print("SVM on Feature Engineered Data")
        print('-'*70)
        print(report)
        print('-'*27, "Confusion Matrix", '-'*25)
        print(conf_matrix)
        print('-'*70)

        fig, ax = plt.subplots(1, 1)
        ax.plot(fpr, tpr, label = 'ROC curve')
        ax.plot([0, 1], [0, 1], linestyle = '--')
        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC curve (Feature Enigneered Data)')
        ax.legend()
        plt.show()
                                            
if __name__ == "__main__":
    prob = Problem(path_to_csv = "./Dataset1.csv", path_to_glove_txt = "./glove.6B/glove.txt")
    # prob.Q3()
    # prob.Q4()
    # prob.Q5()
    # prob.Q6()
    # prob.Q7()
    # prob.Q8()
    prob.Q9()
    