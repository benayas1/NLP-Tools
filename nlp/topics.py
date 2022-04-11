
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import sentence_transformers as st
from rank_bm25 import BM25Okapi
from sentence_transformers import models, losses
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import math
import time


def _prepare_dataset(data, labels=None):
    """
    Format data from a pandas DataFrame or an iterable into a list of InputExample

    Args:
        data (pd.DataFrame or Iterable): Data containing the sentences
        labels (Iterable, optional): List or array of labels. If None, then it will be considered \
        to be included in the data parameter

    Returns:
        list of InputExample: The formatted list of sentence pairs
    """
    # data is a DataFrame
    if isinstance(data, pd.DataFrame):
        sentences = data[[c for c in data if 'sentence' if c]]
        assert sentences.shape[1] >= 2, "At least 2 sentence columns are required"
        if labels is None:
            labels = data['labels'] if 'labels' in data else None
        assert labels is not None, "No labels found"
        sentences = sentences.iloc[:, :2].values
        return [InputExample(i, sentences[i], labels[i]) for i in range(len(sentences))]

    # If data is not a DataFrame, we assume it is an Iterable
    if labels is not None:
        assert len(data) == len(labels), "Number of labels and data does not match"
        return [InputExample(i, data[i], labels[i]) for i in range(len(data))]
    else:
        # If labels is not provided, consider them to be included in the last position of every item
        return [InputExample(i, data[i][:-1], data[i][-1]) for i in range(len(data))]


def augmented_training_bm25(data: List[InputExample],
                            model_name: str,
                            top_k: int = 10,
                            num_epochs: int = 1,
                            batch_size: int = 32,
                            save_path: str = None):
    """
    Augmented data training method for training sentence transformers.
    The algorithm starts training a classifier on the gold dataset and using it \
    to generate labels for the unlabelled data, generating the silver dataset.
    To create the sentence pairs of the silver dataset, it uses BM25 search algorithm.
    The last step is to train a bi-encoder model using both datasets

    Args:
        data (List[InputExample]): sentence pairs
        model_name (str): string to use to load a pretrained model
        top_k (int, optional): Top K sentences to select for each sentence in the dataset. Defaults to 25.
        num_epochs (int, optional): Number of epochs to train. Defaults to 1.
        batch_size (int, optional): Training batch size. Defaults to 32.
        save_path (str, optional): Path to save the model to. Defaults to None.

    Returns:
        torch.nn.Module: A trained model
    """

    # This parameter to be handled through configuration file
    max_seq_length = 128

    # Step 0: Split the dataset into train/val/test
    print("Splitting dataset into train/val/test...")
    gold_ds, val_ds = train_test_split(data,
                                       train_size=0.8,
                                       random_state=42,
                                       shuffle=True)
    val_ds, test_ds = train_test_split(val_ds,
                                       train_size=0.5,
                                       random_state=42,
                                       shuffle=True)

    # Step 1: Train cross-encoder model with gold dataset
    print("Training a model on gold dataset...")
    t = time.time()

    # From HuggingFace
    cross_encoder = st.cross_encoder.CrossEncoder(model_name, num_labels=1)
    train_dataloader = DataLoader(gold_ds,
                                  shuffle=True,
                                  batch_size=batch_size)
    evaluator = CECorrelationEvaluator.from_input_examples(val_ds, name='val')

    # 10% of train data for warm-up
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

    # Train the cross-encoder model
    cross_encoder.fit(train_dataloader=train_dataloader,
                      evaluator=evaluator,
                      epochs=num_epochs,
                      warmup_steps=warmup_steps)
    print(f"Time to train {time.time() - t}")

    # Step 2: Generate silver dataset
    print("Generating a silver dataset...")

    # BM25 algorithm is a text search
    silver_ds = []
    bm25 = BM25Okapi([text for example in gold_ds for text in example.texts])

    for i in range(len(gold_ds)):
        results = bm25.get_top_n(gold_ds[i], gold_ds, n=top_k)
        for hit in results:
            silver_ds.append((gold_ds[i], hit))

    # Finally, infer labels for the silver dataset
    silver_scores = cross_encoder.predict(silver_ds)
    assert all(0.0 <= score <= 1.0 for score in silver_scores)

    silver_ds = [InputExample(texts=[pair[0], pair[1]], label=score) for pair, score in zip(silver_ds, silver_scores)]

    # Step 3: Train bi-encoder model with both gold + silver dataset
    print("Training a bi-encoder model...")

    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    train_dataloader = DataLoader(gold_ds + silver_ds,
                                  shuffle=True,
                                  batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model=bi_encoder)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_ds, name='val')

    # Configure the training.
    # 10% of train data for warm-up
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

    # Train the bi-encoder model
    bi_encoder.fit(train_objectives=[(train_dataloader, train_loss)],
                   evaluator=evaluator,
                   epochs=num_epochs,
                   evaluation_steps=1000,
                   warmup_steps=warmup_steps,
                   output_path=save_path
                   )

    # Step 4 Evaluate
    print("Evaluating the model")
    # load the stored augmented-sbert model
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_ds, name='test')
    test_evaluator(bi_encoder, output_path=save_path)

    return bi_encoder
