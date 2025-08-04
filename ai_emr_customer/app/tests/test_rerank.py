
from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
def test_rerank():
    score = reranker.compute_score(['query', 'passage'])
    print(score)

    scores = reranker.compute_score([['what is panda?', 'hi'], ['我要取消挂号', '退号方法有哪些']])
    print(scores)


def test_rerank_with_local_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('app/model/bge-reranker-large')
    model = AutoModelForSequenceClassification.from_pretrained('app/model/bge-reranker-large')
    model.eval()

    pairs = [['what is panda?', 'hi'], ['我要取消挂号', '退号方法有哪些']]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        print(scores)

def main():
    # test_rerank()
    test_rerank_with_local_model()

if __name__ == "__main__":
    main()