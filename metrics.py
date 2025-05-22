# pip install nltk sentence-transformers rouge-score
# BLEU
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# BERT
from sentence_transformers import SentenceTransformer, util
# rouge
from rouge_score import rouge_scorer

from rouge_score import rouge_scorer
def calculate_bleu(reference: str, hypothesis: str) -> float:
    smoothie = SmoothingFunction().method1
    
    # 띄어쓰기 기준 tokenization (간단한 경우)
    reference_tokens = reference.strip().split()
    hypothesis_tokens = hypothesis.strip().split()
    
    return sentence_bleu(
        [reference_tokens],  # list of references
        hypothesis_tokens,
        smoothing_function=smoothie,
        # weights=(0.25, 0.25, 0.25, 0.25)  # BLEU-4
        weights=(0.33, 0.33, 0.33)          # BLEU-3
    )


def calculate_similarity(result_text, expected_output, question_num):
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # 임베딩 생성
    result_embedding = st_model.encode(result_text)
    output_embedding = st_model.encode(expected_output)
    
    # result_embedding = st_model.encode(result_text, normalize_embeddings=True)
    # output_embedding = st_model.encode(expected_output, normalize_embeddings=True)
    # 유사도 계산
    similarity = util.pytorch_cos_sim(result_embedding, output_embedding)[0][0]
    
    print(f"생성된 답변: {result_text}")
    print(f"예상 답변: {expected_output}")
    print(f"유사도: {similarity:.4f}")
    
    return similarity

def calculate_rouge(reference: str, hypothesis: str) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)

    result = {}
    for key in scores:
        result[key] = {
            'precision': round(scores[key].precision, 4),
            'recall': round(scores[key].recall, 4),
            'f1': round(scores[key].fmeasure, 4)
        }
    return result
