"""
HULAT metrics used for processing speech recognition, from the hulat 
repository
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import evaluate
from bert_score import score as bert_score_fn
from rouge import Rouge


# ---------------------------------------------------------------------------
# Dataclass de resultados
# ---------------------------------------------------------------------------

@dataclass
class TranscriptMetrics:
    """Resultados de todas las métricas para un par (transcripción, referencia)."""

    audio_stem: str

    # SARI
    sari_total: float = 0.0
    sari_add: float = 0.0
    sari_keep: float = 0.0
    sari_del: float = 0.0

    # BLEU
    bleu: float = 0.0
    bleu_precision_1: float = 0.0
    bleu_precision_2: float = 0.0
    bleu_precision_3: float = 0.0
    bleu_precision_4: float = 0.0
    bleu_brevity_penalty: float = 0.0
    bleu_length_ratio: float = 0.0
    bleu_translation_length: int = 0
    bleu_reference_length: int = 0

    # ROUGE
    rouge_1_f1: float = 0.0
    rouge_1_precision: float = 0.0
    rouge_1_recall: float = 0.0
    rouge_2_f1: float = 0.0
    rouge_2_precision: float = 0.0
    rouge_2_recall: float = 0.0
    rouge_l_f1: float = 0.0
    rouge_l_precision: float = 0.0
    rouge_l_recall: float = 0.0

    # BERTScore
    bertscore_precision: float = 0.0
    bertscore_recall: float = 0.0
    bertscore_f1: float = 0.0

    # Metadatos opcionales
    model_size: str = ""
    error: Optional[str] = None  # Si algo falló, se guarda el mensaje aquí


# ---------------------------------------------------------------------------
# SARI
# ---------------------------------------------------------------------------

def calculate_sari(
    source: str,
    prediction: str,
    references: list[str],
) -> tuple[float, float, float, float]:
    """
    Calcula SARI (ADD, KEEP, DELETE, TOTAL) siguiendo Xu et al. (2016).

    Args:
        source:      Texto fuente (transcripción sin refinar / audio original en bruto).
                     Para Whisper, se puede pasar el mismo ``prediction`` si no hay
                     fuente disponible; en ese caso SARI mide únicamente ADD/KEEP.
        prediction:  Texto generado por el modelo (transcripción Whisper).
        references:  Lista con al menos una referencia (texto de referencia).

    Returns:
        (sari_add, sari_keep, sari_del, sari_total)
    """

    def _ngrams(tokens: list[str], n: int) -> list[tuple]:
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return precision, recall, f1

    src_tok = source.split()
    pred_tok = prediction.split()
    refs_tok = [r.split() for r in references]
    ngram_levels = [1, 2, 3, 4]

    add_scores, keep_scores, del_scores = [], [], []

    for n in ngram_levels:
        src_ng = set(_ngrams(src_tok, n))
        pred_ng = set(_ngrams(pred_tok, n))
        ref_ngs = [set(_ngrams(r, n)) for r in refs_tok]
        union_ref = set().union(*ref_ngs)

        keep_ng = src_ng & pred_ng
        add_ng = pred_ng - src_ng
        del_ng = src_ng - pred_ng

        # KEEP
        tp_k = len(keep_ng & union_ref)
        fp_k = len(keep_ng - union_ref)
        fn_k = len((src_ng & union_ref) - keep_ng)
        _, _, keep_f1 = _prf(tp_k, fp_k, fn_k)

        # ADD
        tp_a = len(add_ng & union_ref)
        fp_a = len(add_ng - union_ref)
        fn_a = len((union_ref - src_ng) - add_ng)
        _, _, add_f1 = _prf(tp_a, fp_a, fn_a)

        # DELETE
        good_del = src_ng - union_ref
        tp_d = len(del_ng & good_del)
        fp_d = len(del_ng - good_del)
        del_prec = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0.0

        add_scores.append(add_f1)
        keep_scores.append(keep_f1)
        del_scores.append(del_prec)

    avg_add = sum(add_scores) / len(add_scores)
    avg_keep = sum(keep_scores) / len(keep_scores)
    avg_del = sum(del_scores) / len(del_scores)
    sari = (avg_add + avg_keep + avg_del) / 3

    return round(avg_add, 4), round(avg_keep, 4), round(avg_del, 4), round(sari, 4)


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

# Se carga una sola vez para no pagar el coste de importación en cada llamada
_bleu_metric = evaluate.load("bleu")


def calculate_bleu(prediction: str, reference: str) -> dict:
    """
    Calcula BLEU y sus componentes para un único par (predicción, referencia).

    Returns:
        Diccionario con claves: bleu, bleu_precision_{1..4},
        bleu_brevity_penalty, bleu_length_ratio,
        bleu_translation_length, bleu_reference_length.
    """
    results = _bleu_metric.compute(
        predictions=[prediction],
        references=[[reference]],  # references debe ser lista de listas
    )

    if not results:
        return {}

    return {
        "bleu": round(results["bleu"], 4),
        "bleu_precision_1": round(results["precisions"][0], 4),
        "bleu_precision_2": round(results["precisions"][1], 4),
        "bleu_precision_3": round(results["precisions"][2], 4),
        "bleu_precision_4": round(results["precisions"][3], 4),
        "bleu_brevity_penalty": round(results["brevity_penalty"], 4),
        "bleu_length_ratio": round(results["length_ratio"], 4),
        "bleu_translation_length": results["translation_length"],
        "bleu_reference_length": results["reference_length"],
    }


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------

_rouge = Rouge()


def calculate_rouge(prediction: str, reference: str) -> dict:
    """
    Calcula ROUGE-1, ROUGE-2 y ROUGE-L para un único par.

    Returns:
        Diccionario con claves: rouge_{1,2,l}_{f1,precision,recall}.
    """
    scores = _rouge.get_scores(prediction, reference, avg=True)
    return {
        "rouge_1_f1":        round(scores["rouge-1"]["f"], 4),
        "rouge_1_precision": round(scores["rouge-1"]["p"], 4),
        "rouge_1_recall":    round(scores["rouge-1"]["r"], 4),
        "rouge_2_f1":        round(scores["rouge-2"]["f"], 4),
        "rouge_2_precision": round(scores["rouge-2"]["p"], 4),
        "rouge_2_recall":    round(scores["rouge-2"]["r"], 4),
        "rouge_l_f1":        round(scores["rouge-l"]["f"], 4),
        "rouge_l_precision": round(scores["rouge-l"]["p"], 4),
        "rouge_l_recall":    round(scores["rouge-l"]["r"], 4),
    }


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def calculate_bertscore(
    prediction: str,
    reference: str,
    model_type: str = "xlm-roberta-large",
    lang: str = "en",
    num_layers: int = 17,
) -> tuple[float, float, float]:
    """
    Calcula BERTScore (Precision, Recall, F1) para un único par.

    Args:
        prediction:  Texto generado (transcripción Whisper).
        reference:   Texto de referencia.
        model_type:  Modelo base para BERTScore.
                     - Inglés:  "roberta-large"  (por defecto)
                     - Español: "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
        lang:        Código de idioma ("en" / "es").
        num_layers:  Número de capas del modelo a usar.
                     - roberta-large → 17
                     - roberta-base  → 12

    Returns:
        (precision, recall, f1) redondeados a 4 decimales.
    """
    P, R, F1 = bert_score_fn(
        cands=[prediction],
        refs=[reference],
        model_type=model_type,
        num_layers=num_layers,
        lang=lang,
        rescale_with_baseline=False,
    )
    return round(P.mean().item(), 4), round(R.mean().item(), 4), round(F1.mean().item(), 4)


# ---------------------------------------------------------------------------
# Función principal: calcula todas las métricas de golpe
# ---------------------------------------------------------------------------

def compute_all_metrics(
    audio_stem: str,
    prediction: str,
    reference: str,
    source: Optional[str] = None,
    model_size: str = "",
    bertscore_model: str = "roberta-large",
    bertscore_lang: str = "en",
    bertscore_num_layers: int = 17,
) -> TranscriptMetrics:
    """
    Calcula SARI, BLEU, ROUGE y BERTScore para un par (transcripción, referencia).

    Args:
        audio_stem:   Nombre del audio sin extensión (identificador de fila en el CSV).
        prediction:   Transcripción generada por Whisper.
        reference:    Texto de referencia.
        source:       Texto fuente para SARI. Si es None se usa ``prediction``
                      (SARI medirá solo la similitud con la referencia).
        model_size:   Tamaño del modelo Whisper (tiny, base, medium…).
        bertscore_model:      Modelo para BERTScore (ver calculate_bertscore).
        bertscore_lang:       Idioma para BERTScore.
        bertscore_num_layers: Capas del modelo para BERTScore.

    Returns:
        TranscriptMetrics con todos los campos rellenos.
    """
    m = TranscriptMetrics(audio_stem=audio_stem, model_size=model_size)

    # SARI: si no hay fuente se usa la predicción como proxy
    src = source if source is not None else prediction
    m.sari_add, m.sari_keep, m.sari_del, m.sari_total = calculate_sari(
        source=src,
        prediction=prediction,
        references=[src],
    )

    # BLEU
    bleu = calculate_bleu(prediction, reference)
    m.bleu = bleu["bleu"]
    m.bleu_precision_1 = bleu["bleu_precision_1"]
    m.bleu_precision_2 = bleu["bleu_precision_2"]
    m.bleu_precision_3 = bleu["bleu_precision_3"]
    m.bleu_precision_4 = bleu["bleu_precision_4"]
    m.bleu_brevity_penalty = bleu["bleu_brevity_penalty"]
    m.bleu_length_ratio = bleu["bleu_length_ratio"]
    m.bleu_translation_length = bleu["bleu_translation_length"]
    m.bleu_reference_length = bleu["bleu_reference_length"]

    # ROUGE
    rouge = calculate_rouge(prediction, reference)
    m.rouge_1_f1 = rouge["rouge_1_f1"]
    m.rouge_1_precision = rouge["rouge_1_precision"]
    m.rouge_1_recall = rouge["rouge_1_recall"]
    m.rouge_2_f1 = rouge["rouge_2_f1"]
    m.rouge_2_precision = rouge["rouge_2_precision"]
    m.rouge_2_recall = rouge["rouge_2_recall"]
    m.rouge_l_f1 = rouge["rouge_l_f1"]
    m.rouge_l_precision = rouge["rouge_l_precision"]
    m.rouge_l_recall = rouge["rouge_l_recall"]

    # BERTScore
    m.bertscore_precision, m.bertscore_recall, m.bertscore_f1 = calculate_bertscore(
        prediction=prediction,
        reference=reference,
        model_type=bertscore_model,
        lang=bertscore_lang,
        num_layers=bertscore_num_layers,
    )

    return m


# ---------------------------------------------------------------------------
# Exportar a CSV
# ---------------------------------------------------------------------------

def metrics_to_csv(
    metrics_list: list[TranscriptMetrics],
    output_path: str | Path,
    separator: str = ";",
    decimal: str = ",",
) -> None:
    """
    Guarda una lista de TranscriptMetrics en un CSV.

    Args:
        metrics_list:  Lista de resultados (uno por audio).
        output_path:   Ruta del fichero de salida.
        separator:     Separador de columnas (por defecto ';' — formato europeo).
        decimal:       Separador decimal (por defecto ',' — formato europeo).
    """
    if not metrics_list:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [asdict(m) for m in metrics_list]

    # Convertir separador decimal si es necesario
    if decimal != ".":
        float_fields = {k for k, v in rows[0].items() if isinstance(v, float)}
        for row in rows:
            for f in float_fields:
                if row[f] is not None:
                    row[f] = str(row[f]).replace(".", decimal)

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=separator)
        writer.writeheader()
        writer.writerows(rows)
