REPORT_SECTIONS = [
    "MOTIVO_DE_CONSULTA",
    "RESUMEN_CLINICO",
    "ANAMNESIS",
    "SINTOMAS_REFERIDOS",
    "ANTECEDENTES_PERSONALES",
    "ANTECEDENTES_FAMILIARES",
    "MEDICACION_ACTUAL",
    "ALERGIAS",
    "EXPLORACION_FISICA",
    "PRUEBAS_COMPLEMENTARIAS",
    "VALORACION",
    "PLAN",
    "RED_FLAGS",
]

MEDICAL_PROMPT_TEMPLATE = """Eres un asistente clinico. Convierte la transcripcion medica libre en un informe medico estructurado.

Reglas de salida (obligatorias):
1) Responde SOLO con texto plano (no JSON, no markdown, no bloque de codigo).
2) Usa EXACTAMENTE este formato de secciones y no omitas ninguna:
MOTIVO_DE_CONSULTA:
RESUMEN_CLINICO:
ANAMNESIS:
SINTOMAS_REFERIDOS:
ANTECEDENTES_PERSONALES:
ANTECEDENTES_FAMILIARES:
MEDICACION_ACTUAL:
ALERGIAS:
EXPLORACION_FISICA:
PRUEBAS_COMPLEMENTARIAS:
VALORACION:
PLAN:
RED_FLAGS:
3) Si falta un dato, escribe exactamente: No referido.
4) No inventes diagnosticos, hallazgos o tratamientos no mencionados.
5) Usa lenguaje medico formal, claro y conciso.
6) Mantener idioma: espanol.

Transcripcion:
"""

