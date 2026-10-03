import re


ESG_KEYWORDS = [
    "carbon emissions",
    "co2 emissions",
    "greenhouse gas",
    "ghg",
    "renewable energy",
    "energy efficiency",
    "carbon footprint",
    "sustainability",
    "climate risk",
    "waste management",
    "water conservation",
    "biodiversity"
]


def analyze_esg_text(text):

    if not text:
        return {
            "total_keywords_found": 0,
            "matched_keywords": [],
            "esg_score": 0,
            "assessment": "No text available for ESG analysis"
        }

    text = text.lower()

    matched_keywords = []

    for keyword in ESG_KEYWORDS:

        if re.search(
            r"\b" + re.escape(keyword) + r"\b",
            text
        ):
            matched_keywords.append(keyword)

    # Calculate project-defined ESG keyword score
    esg_score = (
        len(matched_keywords)
        / len(ESG_KEYWORDS)
    ) * 100

    # ESG assessment
    if esg_score >= 75:
        assessment = "Strong ESG keyword coverage"

    elif esg_score >= 50:
        assessment = "Moderate ESG keyword coverage"

    elif esg_score >= 25:
        assessment = "Limited ESG keyword coverage"

    else:
        assessment = "Very limited ESG keyword coverage"

    return {
        "total_keywords_found": len(matched_keywords),
        "matched_keywords": matched_keywords,
        "esg_score": round(esg_score, 2),
        "assessment": assessment
    }