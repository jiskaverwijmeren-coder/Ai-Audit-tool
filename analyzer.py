import json
import re
from mistralai import Mistral

def analyseer_iso(client: Mistral, tekst: str, normen=None) -> dict:
    if normen is None:
        normen = ["ISO 9001", "ISO 14001"]
    
    normen_tekst = ", ".join(normen)
    prompt = f"""Je bent een ISO-expert gespecialiseerd in civiele techniek.
Analyseer de volgende tekst op risico's voor {normen_tekst}.
Antwoord ALLEEN in JSON, geen uitleg erbuiten. Gebruik dit formaat:
{{
  "samenvatting": "korte samenvatting van het document",
  "bevindingen": [
    {{
      "norm": "ISO 9001 of ISO 14001 of ISO 45001",
      "clausule": "bijv. 8.5",
      "titel": "korte titel",
      "ernst": "hoog of gemiddeld of laag",
      "beschrijving": "wat is het probleem",
      "aanbeveling": "wat moet er gebeuren",
      "citaat": "kopieer hier een EXACTE zin uit de tekst hierboven die het probleem aantoont"
    }}
  ]
}}
BELANGRIJK voor het citaat-veld:
- Kopieer de zin LETTERLIJK zoals hij in de tekst staat
- Gebruik geen samenvatting of parafrase
- Maximaal 1 zin
TEKST:
{tekst}
"""
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    inhoud = response.choices[0].message.content
    inhoud = re.sub(r"```json|```", "", inhoud).strip()
    return json.loads(inhoud)

def print_rapport(data: dict, bestandsnaam: str):
    print("\n" + "=" * 60)
    print(f"  ISO ANALYSE RAPPORT")
    print(f"  Bestand: {bestandsnaam}")
    print("=" * 60)
    print(f"\n📋 SAMENVATTING:\n{data['samenvatting']}")
    print(f"\n🔍 BEVINDINGEN ({len(data['bevindingen'])} gevonden):")
    print("-" * 60)
    for b in data["bevindingen"]:
        ernst_emoji = {"hoog": "🔴", "gemiddeld": "🟠", "laag": "🟡"}.get(b["ernst"], "⚪")
        print(f"\n{ernst_emoji}  [{b['norm']} | Clausule {b['clausule']}] {b['titel']}")
        print(f"   Ernst:    {b['ernst'].upper()}")
        print(f"   Probleem: {b['beschrijving']}")
        print(f"   Actie:    {b['aanbeveling']}")
    hoog = sum(1 for b in data["bevindingen"] if b["ernst"] == "hoog")
    gemiddeld = sum(1 for b in data["bevindingen"] if b["ernst"] == "gemiddeld")
    laag = sum(1 for b in data["bevindingen"] if b["ernst"] == "laag")
    print("\n" + "=" * 60)
    print(f"  TOTAAL: 🔴 {hoog} hoog  🟠 {gemiddeld} gemiddeld  🟡 {laag} laag")
    print("=" * 60 + "\n")
