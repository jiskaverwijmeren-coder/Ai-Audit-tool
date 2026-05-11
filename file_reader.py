import pypdf
import os

def laad_bestand(bestandspad: str) -> str:
    extensie = os.path.splitext(bestandspad)[1].lower()

    if extensie == ".txt":
        with open(bestandspad, "r", encoding="utf-8") as f:
            return f.read()

    elif extensie == ".pdf":
        tekst = ""
        with open(bestandspad, "rb") as f:
            reader = pypdf.PdfReader(f)
            for pagina in reader.pages:
                tekst += pagina.extract_text() + "\n"
        return tekst

    else:
        raise ValueError(f"Bestandstype '{extensie}' wordt niet ondersteund. Gebruik .txt of .pdf")